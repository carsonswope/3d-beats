from OpenGL.GL import * 

import pyrealsense2 as rs
import numpy as np
import cv2
import math

import pycuda.driver as cu

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *
from cuda.mean_shift import *
from engine.texture import GpuTexture

from util import make_grid, MAX_UINT16

import argparse

import imgui

import rtmidi

from engine.window import AppBase
from engine.buffer import GpuBuffer

class RunLiveMultiApp(AppBase):
    def __init__(self):
        super().__init__(title="Test-icles", width=1920, height=1500)

        parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
        parser.add_argument('-m0', '--model0', nargs='?', required=True, type=str, help='Path to .npy model input file for arm/hand/fingers/thumb')
        parser.add_argument('-m1', '--model1', nargs='?', required=True, type=str, help='Path to .npy model input file for bones 1-2-3 on fingers')
        parser.add_argument('-m2', '--model2', nargs='?', required=True, type=str, help='Path to .npy model input file for fingers 1-2-3-4')
        # parser.add_argument('-d', '--data', nargs='?', required=True, type=str, help='Directory holding data')
        parser.add_argument('--rs_bag', nargs='?', required=False, type=str, help='Path to optional input realsense .bag file to use instead of live camera stream')
        parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
        # parser.add_argument('--plane_z_threshold', nargs='?', required=True, type=float, help='Z-value threshold in plane coordinates for clipping depth image pixels')
        args = parser.parse_args()


        self.midi_out = rtmidi.MidiOut()
        # available_ports = self.midi_out.get_ports()
        self.midi_out.open_port(1) # loopbe port..

        self.note_on = lambda n,v: [0x90, n, v]
        self.note_off = lambda n: [0x80, n, 0]

        RS_BAG = args.rs_bag

        self.NUM_RANDOM_GUESSES = args.plane_num_iterations or 25000
        self.PLANE_Z_OUTLIER_THRESHOLD = 40.

        self.calibrated_plane = CalibratedPlane(self.NUM_RANDOM_GUESSES, self.PLANE_Z_OUTLIER_THRESHOLD)
        self.calibrate_next_frame = False

        print('loading forest')
        self.m0 = DecisionForest.load(args.model0)
        self.m1 = DecisionForest.load(args.model1)
        self.m2 = DecisionForest.load(args.model2)

        print('compiling CUDA kernels..')
        self.decision_tree_evaluator = DecisionTreeEvaluator()
        self.points_ops = PointsOps()

        print('initializing camera..')
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if RS_BAG:
            self.config.enable_device_from_file(RS_BAG, repeat_playback=True)
            self.config.enable_stream(rs.stream.depth, rs.format.z16)
            self.config.enable_stream(rs.stream.color, rs.format.rgb8)

        else:
            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = self.config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_config_json = open('hand_config.json', 'r').read()
            rs.rs400_advanced_mode(device).load_json(device_config_json)
            device.first_depth_sensor().set_option(rs.option.depth_units, 0.0001)
            self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)

        profile = self.pipeline.start(self.config)
        if RS_BAG:
            profile.get_device().as_playback().set_real_time(False)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrin = depth_profile.get_intrinsics()

        self.DIM_X = depth_intrin.width
        self.DIM_Y = depth_intrin.height

        b = (1024, 1, 1)
        g = make_grid((self.DIM_X * self.DIM_Y, 1, 1), b)

        self.FOCAL = depth_intrin.fx
        self.PP = np.array([depth_intrin.ppx, depth_intrin.ppy], dtype=np.float32)
        self.pts_cu = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.float32)
        self.depth_image_cu = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.depth_image_cu_2 = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.labels_image0_cu = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.labels_image1_cu = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.labels_image2_cu = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.labels_image_composite_cu = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.depth_image_rgba_gpu = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        self.depth_image_rgba_gpu_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.labels_image_rgba_cpu = np.zeros((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        self.labels_image_rgba_cu = cu_array.to_gpu(self.labels_image_rgba_cpu)

        self.mean_shift = MeanShift()

        self.labels_images_ptrs = cu.pagelocked_zeros((3,), dtype=np.int64)
        self.labels_images_ptrs[0] = self.labels_image0_cu.cu().__cuda_array_interface__['data'][0]
        self.labels_images_ptrs[1] = self.labels_image2_cu.cu().__cuda_array_interface__['data'][0]
        self.labels_images_ptrs[2] = self.labels_image1_cu.cu().__cuda_array_interface__['data'][0]
        self.labels_images_ptrs_cu = cu_array.to_gpu(self.labels_images_ptrs)

        mean_shift_variances = np.array([
            100.,
            40.,
            60.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.,
            50.], dtype=np.float32)
        self.mean_shift_variances_cu = cu_array.to_gpu(mean_shift_variances)

        # encoded instructions for making composite labels image using all generated labels images
        # essentially a mini decision-tree
        labels_conditions = np.array([
            # img 1
            [0, 1], # if label 1, ID 1
            [0, 2], # if label 2, ID 2
            [1, 4], # if label 3, look at next img. root of that tree @ IDX 4
            [0, 3], # if label 4, ID 3
            # img 1 == 3 , img 2. keep looking, this determines which finger
            [1, 8],
            [1, 11],
            [1, 14],
            [1, 17],
            # next 4 branches determines which knuckle!
            # img 1 == 3, img 2 == 1, img 3
            [0, 4],
            [0, 5],
            [0, 6],
            # img 1 == 3, img 2 == 2, img 3
            [0, 7],
            [0, 8],
            [0, 9],
            # img 1 == 3, img 2 == 3, img 3
            [0, 10],
            [0, 11],
            [0, 12],
            # img 1 == 3, img 2 == 4, img 3
            [0, 13],
            [0, 14],
            [0, 15],
        ], dtype=np.int32)
        self.labels_conditions_cu = cu_array.to_gpu(labels_conditions)
        self.NUM_COMPOSITE_CLASSES = 15 # 1-15

        self.fingertip_idxes = [5, 8, 11, 14]
        self.fingertip_thresholds = {
            5: 140.,
            8: 140.,
            11: 140.,
            14: 120.,
        }
        self.fingertip_notes = {
            5: 36,
            8: 37,
            11: 38,
            14: 39,
        }
        self.fingertip_states = {i:False for i in self.fingertip_idxes} # start off..
        self.FINGERTIP_UP_THRESHOLD = 20.
        self.MAX_FINGERTIP_POSITIONS = 50
        self.fingertip_positions = {i:[] for i in self.fingertip_idxes}

        labels_colors = np.array([
            # arm
            [68, 128, 137, 255],
            # thumb
            [174, 45, 244, 255],
            # hand
            [214, 244, 40, 255],
            # index
            [255, 0, 0, 255], [255, 150, 150, 255], [120, 0, 0, 255],
            # middle
            [0, 255, 0, 255], [150, 255, 150, 255], [0, 120, 0, 255],
            # ring
            [0, 0, 255, 255], [170, 170, 255, 255], [0, 0, 120, 255],
            # pinky
            [255, 140, 5, 255], [255, 188, 104, 255], [168, 94, 0, 255]
        ], dtype=np.uint8)
        self.labels_colors_cu = cu_array.to_gpu(labels_colors)

        self.labels_image_rgba_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.frame_num = 0

        self.gauss_sigma = 2.5


    def set_fingertip_state(self, f_idx, s):
        note = self.fingertip_notes[f_idx]
        if s and not self.fingertip_states[f_idx]:
            self.fingertip_states[f_idx] = True
            self.midi_out.send_message(self.note_on(note, 127))
        
        elif not s and self.fingertip_states[f_idx]:
            self.fingertip_states[f_idx] = False
            self.midi_out.send_message(self.note_off(note))


    def splash(self):
        imgui.text('loading...')
    
    def tick(self, t):

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            imgui.text('no depth frame!')
            return

        # let camera stabilize for a few frames
        elif self.frame_num < 15:
            imgui.text('loading... ...')
            self.frame_num += 1
            return

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        self.depth_image_cu.cu().set(depth_image)

        block_dim = (1,32,32)
        grid_dim = make_grid((1, self.DIM_X, self.DIM_Y), block_dim)

        # convert depth image to points
        self.points_ops.deproject_points(
            np.array([1, self.DIM_X, self.DIM_Y, -1], dtype=np.int32),
            self.PP,
            np.float32(self.FOCAL),
            self.depth_image_cu.cu(),
            self.pts_cu.cu(),
            grid=grid_dim,
            block=block_dim)

        if not self.calibrated_plane.is_set() or self.calibrate_next_frame:
            if self.calibrated_plane.is_set():
                # attempt to improve plane..
                self.calibrated_plane.make(self.pts_cu, (self.DIM_X, self.DIM_Y), self.calibrated_plane.get_mat())
            else:
                # initialize plane for first time
                self.calibrated_plane.make(self.pts_cu, (self.DIM_X, self.DIM_Y))

        # every point..
        block_dim2 = (1024, 1, 1)
        grid_dim2 = make_grid((self.DIM_X * self.DIM_Y, 1, 1), block_dim2)

        self.points_ops.transform_points(
            np.int32(self.DIM_X * self.DIM_Y),
            self.pts_cu.cu(),
            self.calibrated_plane.get_mat(),
            grid=grid_dim2,
            block=block_dim2)

        self.calibrated_plane.filter_points_by_plane(
            np.int32(self.DIM_X * self.DIM_Y),
            np.float32(self.PLANE_Z_OUTLIER_THRESHOLD),
            self.pts_cu.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.points_ops.remove_missing_3d_points_from_depth_image(
            np.int32(self.DIM_X * self.DIM_Y),
            self.pts_cu.cu(),
            self.depth_image_cu.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.depth_image_cu_2.cu().set(self.depth_image_cu.cu())
        self.points_ops.gaussian_depth_filter(
            self.depth_image_cu_2,
            self.depth_image_cu,
            sigma=self.gauss_sigma,
            k_size=25)

        self.points_ops.convert_0s_to_maxuint(
            np.int32(self.DIM_X * self.DIM_Y),
            self.depth_image_cu.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.points_ops.make_depth_rgba(
            np.array((self.DIM_X, self.DIM_Y), dtype=np.int32),
            np.uint16(2000),
            np.uint16(6000),
            self.depth_image_cu.cu(),
            self.depth_image_rgba_gpu.cu(),
            grid=make_grid((self.DIM_X, self.DIM_Y, 1), (32, 32, 1)),
            block=(32, 32, 1))
        self.depth_image_rgba_gpu_tex.copy_from_gpu_buffer(self.depth_image_rgba_gpu)

        self.labels_image0_cu.cu().fill(MAX_UINT16)
        self.labels_image1_cu.cu().fill(MAX_UINT16)
        self.labels_image2_cu.cu().fill(MAX_UINT16)
        self.labels_image_composite_cu.cu().fill(MAX_UINT16)

        depth_image_cu_reshaped = self.depth_image_cu.cu().reshape((1, self.DIM_Y, self.DIM_X))
        self.decision_tree_evaluator.get_labels_forest(self.m0, depth_image_cu_reshaped, self.labels_image0_cu.cu().reshape((1, self.DIM_Y, self.DIM_X)))
        self.decision_tree_evaluator.get_labels_forest(self.m1, depth_image_cu_reshaped, self.labels_image1_cu.cu().reshape((1, self.DIM_Y, self.DIM_X)))
        self.decision_tree_evaluator.get_labels_forest(self.m2, depth_image_cu_reshaped, self.labels_image2_cu.cu().reshape((1, self.DIM_Y, self.DIM_X)))

        self.mean_shift.make_composite_labels_image(
            self.labels_images_ptrs_cu,
            self.DIM_X,
            self.DIM_Y,
            self.labels_conditions_cu,
            self.labels_image_composite_cu.cu().reshape((1, self.DIM_Y, self.DIM_X)))
        
        label_means = self.mean_shift.run(
            6,
            self.labels_image_composite_cu.cu().reshape((1, self.DIM_Y, self.DIM_X)),
            self.NUM_COMPOSITE_CLASSES,
            self.mean_shift_variances_cu)

        # generate RGB image for debugging!
        self.labels_image_rgba_cu.fill(np.uint8(0))
        self.points_ops.make_rgba_from_labels(
            np.uint32(self.DIM_X),
            np.uint32(self.DIM_Y),
            np.uint32(self.NUM_COMPOSITE_CLASSES),
            self.labels_image_composite_cu.cu(),
            self.labels_colors_cu,
            self.labels_image_rgba_cu,
            grid = ((self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1, 1),
            block = (32,32,1))

        self.labels_image_rgba_cu.get(self.labels_image_rgba_cpu)

        for m in label_means:
            if not math.isnan(m[0]):
                my = int(m[1])
                mx = int(m[0])
                if my > 0 and my < self.DIM_Y - 1 and mx > 0 and mx < self.DIM_X - 1:
                    self.labels_image_rgba_cpu[my, mx, :] = np.array([255, 255, 255, 255], dtype=np.uint8)
                    self.labels_image_rgba_cpu[my+1, mx, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
                    self.labels_image_rgba_cpu[my-1, mx, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
                    self.labels_image_rgba_cpu[my, mx+1, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
                    self.labels_image_rgba_cpu[my, mx-1, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
        self.labels_image_rgba_tex.set(self.labels_image_rgba_cpu)

        pts_cpu = self.pts_cu.cu().get()

        for f_idx in self.fingertip_idxes:
            px, py = label_means[f_idx-1].astype(np.int)
            if px < 0 or py < 0 or px >= self.DIM_X or py >= self.DIM_Y:
                self.fingertip_positions[f_idx].clear()
                self.set_fingertip_state(f_idx, False)
            else:
                d = pts_cpu[py, px]
                d_z = -d[2]
                self.fingertip_positions[f_idx].append(d_z)
                while len(self.fingertip_positions[f_idx]) > self.MAX_FINGERTIP_POSITIONS:
                    self.fingertip_positions[f_idx].pop(0)
            
                if len(self.fingertip_positions[f_idx]) > 10: # arbitrary..
                    last_pos = self.fingertip_positions[f_idx][-1]
                    if last_pos < self.fingertip_thresholds[f_idx]:
                        self.set_fingertip_state(f_idx, True)
                    else:
                        self.set_fingertip_state(f_idx, False)

        imgui.text('running!')

        if imgui.button('recalibrate'):
            self.calibrate_next_frame = True
        else:
            self.calibrate_next_frame = False

        _, self.PLANE_Z_OUTLIER_THRESHOLD = imgui.slider_float('z thresh', self.PLANE_Z_OUTLIER_THRESHOLD, 0., 100.)

        c_x, c_y = imgui.get_cursor_pos()
        graph_dim_x = 400.
        graph_pad_x = 10.
        graph_dim_y = 150.
        graph_scale_z = 500.

        for i, f_idx in zip(range(len(self.fingertip_idxes)), self.fingertip_idxes):

            imgui.set_cursor_pos((c_x + ((graph_dim_x + graph_pad_x) * i), c_y))

            if len(self.fingertip_positions[f_idx]) > 0:
                a = np.array(self.fingertip_positions[f_idx], dtype=np.float32)
            else:
                a = np.array([0], dtype=np.float32)


            cursor_pos = imgui.get_cursor_screen_pos()

            imgui.plot_lines(f'##f{f_idx} pos',
                a,
                scale_max=graph_scale_z,
                scale_min=0.,
                graph_size=(graph_dim_x, graph_dim_y))

            f_threshold = self.fingertip_thresholds[f_idx]

            if self.fingertip_states[f_idx]:
                thresh_color = imgui.get_color_u32_rgba(0.3,1,0.8,0.30)
            else:
                thresh_color = imgui.get_color_u32_rgba(0.3,1,0.8,0.05)

            imgui.get_window_draw_list().add_rect_filled(
                cursor_pos[0],
                cursor_pos[1] + (graph_dim_y * (1 - (f_threshold / graph_scale_z))),
                cursor_pos[0] + graph_dim_x,
                cursor_pos[1] + graph_dim_y,
                thresh_color)

        _, self.gauss_sigma = imgui.slider_float('depth sgma', self.gauss_sigma, 0.1, 10.)

        imgui.image(self.depth_image_rgba_gpu_tex.gl(), self.DIM_X, self.DIM_Y)

        imgui.image(self.labels_image_rgba_tex.gl(), self.DIM_X * 1.5, self.DIM_Y * 1.5)

        self.frame_num += 1


if __name__ == '__main__':
    a = RunLiveMultiApp()
    a.run()
