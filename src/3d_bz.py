from OpenGL.GL import * 

import pyrealsense2 as rs
import numpy as np

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *
from cuda.mean_shift import *

from engine.texture import GpuTexture
from engine.window import AppBase, run_app
from engine.buffer import GpuBuffer
from engine.profile_timer import ProfileTimer
from engine.midi import Midi

import cuda.py_nvcc_utils as py_nvcc_utils
import rs_util

from util import make_grid, MAX_UINT16

from cpp_grouping import CppGrouping

from hand_state import HandState

import argparse

import imgui

class App_3d_bz(AppBase):
    def __init__(self):

        width = 1920
        height = 1080

        super().__init__(title="3d-beats", width=width, height=height)

        self.t = ProfileTimer()

        parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
        parser.add_argument('-cfg', nargs='?', required=True, type=str, help='Path to the layered decision forest config file')
        parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
        py_nvcc_utils.add_args(parser)
        rs_util.add_args(parser)
        args = parser.parse_args()

        py_nvcc_utils.config_compiler(args)

        self.midi = Midi()

        self.NUM_RANDOM_GUESSES = args.plane_num_iterations or 25000
        self.PLANE_Z_OUTLIER_THRESHOLD = 40.

        self.gauss_sigma = 2.5
        self.z_thresh_offset = 25.
        self.min_velocity = 5.
        
        self.group_min_size = 0.06

        self.mean_shift_rounds = 4

        self.calibrated_plane = CalibratedPlane(self.NUM_RANDOM_GUESSES, self.PLANE_Z_OUTLIER_THRESHOLD)
        self.calibrate_next_frame = False

        print('loading forest')

        self.layered_rdf = LayeredDecisionForest.load(args.cfg, (480, 848))

        self.points_ops = PointsOps()

        self.mean_shift = MeanShift()

        print('initializing camera..')
        self.pipeline, self.depth_intrin, self.DIM_X, self.DIM_Y, self.FOCAL, self.PP = rs_util.start_stream(args)

        self.pts_cu = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.float32)
        self.depth_image = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.depth_image_2 = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.depth_image_group = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.depth_image_rgba = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        self.depth_image_rgba_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.depth_mm_level = 3
        self.depth_mm_dims = (self.DIM_Y // (1<<self.depth_mm_level), self.DIM_X // (1<<self.depth_mm_level))
        self.depth_image_mm = GpuBuffer(self.depth_mm_dims, dtype=np.uint16)
        self.depth_image_mm_groups = GpuBuffer(self.depth_mm_dims, dtype=np.uint16)
        self.depth_image_mm_groups_2 = GpuBuffer(self.depth_mm_dims, dtype=np.uint16)
        self.depth_image_mm_rgba = GpuBuffer(self.depth_mm_dims + (4,), dtype=np.uint8)
        self.depth_image_mm_rgba_tex = GpuTexture((self.depth_mm_dims[1], self.depth_mm_dims[0]), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.coord_croups_cpu = np.zeros((self.depth_mm_dims[0] * self.depth_mm_dims[1], 3), dtype=np.int32)
        self.coord_groups_gpu = GpuBuffer((self.depth_mm_dims[0] * self.depth_mm_dims[1], 3), dtype=np.int32)

        self.labels_image = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.labels_image_2 = GpuBuffer((self.DIM_Y, self.DIM_X), dtype=np.uint16)
        self.labels_image_rgba_cpu = np.zeros((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        self.labels_image_rgba = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        self.labels_image_rgba_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        mean_shift_variances = np.array(
            [100., 40., 60., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.],
            dtype=np.float32)
        self.mean_shift_variances = cu_array.to_gpu(mean_shift_variances)

        self.fingertip_idxes = [2, 5, 8, 11, 14]
        self.DEFAULT_FINGERTIP_THRESHOLDS = [240., 140., 140., 140., 140.]
        # (z_thresh, midi_note)
        init_hand_state = lambda n: [(self.DEFAULT_FINGERTIP_THRESHOLDS[i], n+i) for i in range(len(self.fingertip_idxes))]

        on_fn = lambda n,v: self.midi.send([0x90, n, v])
        off_fn = lambda n: self.midi.send([0x80, n, 0])

        # fingers: 0 = index, 1 = middle, 2 = ring, 3 = pinky
        self.hand_states = [
            HandState(init_hand_state(36), on_fn, off_fn),
            HandState(init_hand_state(41), on_fn, off_fn)]

        self.frame_num = 0

    def splash(self):
        self.begin_imgui_main()
        imgui.text('loading...')
        imgui.end()

    def tick(self, _):

        frames = self.pipeline.wait_for_frames()

        # start frame timer after we get the depth frame.
        # we really only care about how long it takes to process it
        self.t.record('initial processing')

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            imgui.text('no depth frame!')
            return

        # let camera stabilize for a few frames
        elif self.frame_num < 15:
            self.begin_imgui_main()
            imgui.text('loading... ...')
            imgui.end()
            self.frame_num += 1
            return

        # Convert images to numpy arrays
        self.depth_image_cpu = np.asanyarray(depth_frame.get_data())
        self.depth_image.cu().set(self.depth_image_cpu)

        block_dim = (1,32,32)
        grid_dim = make_grid((1, self.DIM_X, self.DIM_Y), block_dim)

        # convert depth image to points
        self.points_ops.deproject_points(
            np.array([1, self.DIM_X, self.DIM_Y, -1], dtype=np.int32),
            self.PP,
            np.float32(self.FOCAL),
            self.depth_image.cu(),
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
            self.depth_image.cu(),
            grid=grid_dim2,
            block=block_dim2)

        if self.gauss_sigma > 0.1:
            self.depth_image_2.cu().set(self.depth_image.cu())
            self.points_ops.gaussian_depth_filter(
                self.depth_image_2,
                self.depth_image,
                sigma=self.gauss_sigma,
                k_size=11)

        # make smaller depth image. faster to copy and process cpu-side
        self.points_ops.shrink_image(
            np.array((self.DIM_X, self.DIM_Y), dtype=np.int32),
            np.int32(self.depth_mm_level),
            self.depth_image.cu(),
            self.depth_image_mm.cu(),
            grid=make_grid((self.depth_mm_dims[1], self.depth_mm_dims[0], 1), (32, 32, 1)),
            block=(32, 32, 1))

        self.cu_ctx.synchronize()

        self.t.record('copy to CPU')

        depth_image_mm3 = self.depth_image_mm.cu().get()

        self.t.record('make pixel groups')

        g_info = np.zeros((2, 3), dtype=np.float32)
        CppGrouping().make_groups(depth_image_mm3, self.coord_croups_cpu, g_info, self.group_min_size)
        r_group_size = np.int32(g_info[0, 0])
        l_group_size = np.int32(g_info[1, 0])
        groups_size = r_group_size + l_group_size

        self.t.record('copy pixel groups to image')


        if groups_size > 0:

            self.depth_image_mm_groups_2.cu().fill(0)

            self.coord_groups_gpu.cu()[0:groups_size,:].set(self.coord_croups_cpu[0:groups_size])

            self.points_ops.write_pixel_groups_to_stencil_image(
                self.coord_groups_gpu.cu(),
                np.int32(groups_size),
                self.depth_image_mm_groups_2.cu(),
                np.array(self.depth_mm_dims, dtype=np.int32),
                grid=make_grid((int(groups_size), 1, 1), (32, 1, 1)),
                block=(32, 1, 1))

            # any reason to grow twice?
            self.points_ops.grow_groups(
                np.array([self.depth_mm_dims[1], self.depth_mm_dims[0]], dtype=np.int32),
                self.depth_image_mm_groups_2.cu(),
                self.depth_image_mm_groups.cu(),
                grid=make_grid((self.depth_mm_dims[1], self.depth_mm_dims[0], 1), (32, 32, 1)),
                block=(32, 32, 1))
            
        else:

            self.depth_image_mm_groups.cu().fill(0)

        
        self.points_ops.make_depth_rgba(
            np.array([self.depth_mm_dims[1], self.depth_mm_dims[0]], dtype=np.int32),
            np.uint16(0),
            np.uint16(2),
            self.depth_image_mm_groups.cu(),
            self.depth_image_mm_rgba.cu(),
            grid=make_grid((self.depth_mm_dims[1], self.depth_mm_dims[0], 1), (32, 32, 1)),
            block=(32, 32, 1))
        self.depth_image_mm_rgba_tex.copy_from_gpu_buffer(self.depth_image_mm_rgba)

        
        # generate RGB image for debugging!
        self.labels_image_rgba.cu().fill(np.uint8(0))

        self.t.record('per-hand pipeline 1')
        self.run_per_hand_pipeline(1, False)

        self.t.record('per-hand pipeline 2')
        self.run_per_hand_pipeline(2, True)

        self.labels_image_rgba_tex.copy_from_gpu_buffer(self.labels_image_rgba)

        self.t.record('imgui')

        self.begin_imgui_main()

        imgui.text('running!')

        # per finger calibration
        if imgui.button('reset fingertip calibration'):
            for h in self.hand_states:
                for f, t in zip(h.fingertips, self.DEFAULT_FINGERTIP_THRESHOLDS):
                    f.z_thresh = t

        self.calibrate_next_frame = imgui.button('recalibrate plane')

        _, self.PLANE_Z_OUTLIER_THRESHOLD = imgui.slider_float('plane threshold', self.PLANE_Z_OUTLIER_THRESHOLD, 0., 100.)

        _, self.z_thresh_offset = imgui.slider_float('fingertip z threshold offset', self.z_thresh_offset, 0., 100.)

        _, self.min_velocity = imgui.slider_float('min velocity', self.min_velocity, 0., 100.)

        if imgui.tree_node('R'):
            self.hand_states[0].draw_imgui(self.z_thresh_offset)
            imgui.tree_pop()

        if imgui.tree_node('L'):
            self.hand_states[1].draw_imgui(self.z_thresh_offset)
            imgui.tree_pop()

        imgui.text(':)')

        self.midi.draw_imgui()

        # imgui.image(self.depth_image_mm_rgba_tex.gl(), self.DIM_X / 2., self.DIM_Y / 2.)
        # imgui.image(self.depth_image_rgba_gpu_tex.gl(), self.DIM_X, self.DIM_Y)

        imgui.image(self.labels_image_rgba_tex.gl(), self.DIM_X * self.dpi_scale, self.DIM_Y * self.dpi_scale)

        imgui.end()

        times = self.t.render()
        imgui.begin('profile timer')
        for t in times:
            imgui.text(t)

        imgui.plot_lines('ms per frame',
            np.array(self.ms_per_frame_log, dtype=np.float32),
            scale_max=100.,
            scale_min=0.,
            graph_size=(300,200))
        
        imgui.plot_lines('FPS',
            1000. / np.array(self.ms_per_frame_log, dtype=np.float32),
            scale_max=100.,
            scale_min=0.,
            graph_size=(300,200))

        imgui.end()

        self.frame_num += 1


    def run_per_hand_pipeline(self, g_id, flip_x):

        # self.cu_ctx.synchronize()
        # self.t.record('--preprocessing')

        self.depth_image_group.cu().fill(0)

        self.points_ops.stencil_depth_image_by_group(
            np.array([self.DIM_X, self.DIM_Y], dtype=np.int32),
            np.int32(self.depth_mm_level),
            np.int32(g_id),
            self.depth_image_mm_groups.cu(),
            self.depth_image.cu(),
            self.depth_image_group.cu(),
            grid=make_grid((self.DIM_X, self.DIM_Y, 1), (32, 32, 1)),
            block=(32, 32, 1))

        if flip_x:
            self.points_ops.flip_x(
                np.array([self.DIM_X, self.DIM_Y], dtype=np.int32),
                self.depth_image_group.cu(),
                self.depth_image_2.cu(),
                grid=make_grid((self.DIM_X, self.DIM_Y, 1), (32, 32, 1)),
                block=(32, 32, 1))
        else:
            self.depth_image_2.cu().set(self.depth_image_group.cu())

        self.points_ops.convert_0s_to_maxuint(
            np.int32(self.DIM_X * self.DIM_Y),
            self.depth_image_2.cu(),
            grid=make_grid((self.DIM_X * self.DIM_Y, 1, 1), (1024, 1, 1)),
            block=(1024, 1, 1))

        """
        self.points_ops.make_depth_rgba(
            np.array((self.DIM_X, self.DIM_Y), dtype=np.int32),
            np.uint16(2000),
            np.uint16(6000),
            self.depth_image_cu_2.cu(),
            self.depth_image_rgba_gpu.cu(),
            grid=make_grid((self.DIM_X, self.DIM_Y, 1), (32, 32, 1)),
            block=(32, 32, 1))
        self.depth_image_rgba_gpu_tex.copy_from_gpu_buffer(self.depth_image_rgba_gpu)
        """

        # self.cu_ctx.synchronize()
        # self.t.record('--evals')

        self.layered_rdf.run(self.depth_image_2, self.labels_image)

        if flip_x:
            self.labels_image_2.cu().set(self.labels_image.cu())
            self.points_ops.flip_x(
                np.array([self.DIM_X, self.DIM_Y], dtype=np.int32),
                self.labels_image_2.cu(),
                self.labels_image.cu(),
                grid=make_grid((self.DIM_X, self.DIM_Y, 1), (32, 32, 1)),
                block=(32, 32, 1))

        self.points_ops.make_rgba_from_labels(
            np.uint32(self.DIM_X),
            np.uint32(self.DIM_Y),
            np.uint32(self.layered_rdf.num_layered_classes),
            self.labels_image.cu(),
            self.layered_rdf.label_colors.cu(),
            self.labels_image_rgba.cu(),
            grid = ((self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1, 1),
            block = (32,32,1))

        # self.cu_ctx.synchronize()
        # self.t.record('--mean shift')

        label_means = self.mean_shift.run(
            self.mean_shift_rounds,
            self.labels_image.cu().reshape((1, self.DIM_Y, self.DIM_X)),
            self.layered_rdf.num_layered_classes,
            self.mean_shift_variances)

        # self.cu_ctx.synchronize()
        # self.t.record('--rgba')

        """

        self.cu_ctx.synchronize()
        self.t.record('--more rgba')

        # TODO: this is slow! too much copying back and forth on CPU
        # add labels to debug rgba image to show where calculated fingertips are
        self.labels_image_rgba_cu.cu().get(self.labels_image_rgba_cpu)

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

        self.labels_image_rgba_cu.cu().set(self.labels_image_rgba_cpu)
        
        """
        # self.cu_ctx.synchronize()
        # self.t.record('--pts analysis')

        if flip_x:
            # left hand
            hand_state = self.hand_states[1]
        else:
            # right hand
            hand_state = self.hand_states[0]

        for i, f_idx in zip(range(len(self.fingertip_idxes)), self.fingertip_idxes):

            px, py = label_means[f_idx-1].astype(np.int32)
            if px < 0 or py < 0 or px >= self.DIM_X or py >= self.DIM_Y:
                hand_state.fingertips[i].reset_positions()
            else:
                # z value for depth.
                # look up in original depth image, convert to plane space, get -z coordinate
                z = self.depth_image_cpu[py, px]
                pt = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [px, py], z)
                pt.append(1.)
                pt = self.calibrated_plane.plane @ pt
                pt_z = -pt[2]
                hand_state.fingertips[i].next_z_pos(pt_z, self.z_thresh_offset, self.min_velocity)

if __name__ == '__main__':
    run_app(App_3d_bz)
