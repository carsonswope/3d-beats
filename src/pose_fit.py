from OpenGL.GL import * 
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import imgui

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *
from engine.texture import GpuTexture

from engine.window import AppBase
from engine.buffer import GpuBuffer

class PoseFitApp(AppBase):
    def __init__(self):
        super().__init__(title="Test-icles", width=1600, height=1250)

        parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
        parser.add_argument('-m', '--model', nargs='?', required=True, type=str, help='Path to .npy model input file')
        parser.add_argument('-d', '--data', nargs='?', required=True, type=str, help='Directory holding data')
        parser.add_argument('--rs_bag', nargs='?', required=True, type=str, help='Path to input realsense .bag file to use')
        parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
        parser.add_argument('--plane_z_threshold', nargs='?', required=True, type=float, help='Z-value threshold in plane coordinates for clipping depth image pixels')
        args = parser.parse_args()

        MODEL_PATH = args.model
        DATASET_PATH = args.data
        RS_BAG = args.rs_bag

        NUM_RANDOM_GUESSES = args.plane_num_iterations or 25000
        self.PLANE_Z_OUTLIER_THRESHOLD = args.plane_z_threshold

        self.calibrated_plane = CalibratedPlane(NUM_RANDOM_GUESSES, self.PLANE_Z_OUTLIER_THRESHOLD)

        print('loading forest')
        self.forest = DecisionForest.load(MODEL_PATH)
        self.data_config = DecisionTreeDatasetConfig(DATASET_PATH)

        print('compiling CUDA kernels..')
        self.decision_tree_evaluator = DecisionTreeEvaluator()
        self.points_ops = PointsOps()

        print('initializing camera..')
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_device_from_file(RS_BAG, repeat_playback=False)
        self.config.enable_stream(rs.stream.depth, rs.format.z16)
        self.config.enable_stream(rs.stream.color, rs.format.rgb8)

        profile = self.pipeline.start(self.config)
        profile.get_device().as_playback().set_real_time(False)
        depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrin = depth_profile.get_intrinsics()

        self.DIM_X = depth_intrin.width
        self.DIM_Y = depth_intrin.height

        self.FOCAL = depth_intrin.fx
        self.PP = np.array([depth_intrin.ppx, depth_intrin.ppy], dtype=np.float32)

        self.pts_gpu = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.float32)

        self.depth_image_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), np.uint16)
        self.labels_image_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.labels_image_rgba_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.NUM_FRAMES = 10
        self.frame_num = 0

        self.depth_images = []

        for i in range(self.NUM_FRAMES):
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            depth_image = np.asanyarray(depth_frame.get_data()).reshape((1, self.DIM_Y, self.DIM_X))
            self.depth_images.append(depth_image)


    def splash(self):
        imgui.text('loading...')

    def tick(self, t):
        
        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data()).reshape((1, self.DIM_Y, self.DIM_X))
        self.depth_image_gpu.cu().set(self.depth_images[self.frame_num])

        grid_dim = (1, (self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1)
        block_dim = (1,32,32)

        # convert depth image to points
        self.points_ops.deproject_points(
            np.array([1, self.DIM_X, self.DIM_Y, -1], dtype=np.int32),
            self.PP,
            np.float32(self.FOCAL),
            self.depth_image_gpu.cu(),
            self.pts_gpu.cu(),
            grid=grid_dim,
            block=block_dim)

        if not self.calibrated_plane.is_set():
            self.calibrated_plane.make(self.pts_gpu, (self.DIM_X, self.DIM_Y))

        # every point..
        grid_dim2 = (((self.DIM_X * self.DIM_Y) // 1024) + 1, 1, 1)
        block_dim2 = (1024, 1, 1)

        self.points_ops.transform_points(
            np.int32(self.DIM_X * self.DIM_Y),
            self.pts_gpu.cu(),
            self.calibrated_plane.get_mat(),
            grid=grid_dim2,
            block=block_dim2)

        self.calibrated_plane.filter_points_by_plane(
            np.int32(self.DIM_X * self.DIM_Y),
            np.float32(self.PLANE_Z_OUTLIER_THRESHOLD),
            self.pts_gpu.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.points_ops.setup_depth_image_for_forest(
            np.int32(self.DIM_X * self.DIM_Y),
            self.pts_gpu.cu(),
            self.depth_image_gpu.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.labels_image_gpu.cu().fill(np.uint16(65535))
        self.decision_tree_evaluator.get_labels_forest(self.forest, self.depth_image_gpu.cu(), self.labels_image_gpu.cu())

        # unmap from cuda.. isn't actually necessary, but just to make sure..
        self.depth_image_gpu.gl()

        labels_image_cpu = self.labels_image_gpu.cu().get()
        labels_image_cpu_rgba = self.data_config.convert_ids_to_colors(labels_image_cpu).reshape((480, 848, 4))

        # generate mesh for 3d rendering!
        


        self.labels_image_rgba_tex.set(labels_image_cpu_rgba)

        _, self.frame_num = imgui.slider_int("f", self.frame_num, 0, self.NUM_FRAMES - 1)
        imgui.image(self.labels_image_rgba_tex.gl(), self.DIM_X * 2, self.DIM_Y * 2)


if __name__ == '__main__':
    a = PoseFitApp()
    a.run()
