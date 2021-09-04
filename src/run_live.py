from OpenGL.GL import *
import numpy as np
import argparse
import imgui

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *
from engine.texture import GpuTexture
import cuda.py_nvcc_utils as py_nvcc_utils
import rs_util

from engine.window import AppBase, run_app
from engine.buffer import GpuBuffer

class RunLiveApp(AppBase):
    def __init__(self):
        super().__init__(title="RDF Demo")

        parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
        parser.add_argument('-m', '--model', nargs='?', required=True, type=str, help='Path to .npy model input file')
        parser.add_argument('-d', '--data', nargs='?', required=True, type=str, help='Directory holding data')
        parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
        parser.add_argument('--plane_z_threshold', nargs='?', required=True, type=float, help='Z-value threshold in plane coordinates for clipping depth image pixels')
        py_nvcc_utils.add_args(parser)
        rs_util.add_args(parser)
        args = parser.parse_args()

        py_nvcc_utils.config_compiler(args)

        MODEL_PATH = args.model
        DATASET_PATH = args.data

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
        self.pipeline, self.depth_intrin, self.DIM_X, self.DIM_Y, self.FOCAL, self.PP = rs_util.start_stream(args)

        self.pts = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.float32)

        self.depth_image = GpuBuffer((1, self.DIM_Y, self.DIM_X), np.uint16)
        self.labels_image = GpuBuffer((1, self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.labels_image_rgba_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.frame_num = 0

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
        depth_image = np.asanyarray(depth_frame.get_data()).reshape((1, self.DIM_Y, self.DIM_X))
        self.depth_image.cu().set(depth_image)

        grid_dim = (1, (self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1)
        block_dim = (1,32,32)

        # convert depth image to points
        self.points_ops.deproject_points(
            np.array([1, self.DIM_X, self.DIM_Y, -1], dtype=np.int32),
            self.PP,
            np.float32(self.FOCAL),
            self.depth_image.cu(),
            self.pts.cu(),
            grid=grid_dim,
            block=block_dim)

        if not self.calibrated_plane.is_set():
            self.calibrated_plane.make(self.pts, (self.DIM_X, self.DIM_Y))

        # every point..
        grid_dim2 = (((self.DIM_X * self.DIM_Y) // 1024) + 1, 1, 1)
        block_dim2 = (1024, 1, 1)

        self.points_ops.transform_points(
            np.int32(self.DIM_X * self.DIM_Y),
            self.pts.cu(),
            self.calibrated_plane.get_mat(),
            grid=grid_dim2,
            block=block_dim2)

        self.calibrated_plane.filter_points_by_plane(
            np.int32(self.DIM_X * self.DIM_Y),
            np.float32(self.PLANE_Z_OUTLIER_THRESHOLD),
            self.pts.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.points_ops.setup_depth_image_for_forest(
            np.int32(self.DIM_X * self.DIM_Y),
            self.pts.cu(),
            self.depth_image.cu(),
            grid=grid_dim2,
            block=block_dim2)

        self.labels_image.cu().fill(np.uint16(65535))
        self.decision_tree_evaluator.get_labels_forest(self.forest, self.depth_image.cu(), self.labels_image.cu())

        # unmap from cuda.. isn't actually necessary, but just to make sure..
        self.depth_image.gl()

        # final steps: these are slow.
        # can be polished if/when necessary
        labels_image_cpu = self.labels_image.cu().get()
        labels_image_cpu_rgba = self.data_config.convert_ids_to_colors(labels_image_cpu).reshape((480, 848, 4))

        self.labels_image_rgba_tex.set(labels_image_cpu_rgba)

        self.frame_num += 1

        imgui.text("f: " + str(self.frame_num))
        imgui.image(self.labels_image_rgba_tex.gl(), self.DIM_X * self.dpi_scale, self.DIM_Y * self.dpi_scale)

if __name__ == '__main__':
    run_app(RunLiveApp)
