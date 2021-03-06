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

class RunLive_Layered(AppBase):
    def __init__(self):
        super().__init__(title="Layered RDF Demo", width=848, height=800)

        parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
        parser.add_argument('-cfg', nargs='?', required=True, type=str, help='Path to the layered decision forest config file')
        parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
        parser.add_argument('--plane_z_threshold', nargs='?', required=False, type=float, help='Z-value threshold in plane coordinates for clipping depth image pixels')
        py_nvcc_utils.add_args(parser)
        rs_util.add_args(parser)
        args = parser.parse_args()

        py_nvcc_utils.config_compiler(args)

        NUM_RANDOM_GUESSES = args.plane_num_iterations or 25000
        self.PLANE_Z_OUTLIER_THRESHOLD = args.plane_z_threshold or 40.

        self.calibrated_plane = CalibratedPlane(NUM_RANDOM_GUESSES, self.PLANE_Z_OUTLIER_THRESHOLD)

        self.TRAIN_DIM_X = 848
        self.pipeline, self.depth_intrin, self.DIM_X, self.DIM_Y, self.FOCAL, self.PP = rs_util.start_stream(args)

        self.LABELS_REDUCE = 2

        self.layered_rdf = LayeredDecisionForest.load(args.cfg, (self.DIM_Y, self.DIM_X), self.LABELS_REDUCE)
        self.points_ops = PointsOps()

        self.pts = GpuBuffer((self.DIM_Y, self.DIM_X, 4), dtype=np.float32)

        self.depth_image = GpuBuffer((1, self.DIM_Y, self.DIM_X), np.uint16)


        self.labels_image = GpuBuffer((1, self.DIM_Y // self.LABELS_REDUCE, self.DIM_X // self.LABELS_REDUCE), dtype=np.uint16)

        self.labels_image_rgba = GpuBuffer((self.DIM_Y // self.LABELS_REDUCE, self.DIM_X // self.LABELS_REDUCE, 4), dtype=np.uint8)
        self.labels_image_rgba_tex = GpuTexture((self.DIM_X // self.LABELS_REDUCE, self.DIM_Y // self.LABELS_REDUCE), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.frame_num = 0

        self.x = 5

    def splash(self):
        imgui.text('loading...')

    def cleanup(self):
        self.pipeline.stop()

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

        # run RDF!
        # scale_factor = 
        self.layered_rdf.run(self.depth_image, self.labels_image, self.DIM_X / self.TRAIN_DIM_X)

        # make RGBA image
        self.labels_image_rgba.cu().fill(0)
        self.points_ops.make_rgba_from_labels(
            np.uint32(self.DIM_X // self.LABELS_REDUCE),
            np.uint32(self.DIM_Y // self.LABELS_REDUCE),
            np.uint32(self.layered_rdf.num_layered_classes),
            self.labels_image.cu(),
            self.layered_rdf.label_colors.cu(),
            self.labels_image_rgba.cu(),
            grid = ((self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1, 1),
            block = (32,32,1))
        self.labels_image_rgba_tex.copy_from_gpu_buffer(self.labels_image_rgba)

        self.frame_num += 1

        self.begin_imgui_main()
        imgui.image(self.labels_image_rgba_tex.gl(), self.DIM_X * self.dpi_scale, self.DIM_Y * self.dpi_scale)
        imgui.end()

        imgui.set_next_window_size(200 * self.dpi_scale, 124 * self.dpi_scale)
        imgui.set_next_window_bg_alpha(0.3)
        imgui.begin('profile', imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_SCROLLBAR)
        profile_plot_width = 150 * self.dpi_scale
        profile_plot_height = 60 * self.dpi_scale
        imgui.text(f'ms/frame: {"{:.1f}".format(self.ms_per_frame_log[-1])}')
        imgui.plot_lines('##ms-frame',
            np.array(self.ms_per_frame_log, dtype=np.float32),
            scale_max=100.,
            scale_min=0.,
            graph_size=(profile_plot_width , profile_plot_height))
        imgui.end()

if __name__ == '__main__':
    run_app(RunLive_Layered)
