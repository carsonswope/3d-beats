from OpenGL.GL import * 
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import imgui
import engine.glm_np as glm_np

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *

from engine.window import AppBase
from engine.buffer import GpuBuffer
from engine.texture import GpuTexture
from engine.mesh import GpuMesh
from engine.framebuffer import GpuFramebuffer
from camera.std_camera import StdCamera

from engine.mesh_primitives import make_cylinder

from camera.arcball import ArcBallCam

from util import MAX_UINT16, rs_projection

class PoseFitApp(AppBase):
    def __init__(self):
        super().__init__(title="Test-icles", width=1800, height=1450)

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

        # why cant load all the frames?? seems to block at 64
        self.NUM_FRAMES = 0
        self.frame_num = 3

        self.depth_images = []

        while True:
            frame_present, frames = self.pipeline.try_wait_for_frames()
            if not frame_present:
                break

            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data()).reshape((1, self.DIM_Y, self.DIM_X))
            self.depth_images.append(depth_image)
            self.NUM_FRAMES += 1

        self.depth_cam = StdCamera()

        self.obj_mesh = GpuMesh(
            num_idxes = (self.DIM_X - 1) * (self.DIM_Y - 1) * 6,
            vtxes_shape = (self.DIM_Y, self.DIM_X))

        self.fbo = GpuFramebuffer((self.DIM_X, self.DIM_Y))
        self.fbo_rgba = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.arc_ball = ArcBallCam()
        self.obj_xyz = [0., 0., 0.]

        self.cylinder_mesh = make_cylinder(num_sections=16)

    def splash(self):
        imgui.text('loading...')

    def tick(self, t):
        
        # load depth image
        self.depth_image_gpu.cu().set(self.depth_images[self.frame_num])

        grid_dim = (1, (self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1)
        block_dim = (1,32,32)

        self.pts_gpu.cu().fill(np.float32(0))

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

        self.labels_image_gpu.cu().fill(MAX_UINT16)
        self.decision_tree_evaluator.get_labels_forest(self.forest, self.depth_image_gpu.cu(), self.labels_image_gpu.cu())

        # unmap from cuda.. isn't actually necessary, but just to make sure..
        self.depth_image_gpu.gl()

        labels_image_cpu = self.labels_image_gpu.cu().get()
        labels_image_cpu_rgba = self.data_config.convert_ids_to_colors(labels_image_cpu).reshape((480, 848, 4))
        labels_image_cpu_rgba = labels_image_cpu_rgba[:,:,0:3] # convert to 3 channels!

        # generate mesh for 3d rendering!
        self.obj_mesh.vtx_color.cu().set(labels_image_cpu_rgba)
        self.obj_mesh.vtx_pos.cu().set(self.pts_gpu.cu())
        num_triangles = self.points_ops.make_triangles(self.DIM_X, self.DIM_Y, self.obj_mesh.vtx_pos, self.obj_mesh.idxes)
        self.obj_mesh.num_idxes = int(num_triangles * 3)

        self.fbo.bind(rgba_tex=self.fbo_rgba)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnable(GL_DEPTH_TEST)

        glClearColor(.1, .15, .15, 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.depth_cam.use()
        cam_proj = rs_projection(self.FOCAL, self.DIM_X, self.DIM_Y, self.PP[0], self.PP[1], 50., 50000.)
        self.depth_cam.u_mat4('cam_proj', cam_proj)

        cam_inv_tform = self.arc_ball.get_cam_inv_tform()
        self.depth_cam.u_mat4('cam_inv_tform', cam_inv_tform)

        obj_tform = glm_np.translate((self.obj_xyz[0], self.obj_xyz[1], self.obj_xyz[2],))
        self.depth_cam.u_mat4('obj_tform', obj_tform)

        self.obj_mesh.draw()

        cyl_tform = glm_np.scale((100, 100, 300))
        self.depth_cam.u_mat4('obj_tform', cyl_tform)

        self.cylinder_mesh.draw()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        _, self.frame_num = imgui.slider_int("f", self.frame_num, 0, self.NUM_FRAMES - 1)

        self.arc_ball.draw_control_gui()
        _, self.obj_xyz[0] = imgui.slider_float("obj x", self.obj_xyz[0], -2000, 2000)
        _, self.obj_xyz[1] = imgui.slider_float("obj y", self.obj_xyz[1], -2000, 2000)
        _, self.obj_xyz[2] = imgui.slider_float("obj z", self.obj_xyz[2], -2000, 2000)

        imgui.image(self.fbo_rgba.gl(), self.DIM_X * 2, self.DIM_Y * 2)


if __name__ == '__main__':
    a = PoseFitApp()
    a.run()
