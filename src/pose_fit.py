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
from cuda.mean_shift import *
from cuda.fit_mesh import *

from engine.window import AppBase
from engine.buffer import GpuBuffer
from engine.texture import GpuTexture
from engine.mesh import GpuMesh
from engine.framebuffer import GpuFramebuffer
from camera.std_camera import StdCamera

from engine.mesh_primitives import make_cylinder

from camera.arcball import ArcBallCam

from util import MAX_UINT16, rs_projection

def get_depth_range(d):
    d_active = d[(d > 0) & (d < 10000)]
    min_p = np.min(d_active)
    max_p = np.max(d_active)
    return (min_p, max_p)

def make_debug_depth_img(d, i, range=(0, MAX_UINT16)):
    d_rgba = np.zeros((d.shape[0], d.shape[1], 4), dtype=np.uint8)
    d_rgba[:,:,3] = 255
    d_rgba[d == 0] = np.array([167, 195, 162, 255], dtype=np.uint8) # cute calm green
    d_rgba[d == MAX_UINT16] = np.array([240, 34, 100, 255], dtype=np.uint8) # more reddish

    active_coords = np.where((d > 0) & (d < MAX_UINT16))
    d_active = d[(d > 0) & (d < MAX_UINT16)]
    min_p, max_p = range

    norm_depths = (255. * (1. - (d_active - (min_p * 1.)) / (max_p - min_p))).astype(np.uint8)
    d_rgba[active_coords[0], active_coords[1], 0] = norm_depths
    d_rgba[active_coords[0], active_coords[1], 1] = norm_depths
    d_rgba[active_coords[0], active_coords[1], 2] = norm_depths
    d_rgba[active_coords[0], active_coords[1], 3] = 255

    i.set(d_rgba)

class CylinderTform():
    def __init__(self):
        # translate, rotate, scale
        self.t = np.zeros(3, dtype=np.float32)
        self.r = np.zeros(3, dtype=np.float32)
        self.s = np.zeros(3, dtype=np.float32)
    
    def get_tform(self):
        return glm_np.translate(self.t) @ \
            glm_np.rotate_z(self.r[2]) @ \
            glm_np.rotate_x((np.pi/2) + self.r[0]) @ \
            glm_np.scale(self.s)
    
    def copy(self):
        n = CylinderTform()
        n.t[:] = self.t
        n.r[:] = self.r
        n.s[:] = self.s
        return n

    def make_random(self):
        n = self.copy()

        a = np.random.randint(3)
        if a  == 0: # translation
            b = np.random.randint(3)
            n.t[b] = np.random.normal(n.t[b], 25.)
        elif a == 1: # rotation
            b = np.random.choice([0, 2]) # only rotate x & z axes
            n.r[b] = np.random.normal(n.r[b], 0.1)
        elif a == 2: # scale
            b = np.random.normal(n.s[0], 5.)
            n.s[0] = b * 1.3
            n.s[1] = b

        return n


class PoseFitApp(AppBase):
    def __init__(self):
        super().__init__(title="Test-icles", width=1800, height=1750)

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
        self.mean_shift = MeanShift()
        self.fit_mesh = FitMesh()

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

        self.orig_depth_image_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), np.uint16)
        self.depth_image_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), np.uint16)
        self.labels_image_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.labels_image_rgba_tex = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.NUM_FRAMES = 0
        self.frame_num = 3

        self.depth_images = []

        while True:
            frame_present, frames = self.pipeline.try_wait_for_frames()
            if not frame_present:
                # why cant load all the frames?? seems to block at 64
                break

            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data()).reshape((1, self.DIM_Y, self.DIM_X))
            self.depth_images.append(depth_image)
            self.NUM_FRAMES += 1

        self.depth_cam = StdCamera()
        self.depth_cam_proj = rs_projection(self.FOCAL, self.DIM_X, self.DIM_Y, self.PP[0], self.PP[1], 50., 50000.)

        self.obj_mesh = GpuMesh(
            num_idxes = (self.DIM_X - 1) * (self.DIM_Y - 1) * 6,
            vtxes_shape = (self.DIM_Y, self.DIM_X))

        self.fbo = GpuFramebuffer((self.DIM_X, self.DIM_Y))
        self.fbo_rgba = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))
        self.fbo_depth = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RED_INTEGER, GL_UNSIGNED_SHORT))

        self.rgba_debug = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))
        self.rgba_debug_2 = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        self.arc_ball = ArcBallCam()

        self.cylinder_mesh = make_cylinder(num_sections=16)

        self.cylinder_tform = CylinderTform()
        self.next_cylinder_tform = None

        mean_shift_variances = np.array([100., 100., 100., 100.,], dtype=np.float32)
        self.mean_shift_variances_cu = GpuBuffer((4,), dtype=np.float32)
        self.mean_shift_variances_cu.cu().set(mean_shift_variances)

        self.best_cost = np.inf

    def splash(self):
        imgui.text('loading...')

    def tick(self, t):
        
        # load depth image
        self.orig_depth_image_gpu.cu().set(self.depth_images[self.frame_num])
        self.depth_image_gpu.cu().set(self.depth_images[self.frame_num])

        orig_depth_image_range = get_depth_range(self.depth_images[self.frame_num][0])
        make_debug_depth_img(self.depth_images[self.frame_num][0], self.rgba_debug_2, range=orig_depth_image_range)

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

        if self.next_cylinder_tform is not None:
            test_tform = self.next_cylinder_tform

        # if cylinder tform hasnt been set yet (at 0 z coordinate, silly way to check)
        elif self.cylinder_tform.t[2] < 0.1 and self.cylinder_tform.t[2] > -0.1:

            label_means = self.mean_shift.run(
                6, # iterations to run for mean shift
                self.labels_image_gpu.cu(),
                4, # classes
                self.mean_shift_variances_cu.cu())
            label_means = label_means.astype(np.int)

            l_depth = self.depth_images[self.frame_num][0][label_means[0][1], label_means[0][0]]
            l_point = self.calibrated_plane.plane @ np.array([
                l_depth * (label_means[0][0] - self.PP[0]) / self.FOCAL,
                l_depth * (label_means[0][1] - self.PP[1]) / self.FOCAL,
                l_depth,
                1.
            ], dtype=np.float32)

            self.cylinder_tform.t[:] = l_point[0:3]
            self.cylinder_tform.r[:] = [0., 0., 0.]
            self.cylinder_tform.s[:] = [200. * 1.3, 200., 800.]

            test_tform = self.cylinder_tform
        
        else:

            test_tform = self.cylinder_tform.make_random()

            # print('randomize!')

       # draw real depth scene
        self.fbo.bind(depth_tex=self.fbo_depth)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_DEPTH_TEST)

        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.depth_cam.use()


        cyl_tform = test_tform.get_tform()

        self.depth_cam.u_mat4('cam_proj', self.depth_cam_proj)
        self.depth_cam.u_mat4('cam_inv_tform', np.identity(4, dtype=np.float32))
        self.depth_cam.u_mat4('obj_tform', np.linalg.inv(self.calibrated_plane.plane) @ cyl_tform)

        self.cylinder_mesh.draw()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.fbo_depth.copy_to_gpu_buffer(self.depth_image_gpu)
        img_cost = self.fit_mesh.calc_image_cost(
            self.orig_depth_image_gpu,
            self.depth_image_gpu,
            self.labels_image_gpu,
            1)

        if img_cost < self.best_cost:
            self.cylinder_tform = test_tform
            self.best_cost = img_cost
            print('better cost: ', self.best_cost)

        make_debug_depth_img(self.fbo_depth.get(), self.rgba_debug, range=orig_depth_image_range)


        labels_image_cpu = self.labels_image_gpu.cu().get()
        labels_image_cpu_rgba = self.data_config.convert_ids_to_colors(labels_image_cpu).reshape((480, 848, 4))
        labels_image_cpu_rgba = labels_image_cpu_rgba[:,:,0:3] # convert to 3 channels!

        # generate mesh for 3d rendering!
        self.obj_mesh.vtx_color.cu().set(labels_image_cpu_rgba)
        self.obj_mesh.vtx_pos.cu().set(self.pts_gpu.cu())
        num_triangles = self.points_ops.make_triangles(self.DIM_X, self.DIM_Y, self.obj_mesh.vtx_pos, self.obj_mesh.idxes)
        self.obj_mesh.num_idxes = int(num_triangles * 3)

         # draw debug scene..
        self.fbo.bind(rgba_tex=self.fbo_rgba)

        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glEnable(GL_DEPTH_TEST)

        glClearColor(.1, .15, .15, 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.depth_cam.u_1ui('color_mode', 0)
        self.depth_cam.u_4f('solid_color', np.array([1., 0., 1., 1.], dtype=np.float32))

        cam_inv_tform = self.arc_ball.get_cam_inv_tform() @ glm_np.translate(-self.cylinder_tform.t)
        self.depth_cam.u_mat4('cam_inv_tform', cam_inv_tform)

        obj_tform = np.identity(4, dtype=np.float32)
        self.depth_cam.u_mat4('obj_tform', obj_tform)

        self.obj_mesh.draw()

        self.depth_cam.u_1ui('color_mode', 1)

        self.depth_cam.u_mat4('obj_tform', self.cylinder_tform.get_tform())

        self.cylinder_mesh.draw()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        frame_num_changed, self.frame_num = imgui.slider_int("f", self.frame_num, 0, self.NUM_FRAMES - 1)
        if frame_num_changed:
            self.best_cost = np.inf
            self.next_cylinder_tform = self.cylinder_tform
        else:
            self.next_cylinder_tform = None

        self.arc_ball.draw_control_gui()

        imgui.text(f'cost: {self.best_cost}')

        imgui.image(self.rgba_debug.gl(), self.DIM_X, self.DIM_Y)
        imgui.image(self.rgba_debug_2.gl(), self.DIM_X, self.DIM_Y)
        imgui.image(self.fbo_rgba.gl(), self.DIM_X, self.DIM_Y)

        #         dd = self.fbo_depth.get()

        # dd_rgba = np.zeros((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        # dd_rgba[:,:,3] = 255
        # dd_rgba[dd == 0] = np.array([167, 195, 162, 255], dtype=np.uint8) # cute calm green
        # self.rgba_debug.set(dd_rgba)


if __name__ == '__main__':
    a = PoseFitApp()
    a.run()
