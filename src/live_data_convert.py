from OpenGL.GL import *
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
import imgui
import argparse

import glfw

import glm

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *
np.set_printoptions(suppress=True)

"""
for filtered:
                "-i", "./datagen/sets/full_4class/f1.bag",
                "-o", "./datagen/sets/full_4class/f1_randomized/",
                "--colors", "3",
                "--mask_model", "./models_out/full_4class_d123.npy",
                "--mask_label", "3",
                "--plane_z_threshold", "55.",
                "--max_images", "16"
"""

from util import MAX_UINT16, rs_projection


from engine.window import AppBase
from engine.buffer import GpuBuffer
from engine.texture import GpuTexture
from engine.mesh import GpuMesh
from engine.framebuffer import GpuFramebuffer
from camera.std_camera import StdCamera


class LiveDataConvert(AppBase):
    def __init__(self):
        super().__init__(title="Live Data Conversion")

        self.depth_cam = StdCamera()

        self.obj_demo_mesh = GpuMesh(num_idxes=3, vtxes_shape=(3,))
        self.obj_demo_mesh.idxes.cu().set(np.array([
            0, 1, 2,
        ], dtype=np.uint32))
        self.obj_demo_mesh.vtx_pos.cu().set(np.array([
            # [-0.5, -0.5, 0., 1.],
            # [ 0.5, -0.5, 0., 1.],
            # [ 0.5,  0.5, 0., 1.],
            [ 0, 0, 4000., 1.],
            [ 400, 400, 4100., 1.],
            [ 400,  0, 3900., 1.],
        ], dtype=np.float32))
        self.obj_demo_mesh.vtx_color.cu().set(np.array([
            [250, 0, 0],
            [ 0, 250, 0],
            [ 0,  0, 250],
        ], dtype=np.uint8))
        
        parser = argparse.ArgumentParser(description='Convert a realsense .bag file into training data for RDF')
        parser.add_argument('-i', '--bag_in', nargs='?', required=True, type=str, help='Path to realsense .bag input file')
        parser.add_argument('-o', '--out', nargs='?', required=True, type=str, help='Directory to save formatted date')
        parser.add_argument('--colors', nargs='?', required=True, type=int, help='Num colors to look for in input image, to convert to labels')
        parser.add_argument('--colors_num_restarts', nargs='?', required=False, type=int, help='Num times to run EM algorithm in search of best fit for colors/labels')
        parser.add_argument('--colors_num_iterations', nargs='?', required=False, type=int, help='Num rounds to run EM iteration per full pass of algorithm in search of best fit for colors/labels')
        parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
        parser.add_argument('--plane_z_threshold', nargs='?', required=True, type=float, help='Z-value threshold in plane coordinates for clipping depth image pixels')
        parser.add_argument('--max_images', nargs='?', required=False, type=int, help='Maximum number of images to process')
        parser.add_argument('--frames_timestamp_max_diff', nargs='?', required=False, type=float, help='Only process a frems if the depth & color frame have timestamps that are different by less than X (ms?)')
        parser.add_argument('--mask_model', nargs='?', required=False, type=str, help='Path to model to run to get mask')
        parser.add_argument('--mask_label', nargs='?', required=False, type=int, help='ID from given mask model to filter by')

        args = parser.parse_args()

        IN_PATH = args.bag_in
        self.OUT_PATH = args.out

        self.COLOR_EM_NUM_COLORS = args.colors
        self.COLOR_EM_NUM_TRIES = args.colors_num_restarts or 8
        self.COLOR_EM_ITERATIONS = args.colors_num_iterations or 32

        self.PLANE_RANSAC_NUM_CANDIDATES = args.plane_num_iterations or 25000
        self.PLANE_Z_THRESHOLD = args.plane_z_threshold

        self.calibrated_plane = CalibratedPlane(self.PLANE_RANSAC_NUM_CANDIDATES, self.PLANE_Z_THRESHOLD)

        self.MAX_IMAGES = args.max_images or np.Infinity

        self.FRAMES_TIMESTAMP_MAX_DIFF = args.frames_timestamp_max_diff or 6.

        self.FRAMES_PER_RECOMPUTE_PLANE = 20

        self.MASK_MODEL_PATH = args.mask_model
        self.MASK_LABEL = args.mask_label

        if (self.MASK_MODEL_PATH and not self.MASK_LABEL) or (not self.MASK_MODEL_PATH and self.MASK_LABEL):
            print('--mask_path and --mask_label are both required if using mask')
            return

        if self.MASK_MODEL_PATH:
            self.mask_model = DecisionForest.load(self.MASK_MODEL_PATH)
            self.decision_tree_evaluator = DecisionTreeEvaluator()
        else:
            self.mask_model = None

        self.points_ops = PointsOps()

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(IN_PATH, repeat_playback=False)

        self.config.enable_stream(rs.stream.depth, rs.format.z16)
        self.config.enable_stream(rs.stream.color, rs.format.rgb8)

        pf = self.pipeline.start(self.config)
        pf.get_device().as_playback().set_real_time(False)

        depth_intr = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        self.DIM_X = depth_intr.width
        self.DIM_Y = depth_intr.height
        self.FOCAL = np.float32(depth_intr.fx) # should be same as fy..
        self.PP = np.array([depth_intr.ppx, depth_intr.ppy], dtype=np.float32)

        self.align = rs.align(rs.stream.depth)

        self.obj_mesh = GpuMesh(
            num_idxes = (self.DIM_X - 1) * (self.DIM_Y - 1) * 6,
            vtxes_shape = (self.DIM_Y, self.DIM_X))

        self.depth_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), dtype=np.uint16)

        if self.mask_model:
            self.mask_labels = np.zeros((1, self.DIM_Y, self.DIM_X), dtype=np.uint16)
            self.mask_labels_gpu = GpuBuffer((1, self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.color_mapping = None
        self.color_mapping_gpu = GpuBuffer((self.COLOR_EM_NUM_COLORS, 3), dtype=np.uint8)

        self.labels_image = np.zeros((self.DIM_Y, self.DIM_X), dtype=np.uint16)

        self.color_image_rgba = np.zeros((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        self.color_image_rgba_gpu = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))

        # Streaming loop
        self.frame_count = 0

        self.fbo = GpuFramebuffer((self.DIM_X, self.DIM_Y))

        self.fbo_rgba = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGBA, GL_UNSIGNED_BYTE))
        self.fbo_depth = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RED_INTEGER, GL_UNSIGNED_SHORT))

        self.fbo_rgb_2 = GpuTexture((self.DIM_X, self.DIM_Y), (GL_RGB, GL_UNSIGNED_BYTE))


    # def make_modified

    def splash(self):
        imgui.text('loading..')

    def make_color_mapping(self):

        print('making color mapping... ')

        best_colors_diffs = np.Infinity
        best_colors = np.zeros((self.COLOR_EM_NUM_COLORS, 3), dtype=np.uint8)

        colors_gpu = GpuBuffer((self.COLOR_EM_NUM_COLORS, 3), dtype=np.uint8)
        pixel_counts_per_group_gpu = GpuBuffer((self.COLOR_EM_NUM_COLORS, 5), dtype=np.uint64)

        for _1 in range(self.COLOR_EM_NUM_TRIES):

            colors = np.random.uniform(0, 255, (self.COLOR_EM_NUM_COLORS, 3)).astype(np.uint8)

            for _2 in range(self.COLOR_EM_ITERATIONS):

                colors_gpu.cu().set(colors)

                pixel_counts_per_group_gpu.cu().fill(np.uint64(0))

                grid_dim3 = ((self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1, 1)
                block_dim3 = (32,32,1)

                self.points_ops.split_pixels_by_nearest_color(
                    np.int32(self.DIM_X),
                    np.int32(self.DIM_Y),
                    np.int32(self.COLOR_EM_NUM_COLORS),
                    colors_gpu.cu(),
                    self.obj_mesh.vtx_color.cu(),
                    pixel_counts_per_group_gpu.cu(),
                    grid=grid_dim3,
                    block=block_dim3)

                pixel_counts_per_group = pixel_counts_per_group_gpu.cu().get()

                grouping_cost = np.sum(pixel_counts_per_group[:,4].view(np.float64))

                colors = (pixel_counts_per_group[:,1:4].T / pixel_counts_per_group[:,0]).T.astype(np.uint8)
            
            if grouping_cost < best_colors_diffs:
                best_colors_diffs = grouping_cost
                best_colors = np.copy(colors)
        
        del colors_gpu
        del pixel_counts_per_group_gpu

        print('made.')

        return best_colors
    
    # converts incoming depth image to geometry, 
    def rerender_image(self, vtx_center):
        # these variables should be good:
        # self.color_image_gpu
        # self.depth_gpu
        # self.pts_gpu

        # also need:
        # - cam intrinsics
        # - plane matrix.

        # 1. create idxes for triangles to re-create depth image.
        # 2. convert transform matrix of the following operations:
        #    - convert pts to plane space
        #    - apply random transformation to pts:
        #      - scale
        #      - rotation
        #      - skew? translate?
        #    - convert pts back to camera space
        # 3. re-render color and depth images

        # OUTPUT: new values to:
        # self.color_image_gpu
        # self.depth_gpu

        num_triangles = self.points_ops.make_triangles(self.DIM_X, self.DIM_Y, self.obj_mesh.vtx_pos, self.obj_mesh.idxes)
        self.obj_mesh.num_idxes = int(num_triangles * 3)

        # durr, some dummy OpenGL
        self.fbo.bind(self.fbo_rgba, self.fbo_depth)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_DEPTH_TEST)

        glClearColor(0., 0., 0., 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.depth_cam.use()

        cam_proj = rs_projection(self.FOCAL, self.DIM_X, self.DIM_Y, self.PP[0], self.PP[1], 50., 50000.)
        self.depth_cam.u_mat4('cam_proj', cam_proj)

        # dont randomly transform 1st frame.
        if self.frame_count > 2:
            SCALE_VARIANCE = 0.2
            SCALE_SKEW_VARIANCE = 0.06
            ROTATE_VARIANCE = 0.3
            TRANSLATE_VARIANCE = 150
        else:
            SCALE_VARIANCE = 0
            SCALE_SKEW_VARIANCE = 0
            ROTATE_VARIANCE = 0
            TRANSLATE_VARIANCE = 0
        tform_scale = np.random.default_rng().normal(1, SCALE_VARIANCE, 1)[0]
        tform_skew = np.random.default_rng().normal(0, SCALE_SKEW_VARIANCE, 3)
        tform_rotate = np.random.default_rng().normal(0., ROTATE_VARIANCE, 1)[0]
        tform_translate = np.random.default_rng().normal(0., TRANSLATE_VARIANCE, 3)

        # convert to plane space -> subtract mean -> random transform -> re-add mean -> convert back to camera space
        obj_tform = np.linalg.inv(self.calibrated_plane.plane) @ \
            np.array(glm.translate(glm.mat4(), glm.vec3( vtx_center[0],  vtx_center[1],  vtx_center[2]))) @ \
            np.array(glm.translate(glm.mat4(), glm.vec3(tform_translate[0], tform_translate[1], tform_translate[2]))) @ \
            np.array(glm.scale(glm.mat4(), glm.vec3(tform_scale + tform_skew[0], tform_scale + tform_skew[1], tform_scale + tform_skew[2]))) @ \
            np.array(glm.translate(glm.mat4(), glm.vec3(-vtx_center[0], -vtx_center[1], -vtx_center[2]))) @ \
            self.calibrated_plane.plane @ \
            np.array(glm.rotate(glm.mat4(), tform_rotate, glm.vec3(0., 0., 1.)))
        self.depth_cam.u_mat4('obj_tform', obj_tform)

        self.obj_mesh.draw()


        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # TODO: implement more efficient copy from (opengl) texture to (cuda) array buffer.
        # going through CPU is just silly
        rgb_rendered = self.fbo_rgba.get()
        rgb_rendered = rgb_rendered[:,:,0:3]
        self.obj_mesh.vtx_color.cu().set(rgb_rendered)

        rgb_depth_rendered = self.fbo_depth.get()
        self.depth_gpu.cu().set(rgb_depth_rendered)
    
    def finish(self):
        glfw.set_window_should_close(self.window, True)

        # write json config as entry point into model
        obj= {}
        obj['img_dims'] = [self.DIM_X, self.DIM_Y]
        obj['num_images'] = self.frame_count
        obj['id_to_color'] = {'0': [0, 0, 0, 0]}
        for c_id in range(self.COLOR_EM_NUM_COLORS):
            c = self.color_mapping[c_id]
            obj['id_to_color'][str(c_id + 1)] = [int(c[0]), int(c[1]), int(c[2]), 255]

        cfg_json_file = open(f'{self.OUT_PATH}/config.json', 'w')
        cfg_json_file.write(json.dumps(obj))
        cfg_json_file.close()

    def tick(self, t):

        if self.frame_count >= self.MAX_IMAGES:
            self.finish()
            return
        
        frames = None
        while frames == None:
            try:
                frames = self.pipeline.wait_for_frames(1000)
                df_time = frames.get_depth_frame().get_timestamp()
                cf_time = frames.get_color_frame().get_timestamp()
                # only process frame pairs whose timestamps overlap reasonably well
                if np.abs(df_time - cf_time) > self.FRAMES_TIMESTAMP_MAX_DIFF:
                    frames = None
            except:
                self.finish()
                return

        self.frame_count += 1

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        # depth_np = 
        self.depth_gpu.cu().set(np.asanyarray(depth_frame.data))

        grid_dim = (1, (self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1)
        block_dim = (1,32,32)

        # deproject depth image to 3D points in camera space
        self.points_ops.deproject_points(
            np.array([1, self.DIM_X, self.DIM_Y, -1], dtype=np.int32),
            self.PP,
            self.FOCAL,
            self.depth_gpu.cu(),
            self.obj_mesh.vtx_pos.cu(),
            grid=grid_dim,
            block=block_dim)

        if not self.calibrated_plane.is_set() or self.frame_count % self.FRAMES_PER_RECOMPUTE_PLANE == 0:
            self.calibrated_plane.make(self.obj_mesh.vtx_pos, (self.DIM_X, self.DIM_Y))
        
        # every point..
        grid_dim2 = (((self.DIM_X * self.DIM_Y) // 1024) + 1, 1, 1)
        block_dim2 = (1024, 1, 1)

        # convert deprojected points to plane space
        self.points_ops.transform_points(
            np.int32(self.DIM_X * self.DIM_Y),
            self.obj_mesh.vtx_pos.cu(),
            self.calibrated_plane.get_mat(),
            grid=grid_dim2,
            block=block_dim2)
        
        # filter deprojected points in plane space
        self.calibrated_plane.filter_points_by_plane(
            np.int32(self.DIM_X * self.DIM_Y),
            np.float32(self.PLANE_Z_THRESHOLD),
            self.obj_mesh.vtx_pos.cu(),
            grid=grid_dim2,
            block=block_dim2)

        # get mean of vtxes in plane space (for later)
        vtx_center = np.sum(self.obj_mesh.vtx_pos.cu().get().reshape((-1, 4)), axis=0)
        vtx_center = vtx_center / vtx_center[3]

        # convert deprojected points back to camera space
        self.points_ops.transform_points(
            np.int32(self.DIM_X * self.DIM_Y),
            self.obj_mesh.vtx_pos.cu(),
            np.linalg.inv(self.calibrated_plane.get_mat()),
            grid=grid_dim2,
            block=block_dim2)
        
        self.depth_gpu.cu().fill(np.uint16(0))

        # regenerate depth image from 3D points
        self.points_ops.depths_from_points(
            np.array([1, self.DIM_X, self.DIM_Y, -1], dtype=np.int32),
            self.depth_gpu.cu(),
            self.obj_mesh.vtx_pos.cu(),
            grid=grid_dim,
            block=block_dim)

        # copy back to cpu-side depth frame memory, so align processing block can run
        # self.depth_gpu.cu().get(depth_np)

        frames_aligned = self.align.process(frames)
        color_frame = frames_aligned.get_color_frame()

        # color image is now aligned to filtered depth image
        color_image = np.asanyarray(color_frame.get_data())
        # color_image_cu.set(color_image)

        # how you could crop if you wanted..
        # should be an input option??
        # color_image[:,750:] = np.array([0, 0, 0], dtype=np.uint8)

        self.obj_mesh.vtx_color.cu().set(color_image)
        self.rerender_image(vtx_center)

        color_image = self.obj_mesh.vtx_color.cu().get()
        depth_np = self.depth_gpu.cu().get()[0]

        if self.mask_model:

            depth_np[depth_np == 0] = MAX_UINT16
            self.depth_gpu.cu().set(depth_np)

            self.mask_labels_gpu.cu().fill(np.uint16(0))
            self.decision_tree_evaluator.get_labels_forest(self.mask_model, self.depth_gpu.cu(), self.mask_labels_gpu.cu())
            self.mask_labels_gpu.cu().get(self.mask_labels)
            color_image[self.mask_labels[0] != self.MASK_LABEL] = np.array([0, 0, 0], dtype=np.uint8)
            depth_np[depth_np == MAX_UINT16] = 0
            self.depth_gpu.cu().set(depth_np)

        self.obj_mesh.vtx_color.cu().set(color_image)

        if self.color_mapping is None:
            self.color_mapping = self.make_color_mapping()
            self.color_mapping_gpu.cu().set(self.color_mapping)

        grid_dim3 = ((self.DIM_X // 32) + 1, (self.DIM_Y // 32) + 1, 1)
        block_dim3 = (32,32,1)

        self.points_ops.apply_point_mapping(
            np.int32(self.DIM_X),
            np.int32(self.DIM_Y),
            np.int32(self.COLOR_EM_NUM_COLORS),
            self.color_mapping_gpu.cu(),
            self.obj_mesh.vtx_color.cu(),
            grid=grid_dim3,
            block=block_dim3)

        # render raw labels image from color image (1..n)
        self.obj_mesh.vtx_color.cu().get(color_image)
        self.labels_image[:,:] = 0
        for xx in range(self.COLOR_EM_NUM_COLORS):
            self.labels_image[np.where(np.all(color_image == self.color_mapping[xx], axis=2))] = xx + 1 # group 0 is null group, starts at 1
        Image.fromarray(self.labels_image).save(f'{self.OUT_PATH}/{str(self.frame_count - 1).zfill(8)}_labels.png')

        # debug rendering of labels image
        self.color_image_rgba[:,:,3] = 0
        self.color_image_rgba[:,:,0:3] = color_image
        self.color_image_rgba[np.any(color_image > 0, axis=2),3] = 255
        Image.fromarray(self.color_image_rgba).save(f'{self.OUT_PATH}/{str(self.frame_count - 1).zfill(8)}_labels_rgba.png')

        # render raw depth image
        depth_np[depth_np == 0] = MAX_UINT16
        Image.fromarray(depth_np).save(f'{self.OUT_PATH}/{str(self.frame_count - 1).zfill(8)}_depth.png')

        # debug rendering of depth image
        depth_rgba_np = np.zeros((self.DIM_Y, self.DIM_X, 4), dtype=np.uint8)
        depth_rgba_np[depth_np == MAX_UINT16 ] = np.array([167, 195, 162, 255], dtype=np.uint8) # cute calm green
        active_coords = np.where(depth_np < MAX_UINT16)
        max_depth = np.max(depth_np[depth_np < MAX_UINT16])
        min_depth = np.min(depth_np[depth_np < MAX_UINT16])
        norm_depths = (255. * (1. - (depth_np[depth_np < MAX_UINT16] - (min_depth * 1.)) / (max_depth - min_depth))).astype(np.uint8)
        depth_rgba_np[active_coords[0], active_coords[1], 0] = norm_depths
        depth_rgba_np[active_coords[0], active_coords[1], 1] = norm_depths
        depth_rgba_np[active_coords[0], active_coords[1], 2] = norm_depths
        depth_rgba_np[active_coords[0], active_coords[1], 3] = 255
        Image.fromarray(depth_rgba_np).save(f'{self.OUT_PATH}/{str(self.frame_count - 1).zfill(8)}_depth_rgba.png')

        self.color_image_rgba_gpu.set(self.color_image_rgba)

        imgui.text('image below')
        # poor mans dpi for now..
        imgui.image(self.color_image_rgba_gpu.gl(), self.DIM_X, self.DIM_Y)
        # imgui.text('diff..')
        # imgui.image(self.fbo_rgb_2.gl(), self.DIM_X // 2, self.DIM_Y // 2)

        imgui.text('re-rendered')
        # imgui.image(self.fbo_rgb_2.gl(), self.DIM_X, self.DIM_Y)
        imgui.image(self.fbo_rgba.gl(), self.DIM_X, self.DIM_Y)


if __name__ == '__main__':
    a = LiveDataConvert()
    a.run()
