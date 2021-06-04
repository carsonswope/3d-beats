import pyrealsense2 as rs
import numpy as np
import cv2

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.curandom as cu_rand

from decision_tree import *
from cuda.points_ops import *
np.set_printoptions(suppress=True)

import time

MODEL_OUT_NAME = 'models_out/live1.npy'
DATASET_PATH ='datagen/sets/live1/data/'

RS_BAG = 'datagen/sets/live1/t2.bag'

print('loading forest')
forest = DecisionForest.load(MODEL_OUT_NAME)
data_config = DecisionTreeDatasetConfig(DATASET_PATH)

print('compiling CUDA kernels..')
decision_tree_evaluator = DecisionTreeEvaluator()
points_ops = PointsOps()

DIM_X = 848
DIM_Y = 480
FOCAL = 615.
pts_cu = cu_array.GPUArray((DIM_Y, DIM_X, 4), dtype=np.float32)

print('initializing camera..')
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

if RS_BAG:
    config.enable_device_from_file(RS_BAG, repeat_playback=True)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

else:
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_config_json = open('hand_config.json', 'r').read()
    rs.rs400_advanced_mode(device).load_json(device_config_json)
    device.first_depth_sensor().set_option(rs.option.depth_units, 0.0001)
    config.enable_stream(rs.stream.depth, DIM_X, DIM_Y, rs.format.z16, 90)

NUM_RANDOM_GUESSES = 10000
candidate_planes_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES, 4, 4), dtype=np.float32)
num_inliers_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES), dtype=np.int32)


depth_image_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)
labels_image_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)

profile = pipeline.start(config)
if RS_BAG:
    profile.get_device().as_playback().set_real_time(False)
depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

PLANE_Z_OUTLIER_THRESHOLD = 40.

rand_generator = cu_rand.XORWOWRandomNumberGenerator(seed_getter=cu_rand.seed_getter_unique)
rand_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES, 32), dtype=np.float32)

calibrated_plane = None

def make_calibrated_plane():

    rand_generator.fill_uniform(rand_cu)
    candidate_planes_cu.fill(np.float(0))

    points_ops.make_plane_candidates(
        np.int32(NUM_RANDOM_GUESSES),
        np.int32(DIM_X),
        np.int32(DIM_Y),
        rand_cu,
        pts_cu,
        candidate_planes_cu,
        grid=((NUM_RANDOM_GUESSES // 32) + 1, 1, 1),
        block=(32, 1, 1))

    num_inliers_cu.fill(np.int32(0))
            
    # every point..
    grid_dim = (((DIM_X * DIM_Y) // 1024) + 1, 1, 1)
    block_dim = (1024, 1, 1)

    points_ops.find_plane_ransac(
        np.int32(NUM_RANDOM_GUESSES),
        np.float32(PLANE_Z_OUTLIER_THRESHOLD),
        np.int32(DIM_X * DIM_Y),
        pts_cu,
        candidate_planes_cu,
        num_inliers_cu,
        grid=grid_dim,
        block=block_dim)

    num_inliers = num_inliers_cu.get()
    best_inlier_idx = np.argmax(num_inliers)

    calibrated_plane = np.zeros((4, 4), dtype=np.float32)
    cu.memcpy_dtoh(calibrated_plane, candidate_planes_cu[best_inlier_idx].ptr)

    return calibrated_plane

last_time = time.time()

try:

    frame_num = 0

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # let camera stabilize for a few frames
        if frame_num < 15:
            frame_num += 1
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image_cu.set(depth_image)

        grid_dim = (1, (DIM_X // 32) + 1, (DIM_Y // 32) + 1)
        block_dim = (1,32,32)

        # convert depth image to points
        points_ops.deproject_points(
            np.array([1, DIM_X, DIM_Y, -1], dtype=np.int32),
            np.array([DIM_X / 2, DIM_Y / 2], dtype=np.float32),
            np.float32(FOCAL),
            depth_image_cu,
            pts_cu,
            grid=grid_dim,
            block=block_dim)

        if calibrated_plane is None:
            calibrated_plane = make_calibrated_plane()

        # every point..
        grid_dim2 = (((DIM_X * DIM_Y) // 1024) + 1, 1, 1)
        block_dim2 = (1024, 1, 1)

        points_ops.transform_points(
            np.int32(DIM_X * DIM_Y),
            pts_cu,
            calibrated_plane,
            grid=grid_dim2,
            block=block_dim2)

        points_ops.filter_points_by_plane(
            np.int32(DIM_X * DIM_Y),
            np.float32(PLANE_Z_OUTLIER_THRESHOLD),
            pts_cu,
            grid=grid_dim2,
            block=block_dim2)

        points_ops.setup_depth_image_for_forest(
            np.int32(DIM_X * DIM_Y),
            pts_cu,
            depth_image_cu,
            grid=grid_dim2,
            block=block_dim2)

        labels_image_cu.fill(np.uint16(65535))
        decision_tree_evaluator.get_labels_forest(forest, depth_image_cu, labels_image_cu)

        # final steps: these are slow.
        # can be polished if/when necessary
        labels_image_cpu = labels_image_cu.get()
        labels_image_cpu_rgba = data_config.convert_ids_to_colors(labels_image_cpu).reshape((480, 848, 4))

        labels_image_cpu_bgra = cv2.cvtColor(labels_image_cpu_rgba, cv2.COLOR_RGB2BGR)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', labels_image_cpu_bgra)
        
        cv2.waitKey(1)

        """
        now_time = time.time()
        fps = 1 / (now_time - last_time)
        last_time = now_time
        print(fps)
        """

        frame_num += 1


finally:

    # Stop streaming
    pipeline.stop()