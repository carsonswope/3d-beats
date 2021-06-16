import pyrealsense2 as rs
import numpy as np
import cv2
import math

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.curandom as cu_rand

from decision_tree import *
from cuda.points_ops import *
from calibrated_plane import *
from cuda.mean_shift import *

np.set_printoptions(suppress=True)

import argparse

def main():

    parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
    parser.add_argument('-m0', '--model0', nargs='?', required=True, type=str, help='Path to .npy model input file for arm/hand/fingers/thumb')
    parser.add_argument('-m1', '--model1', nargs='?', required=True, type=str, help='Path to .npy model input file for bones 1-2-3 on fingers')
    parser.add_argument('-m2', '--model2', nargs='?', required=True, type=str, help='Path to .npy model input file for fingers 1-2-3-4')
    # parser.add_argument('-d', '--data', nargs='?', required=True, type=str, help='Directory holding data')
    parser.add_argument('--rs_bag', nargs='?', required=False, type=str, help='Path to optional input realsense .bag file to use instead of live camera stream')
    parser.add_argument('--plane_num_iterations', nargs='?', required=False, type=int, help='Num random planes to propose looking for best fit')
    parser.add_argument('--plane_z_threshold', nargs='?', required=True, type=float, help='Z-value threshold in plane coordinates for clipping depth image pixels')
    args = parser.parse_args()

    # MODEL_OUT_NAME = args.model
    # DATASET_PATH = args.data
    RS_BAG = args.rs_bag

    NUM_RANDOM_GUESSES = args.plane_num_iterations or 25000
    PLANE_Z_OUTLIER_THRESHOLD = args.plane_z_threshold

    calibrated_plane = CalibratedPlane(NUM_RANDOM_GUESSES, PLANE_Z_OUTLIER_THRESHOLD)

    print('loading forest')
    m0 = DecisionForest.load(args.model0)
    m1 = DecisionForest.load(args.model1)
    m2 = DecisionForest.load(args.model2)

    print('compiling CUDA kernels..')
    decision_tree_evaluator = DecisionTreeEvaluator()
    points_ops = PointsOps()

    print('initializing camera..')
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    if RS_BAG:
        config.enable_device_from_file(RS_BAG, repeat_playback=True)
        config.enable_stream(rs.stream.depth, rs.format.z16)
        config.enable_stream(rs.stream.color, rs.format.rgb8)

    else:
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_config_json = open('hand_config.json', 'r').read()
        rs.rs400_advanced_mode(device).load_json(device_config_json)
        device.first_depth_sensor().set_option(rs.option.depth_units, 0.0001)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)

    profile = pipeline.start(config)
    if RS_BAG:
        profile.get_device().as_playback().set_real_time(False)
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intrin = depth_profile.get_intrinsics()

    DIM_X = depth_intrin.width
    DIM_Y = depth_intrin.height

    FOCAL = depth_intrin.fx
    PP = np.array([depth_intrin.ppx, depth_intrin.ppy], dtype=np.float32)
    pts_cu = cu_array.GPUArray((DIM_Y, DIM_X, 4), dtype=np.float32)
    depth_image_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)

    labels_image0_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)
    # labels_image0_cpu = labels_image0_cu.get()

    labels_image1_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)
    # labels_image1_cpu = labels_image1_cu.get()

    labels_image2_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)
    # labels_image2_cpu = labels_image2_cu.get()

    labels_image_composite_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)
    # labels_image_composite_cpu = labels_image_composite_cu.get()

    labels_image_rgba_cpu = np.zeros((DIM_Y, DIM_X, 4), dtype=np.uint8)
    labels_image_rgba_cu = cu_array.to_gpu(labels_image_rgba_cpu)

    mean_shift = MeanShift()

    labels_images_ptrs = cu.pagelocked_zeros((3,), dtype=np.int64)
    labels_images_ptrs[0] = labels_image0_cu.__cuda_array_interface__['data'][0]
    labels_images_ptrs[1] = labels_image2_cu.__cuda_array_interface__['data'][0]
    labels_images_ptrs[2] = labels_image1_cu.__cuda_array_interface__['data'][0]
    labels_images_ptrs_cu = cu_array.to_gpu(labels_images_ptrs)



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
    mean_shift_variances_cu = cu_array.to_gpu(mean_shift_variances)

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
    labels_conditions_cu = cu_array.to_gpu(labels_conditions)
    NUM_COMPOSITE_CLASSES = 15 # 1-15

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
    labels_colors_cu = cu_array.to_gpu(labels_colors)

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
                PP,
                np.float32(FOCAL),
                depth_image_cu,
                pts_cu,
                grid=grid_dim,
                block=block_dim)

            if not calibrated_plane.is_set() or frame_num % 45 == 0:
                if calibrated_plane.is_set():
                    # attempt to improve plane..
                    calibrated_plane.make(pts_cu, (DIM_X, DIM_Y), calibrated_plane.get_mat())
                else:
                    # initialize plane..
                    calibrated_plane.make(pts_cu, (DIM_X, DIM_Y))

            # every point..
            grid_dim2 = (((DIM_X * DIM_Y) // 1024) + 1, 1, 1)
            block_dim2 = (1024, 1, 1)

            points_ops.transform_points(
                np.int32(DIM_X * DIM_Y),
                pts_cu,
                calibrated_plane.get_mat(),
                grid=grid_dim2,
                block=block_dim2)

            calibrated_plane.filter_points_by_plane(
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

            labels_image0_cu.fill(MAX_UINT16)
            labels_image1_cu.fill(MAX_UINT16)
            labels_image2_cu.fill(MAX_UINT16)
            labels_image_composite_cu.fill(MAX_UINT16)

            decision_tree_evaluator.get_labels_forest(m0, depth_image_cu, labels_image0_cu)
            decision_tree_evaluator.get_labels_forest(m1, depth_image_cu, labels_image1_cu)
            decision_tree_evaluator.get_labels_forest(m2, depth_image_cu, labels_image2_cu)

            mean_shift.make_composite_labels_image(
                labels_images_ptrs_cu,
                DIM_X,
                DIM_Y,
                labels_conditions_cu,
                labels_image_composite_cu)

            label_means = mean_shift.run(
                6,
                labels_image_composite_cu,
                NUM_COMPOSITE_CLASSES,
                mean_shift_variances_cu)

            # generate RGB image for debugging!
            labels_image_rgba_cu.fill(np.uint8(0))
            points_ops.make_rgba_from_labels(
                np.uint32(DIM_X),
                np.uint32(DIM_Y),
                np.uint32(NUM_COMPOSITE_CLASSES),
                labels_image_composite_cu,
                labels_colors_cu,
                labels_image_rgba_cu,
                grid = ((DIM_X // 32) + 1, (DIM_Y // 32) + 1, 1),
                block = (32,32,1))

            labels_image_rgba_cu.get(labels_image_rgba_cpu)

            for m in label_means:
                if not math.isnan(m[0]):
                    my = int(m[1])
                    mx = int(m[0])
                    if my > 0 and my < DIM_Y - 1 and mx > 0 and mx < DIM_X - 1:
                        labels_image_rgba_cpu[my, mx, :] = np.array([255, 255, 255, 255], dtype=np.uint8)
                        labels_image_rgba_cpu[my+1, mx, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
                        labels_image_rgba_cpu[my-1, mx, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
                        labels_image_rgba_cpu[my, mx+1, :] = np.array([0, 0, 0, 255], dtype=np.uint8)
                        labels_image_rgba_cpu[my, mx-1, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

            # labels_image_cpu_bgra = cv2.cvtColor(labels_image_rgba_cpu, cv2.COLOR_RGB2BGR)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', labels_image_rgba_cpu)
            cv2.waitKey(1)

            frame_num += 1


    finally:

        # Stop streaming
        pipeline.stop()

if __name__ == '__main__':
    main()
