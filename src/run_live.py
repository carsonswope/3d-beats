import pyrealsense2 as rs
import numpy as np
import cv2

import pycuda.driver as cu
import pycuda.autoinit

from decision_tree import *
np.set_printoptions(suppress=True)

print('loading forest')
forest = DecisionForest.load('models_out/model.npy')
data_config = DecisionTreeDatasetConfig('datagen/genstereo/', load_test=False, load_train=False)

print('compiling CUDA kernels..')
decision_tree_evaluator = DecisionTreeEvaluator()

print('initializing camera..')
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

device.first_depth_sensor().set_option(rs.option.depth_units, 0.0001)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)


depth_image_cu = cu_array.GPUArray((1, 480, 848), dtype=np.uint16)
labels_image_cu = cu_array.GPUArray((1, 480, 848), dtype=np.uint16)

profile = pipeline.start(config)
depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

try:

    i = 0

    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        """

        # Convert images to numpy arrays
        depth_image_f = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_image_f *= 1.8
        depth_image = depth_image_f.astype(np.uint16)
        depth_image[depth_image == 0] = 65535
        depth_image_cu.set(depth_image)

        labels_image_cu.fill(np.uint16(65535))
        decision_tree_evaluator.get_labels_forest(forest, depth_image_cu, labels_image_cu)

        labels_image_cpu = labels_image_cu.get()
        labels_image_cpu_rgba = data_config.convert_ids_to_colors(labels_image_cpu).reshape((480, 848, 4))

        labels_image_cpu_bgra = cv2.cvtColor(labels_image_cpu_rgba, cv2.COLOR_RGB2BGR)

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', labels_image_cpu_bgra)

        """


        
        cv2.waitKey(1)


finally:

    # Stop streaming
    pipeline.stop()