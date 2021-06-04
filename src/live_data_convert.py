"""
how to generate properly labeled dataset given:
 a. realsense .bag file of painted hand
 b. # of expected colors on hand

1. for first frame:
 - find plane

2. for each frame:
 a. filter out depth pixels given plane
 b. align color frame to depth frame
 c. remove any depth pixel w/o corresponding color pixel
 d. convert color frame to HSV with S and V both 255

3. for all frames ? 1st frame ? each frame ?
 a. run e.m algorithm on colors to determine which hues are the true classes
   i. pick random hue
   ii. categorize pixels according to hue
   iii. pick new hue for each group
   iiii. goto ii until stabilization

4. for each frame:
  a. convert color frame to 'compressed' version, generate labels image

5. profit
"""

IN_PATH = './datagen/sets/live1/t2.bag'
OUT_PATH = './datagen/sets/live1/data/'

import pyrealsense2 as rs
import numpy as np
import cv2

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.curandom as cu_rand

from decision_tree import *
from cuda.points_ops import *
np.set_printoptions(suppress=True)

from PIL import Image

points_ops = PointsOps()

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(IN_PATH, repeat_playback=False)
# rs.config.enable_device_from_file(config, IN_PATH)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

pf = pipeline.start(config)
pf.get_device().as_playback().set_real_time(False)

# Create opencv window to render image in
cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

depth_intr = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

DIM_X = depth_intr.width
DIM_Y = depth_intr.height
FOCAL = np.float32(depth_intr.fx) # should be same as fy..
PP = np.array([depth_intr.ppx, depth_intr.ppy], dtype=np.float32)

# Create colorizer object
colorizer = rs.colorizer()
align = rs.align(rs.stream.depth)

NUM_RANDOM_GUESSES = 25000
candidate_planes_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES, 4, 4), dtype=np.float32)
num_inliers_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES), dtype=np.int32)

PLANE_Z_OUTLIER_THRESHOLD = 55.

NUM_COLORS = 3

rand_generator = cu_rand.XORWOWRandomNumberGenerator(seed_getter=cu_rand.seed_getter_unique)
rand_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES, 32), dtype=np.float32)

pts_cu = cu_array.GPUArray((DIM_Y, DIM_X, 4), dtype=np.float32)

depth_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)

color_image_cu = cu_array.GPUArray((DIM_Y, DIM_X, 3), dtype=np.uint8)

color_mapping_cu = cu_array.GPUArray((NUM_COLORS, 3), dtype=np.uint8)

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


color_mapping = None

NUM_TESTS = 15
NUM_ITERATIONS = 30

def make_color_mapping():

    print('making color mapping... ')

    best_colors_diffs = np.Infinity
    best_colors = np.zeros((NUM_COLORS, 3), dtype=np.uint8)

    colors_cu = cu_array.GPUArray((NUM_COLORS, 3), dtype=np.uint8)
    pixel_counts_per_group_cu = cu_array.GPUArray((NUM_COLORS, 5), dtype=np.uint64)

    for _1 in range(NUM_TESTS):

        colors = np.random.uniform(0, 255, (NUM_COLORS, 3)).astype(np.uint8)

        for _2 in range(NUM_ITERATIONS):


            colors_cu.set(colors)

            pixel_counts_per_group_cu.fill(np.uint64(0))

            grid_dim3 = ((DIM_X // 32) + 1, (DIM_Y // 32) + 1, 1)
            block_dim3 = (32,32,1)
            
            points_ops.split_pixels_by_nearest_color(
                np.int32(DIM_X),
                np.int32(DIM_Y),
                np.int32(NUM_COLORS),
                colors_cu,
                color_image_cu,
                pixel_counts_per_group_cu,
                grid=grid_dim3,
                block=block_dim3)

            pixel_counts_per_group = pixel_counts_per_group_cu.get()

            grouping_cost = np.sum(pixel_counts_per_group[:,4].view(np.float64))

            colors = (pixel_counts_per_group[:,1:4].T / pixel_counts_per_group[:,0]).T.astype(np.uint8)
        
        if grouping_cost < best_colors_diffs:
            best_colors_diffs = grouping_cost
            best_colors = np.copy(colors)
    
    del colors_cu
    del pixel_counts_per_group_cu

    print('made.')

    return best_colors

labels_image = np.zeros((DIM_Y, DIM_X), dtype=np.uint16)

color_image_rgba = np.zeros((DIM_Y, DIM_X, 4),dtype=np.uint8)

# Streaming loop
frame_count = 0
while True:
    # Get frameset of depth
    try:
        # are there double frames
        frames = pipeline.wait_for_frames(1000)
    except:
        print('concluded !')
        break

    frame_count += 1

    # Get depth frame
    depth_frame = frames.get_depth_frame()

    depth_np = np.asanyarray(depth_frame.data)
    depth_cu.set(depth_np)

    grid_dim = (1, (DIM_X // 32) + 1, (DIM_Y // 32) + 1)
    block_dim = (1,32,32)

    # convert depth image to points
    points_ops.deproject_points(
        np.array([1, DIM_X, DIM_Y, -1], dtype=np.int32),
        PP,
        FOCAL,
        depth_cu,
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

    points_ops.transform_points(
        np.int32(DIM_X * DIM_Y),
        pts_cu,
        np.linalg.inv(calibrated_plane),
        grid=grid_dim2,
        block=block_dim2)
    
    depth_cu.fill(np.uint16(0))

    points_ops.depths_from_points(
        np.array([1, DIM_X, DIM_Y, -1], dtype=np.int32),
        depth_cu,
        pts_cu,
        grid=grid_dim,
        block=block_dim)

    # copy back to cpu-side depth frame memory, so align processing block can run
    depth_cu.get(depth_np)

    frames_aligned = align.process(frames)
    color_frame = frames_aligned.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    color_image_cu.set(color_image)

    if color_mapping is None:
        color_mapping = make_color_mapping()
        color_mapping_cu.set(color_mapping)

    grid_dim3 = ((DIM_X // 32) + 1, (DIM_Y // 32) + 1, 1)
    block_dim3 = (32,32,1)

    points_ops.apply_point_mapping(
        np.int32(DIM_X),
        np.int32(DIM_Y),
        np.int32(NUM_COLORS),
        color_mapping_cu,
        color_image_cu,
        grid=grid_dim3,
        block=block_dim3)

    color_image_cu.get(color_image)
    color_image_rgba[:,:,0:3] = color_image
    # transparency...
    color_image_rgba[np.all(color_image > 0, axis=2),3] = 255

    # convert color image to
    labels_image[:,:] = 0
    for xx in range(NUM_COLORS):
        labels_image[np.where(np.all(color_image == color_mapping[xx], axis=2))] = xx + 1 # group 0 is null group, starts at 1

    Image.fromarray(labels_image).save(f'{OUT_PATH}/train{str(frame_count - 1).zfill(8)}_labels.png')
    Image.fromarray(color_image_rgba).save(f'{OUT_PATH}/train{str(frame_count - 1).zfill(8)}_colors.png')
    Image.fromarray(depth_np).save(f'{OUT_PATH}/train{str(frame_count - 1).zfill(8)}_depth.png')

    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    # Render image in opencv window
    cv2.imshow("Depth Stream", color_image)
    key = cv2.waitKey(1)
    # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break

# write json config at the end of it..
obj= {}
obj['img_dims'] = [DIM_X, DIM_Y]
obj['num_train'] = frame_count
obj['num_test'] = 0
obj['id_to_color'] = {'0': [0, 0, 0, 0]}
for c_id in range(NUM_COLORS):
    c = color_mapping[c_id]
    obj['id_to_color'][str(c_id + 1)] = [int(c[0]), int(c[1]), int(c[2]), 255]

cfg_json_file = open(f'{OUT_PATH}/config.json', 'w')
cfg_json_file.write(json.dumps(obj))
cfg_json_file.close()

# print('hi')

# DIMobprint('hi')