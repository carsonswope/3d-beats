import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.curandom as cu_rand

from decision_tree import *
from cuda.points_ops import *
np.set_printoptions(suppress=True)


from util import MAX_UINT16

import argparse

def main():

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
    OUT_PATH = args.out

    COLOR_EM_NUM_COLORS = args.colors
    COLOR_EM_NUM_TRIES = args.colors_num_restarts or 8
    COLOR_EM_ITERATIONS = args.colors_num_iterations or 32

    PLANE_RANSAC_NUM_CANDIDATES = args.plane_num_iterations or 25000
    PLANE_Z_THRESHOLD = args.plane_z_threshold

    MAX_IMAGES = args.max_images or np.Infinity

    FRAMES_TIMESTAMP_MAX_DIFF = args.frames_timestamp_max_diff or 6.

    FRAMES_PER_RECOMPUTE_PLANE = 20

    MASK_MODEL_PATH = args.mask_model
    MASK_LABEL = args.mask_label

    if (MASK_MODEL_PATH and not MASK_LABEL) or (not MASK_MODEL_PATH and MASK_LABEL):
        print('--mask_path and --mask_label are both required if using mask')
        return

    if MASK_MODEL_PATH:
        mask_model = DecisionForest.load(MASK_MODEL_PATH)
    else:
        mask_model = None

    points_ops = PointsOps()
    decision_tree_evaluator = DecisionTreeEvaluator()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(IN_PATH, repeat_playback=False)

    config.enable_stream(rs.stream.depth, rs.format.z16)
    config.enable_stream(rs.stream.color, rs.format.rgb8)

    pf = pipeline.start(config)
    pf.get_device().as_playback().set_real_time(False)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    depth_intr = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    DIM_X = depth_intr.width
    DIM_Y = depth_intr.height
    FOCAL = np.float32(depth_intr.fx) # should be same as fy..
    PP = np.array([depth_intr.ppx, depth_intr.ppy], dtype=np.float32)

    align = rs.align(rs.stream.depth)

    candidate_planes_cu = cu_array.GPUArray((PLANE_RANSAC_NUM_CANDIDATES, 4, 4), dtype=np.float32)
    num_inliers_cu = cu_array.GPUArray(PLANE_RANSAC_NUM_CANDIDATES, dtype=np.int32)

    rand_generator = cu_rand.XORWOWRandomNumberGenerator(seed_getter=cu_rand.seed_getter_unique)
    rand_cu = cu_array.GPUArray((PLANE_RANSAC_NUM_CANDIDATES, 32), dtype=np.float32)

    pts_cu = cu_array.GPUArray((DIM_Y, DIM_X, 4), dtype=np.float32)

    depth_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)

    if mask_model:
        mask_labels = np.zeros((1, DIM_Y, DIM_X), dtype=np.uint16)
        mask_labels_cu = cu_array.GPUArray((1, DIM_Y, DIM_X), dtype=np.uint16)

    color_image_cu = cu_array.GPUArray((DIM_Y, DIM_X, 3), dtype=np.uint8)

    color_mapping_cu = cu_array.GPUArray((COLOR_EM_NUM_COLORS, 3), dtype=np.uint8)

    calibrated_plane = None

    def make_calibrated_plane():

        rand_generator.fill_uniform(rand_cu)
        candidate_planes_cu.fill(np.float(0))

        points_ops.make_plane_candidates(
            np.int32(PLANE_RANSAC_NUM_CANDIDATES),
            np.int32(DIM_X),
            np.int32(DIM_Y),
            rand_cu,
            pts_cu,
            candidate_planes_cu,
            grid=((PLANE_RANSAC_NUM_CANDIDATES // 32) + 1, 1, 1),
            block=(32, 1, 1))

        num_inliers_cu.fill(np.int32(0))
                
        # every point..
        grid_dim = (((DIM_X * DIM_Y) // 1024) + 1, 1, 1)
        block_dim = (1024, 1, 1)

        points_ops.find_plane_ransac(
            np.int32(PLANE_RANSAC_NUM_CANDIDATES),
            np.float32(PLANE_Z_THRESHOLD),
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

    def make_color_mapping():

        print('making color mapping... ')

        best_colors_diffs = np.Infinity
        best_colors = np.zeros((COLOR_EM_NUM_COLORS, 3), dtype=np.uint8)

        colors_cu = cu_array.GPUArray((COLOR_EM_NUM_COLORS, 3), dtype=np.uint8)
        pixel_counts_per_group_cu = cu_array.GPUArray((COLOR_EM_NUM_COLORS, 5), dtype=np.uint64)

        for _1 in range(COLOR_EM_NUM_TRIES):

            colors = np.random.uniform(0, 255, (COLOR_EM_NUM_COLORS, 3)).astype(np.uint8)

            for _2 in range(COLOR_EM_ITERATIONS):


                colors_cu.set(colors)

                pixel_counts_per_group_cu.fill(np.uint64(0))

                grid_dim3 = ((DIM_X // 32) + 1, (DIM_Y // 32) + 1, 1)
                block_dim3 = (32,32,1)
                
                points_ops.split_pixels_by_nearest_color(
                    np.int32(DIM_X),
                    np.int32(DIM_Y),
                    np.int32(COLOR_EM_NUM_COLORS),
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
    while frame_count < MAX_IMAGES:
        # Get frameset of depth
        try:
            frames = pipeline.wait_for_frames(1000)
            df_time = frames.get_depth_frame().get_timestamp()
            cf_time = frames.get_color_frame().get_timestamp()
            # only process frame pairs whose timestamps overlap reasonably well
            if np.abs(df_time - cf_time) > FRAMES_TIMESTAMP_MAX_DIFF:
                continue
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

        if calibrated_plane is None or frame_count % FRAMES_PER_RECOMPUTE_PLANE == 0:
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
            np.float32(PLANE_Z_THRESHOLD),
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
        # color_image_cu.set(color_image)

        # how you could crop if you wanted..
        # should be an input option??
        # color_image[:,750:] = np.array([0, 0, 0], dtype=np.uint8)

        if mask_model:

            depth_np[depth_np == 0] = MAX_UINT16
            depth_cu.set(depth_np)

            mask_labels_cu.fill(np.uint16(0))
            decision_tree_evaluator.get_labels_forest(mask_model, depth_cu, mask_labels_cu)
            mask_labels_cu.get(mask_labels)
            color_image[mask_labels[0] != MASK_LABEL] = np.array([0, 0, 0], dtype=np.uint8)
            depth_np[depth_np == MAX_UINT16] = 0
            depth_cu.set(depth_np)

        color_image_cu.set(color_image)

        if color_mapping is None:
            color_mapping = make_color_mapping()
            color_mapping_cu.set(color_mapping)

        grid_dim3 = ((DIM_X // 32) + 1, (DIM_Y // 32) + 1, 1)
        block_dim3 = (32,32,1)

        points_ops.apply_point_mapping(
            np.int32(DIM_X),
            np.int32(DIM_Y),
            np.int32(COLOR_EM_NUM_COLORS),
            color_mapping_cu,
            color_image_cu,
            grid=grid_dim3,
            block=block_dim3)

        # render raw labels image from color image (1..n)
        color_image_cu.get(color_image)
        labels_image[:,:] = 0
        for xx in range(COLOR_EM_NUM_COLORS):
            labels_image[np.where(np.all(color_image == color_mapping[xx], axis=2))] = xx + 1 # group 0 is null group, starts at 1
        Image.fromarray(labels_image).save(f'{OUT_PATH}/{str(frame_count - 1).zfill(8)}_labels.png')

        # debug rendering of labels image
        color_image_rgba[:,:,3] = 0
        color_image_rgba[:,:,0:3] = color_image
        color_image_rgba[np.any(color_image > 0, axis=2),3] = 255
        Image.fromarray(color_image_rgba).save(f'{OUT_PATH}/{str(frame_count - 1).zfill(8)}_labels_rgba.png')

        # render raw depth image
        depth_np[depth_np == 0] = MAX_UINT16
        Image.fromarray(depth_np).save(f'{OUT_PATH}/{str(frame_count - 1).zfill(8)}_depth.png')

        # debug rendering of
        depth_rgba_np = np.zeros((DIM_Y, DIM_X, 4), dtype=np.uint8)
        depth_rgba_np[depth_np == MAX_UINT16 ] = np.array([167, 195, 162, 255], dtype=np.uint8)
        active_coords = np.where(depth_np < MAX_UINT16)
        max_depth = np.max(depth_np[depth_np < MAX_UINT16])
        min_depth = np.min(depth_np[depth_np < MAX_UINT16])
        norm_depths = (255. * (1. - (depth_np[depth_np < MAX_UINT16] - (min_depth * 1.)) / (max_depth - min_depth))).astype(np.uint8)
        depth_rgba_np[active_coords[0], active_coords[1], 0] = norm_depths
        depth_rgba_np[active_coords[0], active_coords[1], 1] = norm_depths
        depth_rgba_np[active_coords[0], active_coords[1], 2] = norm_depths
        depth_rgba_np[active_coords[0], active_coords[1], 3] = 255
        Image.fromarray(depth_rgba_np).save(f'{OUT_PATH}/{str(frame_count - 1).zfill(8)}_depth_rgba.png')

        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Render image in opencv window
        cv2.imshow("Depth Stream", color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break

    # write json config as entry point into model
    obj= {}
    obj['img_dims'] = [DIM_X, DIM_Y]
    obj['num_images'] = frame_count
    obj['id_to_color'] = {'0': [0, 0, 0, 0]}
    for c_id in range(COLOR_EM_NUM_COLORS):
        c = color_mapping[c_id]
        obj['id_to_color'][str(c_id + 1)] = [int(c[0]), int(c[1]), int(c[2]), 255]

    cfg_json_file = open(f'{OUT_PATH}/config.json', 'w')
    cfg_json_file.write(json.dumps(obj))
    cfg_json_file.close()

if __name__ == '__main__':
    main()
