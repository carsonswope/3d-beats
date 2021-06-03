import pptk
import numpy as np

from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array

from cuda.points_ops import PointsOps

np.set_printoptions(suppress=True)

points_ops = PointsOps()

d = np.array(Image.open('./live_depth_filtered2.png')).astype(np.uint16)
# d = np.array(Image.open('./datagen/genstereo-filterable/train00000000_depth.png')).astype(np.uint16)
d_cu = cu_array.to_gpu(d)

DIM_X = 848
DIM_Y = 480
FOCAL = 615.
pts_cu = cu_array.GPUArray((DIM_Y, DIM_X, 4), dtype=np.float32)

grid_dim = (1, (DIM_X // 32) + 1, (DIM_Y // 32) + 1)
block_dim = (1,32,32)

points_ops.deproject_points(
    np.array([1, DIM_X, DIM_Y, -1], dtype=np.int32),
    np.array([DIM_X / 2, DIM_Y / 2], dtype=np.float32),
    np.float32(FOCAL),
    d_cu,
    pts_cu,
    grid=grid_dim,
    block=block_dim)

pts_cpu = pts_cu.get()

NUM_RANDOM_GUESSES = 256

candidate_planes = np.zeros((NUM_RANDOM_GUESSES, 4, 4), dtype=np.float32)

for j in range(NUM_RANDOM_GUESSES):
    pts = []
    for i in range(3):
        found=False
        while not found:
            
            pt = pts_cpu[np.random.randint(0, DIM_Y), np.random.randint(0, DIM_X)]
            if pt[3] == 1.:
                found=True
                pts.append(pt)
    v0 = (pts[1] - pts[0])[0:3]
    v0 /= np.linalg.norm(v0)
    v1 = (pts[2] - pts[0])[0:3]
    v1 /= np.linalg.norm(v1)

    z_axis = np.cross(v0, v1)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = v0

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    tf_mat = np.identity(4, dtype=np.float32)

    tf_mat[0,0:3] = x_axis
    tf_mat[1,0:3] = y_axis
    tf_mat[2,0:3] = z_axis

    tf_mat[0:3,3] = -pts[0][0:3]
    candidate_planes[j] = tf_mat

candidate_planes_cu = cu_array.to_gpu(candidate_planes)

PLANE_Z_OUTLIER_THRESHOLD = 150

num_inliers_cu = cu_array.GPUArray((NUM_RANDOM_GUESSES), dtype=np.int32)
num_inliers_cu.fill(np.int32(0))

# every point..
grid_dim2 = (((DIM_X * DIM_Y) // 1024) + 1, 1, 1)
block_dim2 = (1024, 1, 1)

points_ops.find_plane_ransac(
    np.int32(NUM_RANDOM_GUESSES),
    np.float32(PLANE_Z_OUTLIER_THRESHOLD),
    np.int32(DIM_X * DIM_Y),
    pts_cu,
    candidate_planes_cu,
    num_inliers_cu,
    grid=grid_dim2,
    block=block_dim2)

num_inliers = num_inliers_cu.get()
best_inlier_idx = np.argmax(num_inliers)

best_plane = candidate_planes[best_inlier_idx]


points_ops.transform_points(
    np.int32(DIM_X * DIM_Y),
    pts_cu,
    best_plane,
    grid=grid_dim2,
    block=block_dim2)

PLANE_Z_FILTER_THRESHOLD = 15.

"""
points_ops.filter_points_by_plane(
    np.int32(DIM_X * DIM_Y),
    np.float32(PLANE_Z_FILTER_THRESHOLD),
    pts_cu,
    grid=grid_dim2,
    block=block_dim2)

"""
pts_cpu = pts_cu.get()


# pts_cpu = pts_cu.get()


v = pptk.viewer(pts_cpu[:,:,0:3])
v.set(lookat = (0., 0., 0.))
v.wait()