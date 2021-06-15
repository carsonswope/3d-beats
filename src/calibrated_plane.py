import pycuda.driver as cu
import pycuda.curandom as cu_rand
import pycuda.gpuarray as cu_array

import numpy as np

import cuda.py_nvcc_utils as py_nvcc_utils

class CalibratedPlane():

    def __init__(self, num_random_guesses, plane_z_outlier_threshold):
        self.num_random_guesses = num_random_guesses
        self.plane_z_outlier_threshold = plane_z_outlier_threshold
        self.rand_generator = cu_rand.XORWOWRandomNumberGenerator(seed_getter=cu_rand.seed_getter_unique)
        self.rand_cu = cu_array.GPUArray((self.num_random_guesses, 32), dtype=np.float32)

        self.candidate_planes_cu = cu_array.GPUArray((self.num_random_guesses, 4, 4), dtype=np.float32)
        self.num_inliers_cu = cu_array.GPUArray((self.num_random_guesses), dtype=np.int32)

        self.plane = None

        cu_mod = py_nvcc_utils.get_module('src/cuda/calibrated_plane.cu')
        self.make_plane_candidates = cu_mod.get_function('make_plane_candidates')
        self.find_plane_ransac = cu_mod.get_function('find_plane_ransac')
        self.filter_points_by_plane = cu_mod.get_function('filter_points_by_plane')

    def is_set(self):
        return not self.plane is None

    def get_mat(self):
        assert(self.is_set())
        return self.plane

    def make(self, pts_cu, img_dims):

        self.rand_generator.fill_uniform(self.rand_cu)
        self.candidate_planes_cu.fill(np.float(0))

        DIM_X = img_dims[0]
        DIM_Y = img_dims[1]

        self.make_plane_candidates(
            np.int32(self.num_random_guesses),
            np.int32(DIM_X),
            np.int32(DIM_Y),
            self.rand_cu,
            pts_cu,
            self.candidate_planes_cu,
            grid=((self.num_random_guesses // 32) + 1, 1, 1),
            block=(32, 1, 1))

        self.num_inliers_cu.fill(np.int32(0))
                
        # every point..
        grid_dim = (((DIM_X * DIM_Y) // 1024) + 1, 1, 1)
        block_dim = (1024, 1, 1)

        self.find_plane_ransac(
            np.int32(self.num_random_guesses),
            np.float32(self.plane_z_outlier_threshold),
            np.int32(DIM_X * DIM_Y),
            pts_cu,
            self.candidate_planes_cu,
            self.num_inliers_cu,
            grid=grid_dim,
            block=block_dim)

        num_inliers = self.num_inliers_cu.get()
        best_inlier_idx = np.argmax(num_inliers)

        calibrated_plane = np.zeros((4, 4), dtype=np.float32)
        cu.memcpy_dtoh(calibrated_plane, self.candidate_planes_cu[best_inlier_idx].ptr)

        self.plane = np.copy(calibrated_plane)
