import pycuda.gpuarray as cu_array

import numpy as np

import cuda.py_nvcc_utils as py_nvcc_utils


# uses mean-shift to find 'mode' of group of 2d points
# https://en.wikipedia.org/wiki/Mean_shift

class MeanShift:
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('mean_shift')
        self._run = cu_mod.get_function('run')

        self.means = None
        self.temp_sums = None

    def run(self, num_rounds, labels, num_labels, variances):

        if not self.temp_sums or self.temp_sums.shape != (num_labels, 3):
            self.means = cu_array.GPUArray((num_labels, 2), dtype=np.float64)
            self.temp_sums = cu_array.GPUArray((num_labels, 3), dtype=np.float64)

        dim_y, dim_x = labels.shape[1:]

        # every point..
        grid_dim = ((dim_x // 32) + 1, (dim_y // 32) + 1, 1)
        block_dim = (32, 32, 1)

        self.means.fill(np.float64(0.))



        for i in range(num_rounds):
            self.temp_sums.fill(np.float64(0.))

            self._run(
                labels,
                np.int32(num_labels),
                np.int32(dim_x),
                np.int32(dim_y),
                variances,
                self.means,
                np.int32(i),
                self.temp_sums,
                grid=grid_dim,
                block=block_dim)
            
            temp_sums_cpu = self.temp_sums.get()
            means_cpu = self.means.get()

            mean_shift = temp_sums_cpu[:,0:2] / temp_sums_cpu[:,2].reshape((num_labels, 1))
            means_cpu += mean_shift
            self.means.set(means_cpu)
        
        means_cpu = self.means.get()

        return means_cpu

    """
    def run(self, match, variance):

        x = np.where(match)
        x = np.array([x[0], x[1]]).T

        start_mean = np.sum(x, axis=0) / x.shape[0]
        mean = np.copy(start_mean)

        ROUNDS = 10

        for _ in range(ROUNDS):
            diff = x - mean
            dist_sq = np.sum(np.power(diff, 2), axis=1)
            e_pow = -dist_sq / (2 * variance * variance)
            f = np.power(np.e, e_pow)
            m = np.sum(f.reshape((f.shape[0], 1)) * diff, axis=0) / np.sum(f) 
            mean += m

        return mean
    """
