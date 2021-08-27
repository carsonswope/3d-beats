import pycuda.gpuarray as cu_array

import numpy as np

import cuda.py_nvcc_utils as py_nvcc_utils
from engine.buffer import GpuBuffer

class FitMesh:
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('fit_mesh')
        self._calc_image_cost = cu_mod.get_function('calc_image_cost')

        self._cost = GpuBuffer((1,), dtype=np.float32)

    def calc_image_cost(self, orig_depth_img, render_depth_img, labels_img, target_label):
        assert orig_depth_img.shape == render_depth_img.shape and orig_depth_img.shape == labels_img.shape
        assert orig_depth_img.dtype == np.uint16 and render_depth_img.dtype == np.uint16 and labels_img.dtype == np.uint16

        _, dim_y, dim_x = orig_depth_img.shape

        self._cost.cu().fill(np.float32(0.))

        grid_dim=((dim_x // 32) + 1, (dim_y // 32) + 1, 1)
        block_dim=(32,32,1)

        self._calc_image_cost(
            np.int32(dim_x),
            np.int32(dim_y),
            orig_depth_img.cu(),
            render_depth_img.cu(),
            labels_img.cu(),
            np.uint16(target_label),
            self._cost.cu(),
            grid=grid_dim,
            block=block_dim)

        cost_out = self._cost.cu().get()[0]

        return cost_out


        print('hi')
        # self._image_cost