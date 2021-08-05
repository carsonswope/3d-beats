import cuda.py_nvcc_utils as py_nvcc_utils
from engine.buffer import GpuBuffer
from util import PagelockedCounter

import numpy as np

class PointsOps():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('src/cuda/points_ops.cu')
        self.deproject_points = cu_mod.get_function('deproject_points')
        self.depths_from_points = cu_mod.get_function('depths_from_points')
        self.transform_points = cu_mod.get_function('transform_points')
        self.setup_depth_image_for_forest = cu_mod.get_function('setup_depth_image_for_forest')
        self.apply_point_mapping = cu_mod.get_function('apply_point_mapping')
        self.split_pixels_by_nearest_color = cu_mod.get_function('split_pixels_by_nearest_color')
        self.make_rgba_from_labels = cu_mod.get_function('make_rgba_from_labels')

        self._make_triangles = cu_mod.get_function('make_triangles')
        self._triangle_count = GpuBuffer((1,), dtype=np.uint64)
        # self._triangle_count = PagelockedCounter()
    
    # returns num triangles!
    def make_triangles(self, DIM_X, DIM_Y, pts: GpuBuffer, idxes: GpuBuffer):
        self._triangle_count.cu().set(np.array([0], dtype=np.uint64))

        BLOCK_DIM = 32
        

        self._make_triangles(
            np.int32(DIM_X),
            np.int32(DIM_Y),
            self._triangle_count.cu(),
            pts.cu(),
            idxes.cu(),
            grid=((DIM_X // BLOCK_DIM) + 1, (DIM_Y // BLOCK_DIM) + 1, 1),
            block=(32, 32, 1))

        num_triangles = self._triangle_count.cu().get()[0]
        return num_triangles

