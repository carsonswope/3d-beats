import cuda.py_nvcc_utils as py_nvcc_utils
from engine.buffer import GpuBuffer
from util import PagelockedCounter, make_grid

import numpy as np
import scipy.stats

def gaussian_kernel(k_size, sigma):
    assert k_size % 2 == 1, 'kernel must be odd'
    l = (k_size // 2)
    kern1d = scipy.stats.norm.pdf(np.linspace(-l, l, k_size), 0., sigma)
    kern2d = np.outer(kern1d, kern1d)
    return (kern2d/kern2d.sum()).astype(np.float32)

class PointsOps():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('points_ops')
        self.deproject_points = cu_mod.get_function('deproject_points')
        self.depths_from_points = cu_mod.get_function('depths_from_points')
        self.transform_points = cu_mod.get_function('transform_points')
        self.convert_0s_to_maxuint = cu_mod.get_function('convert_0s_to_maxuint')
        self.remove_missing_3d_points_from_depth_image = cu_mod.get_function('remove_missing_3d_points_from_depth_image')
        self.setup_depth_image_for_forest = cu_mod.get_function('setup_depth_image_for_forest')
        self.apply_point_mapping = cu_mod.get_function('apply_point_mapping')
        self.split_pixels_by_nearest_color = cu_mod.get_function('split_pixels_by_nearest_color')
        self.make_rgba_from_labels = cu_mod.get_function('make_rgba_from_labels')

        self.stencil_depth_image_by_group = cu_mod.get_function('stencil_depth_image_by_group')

        self.write_pixel_groups_to_stencil_image = cu_mod.get_function('write_pixel_groups_to_stencil_image')
        self.shrink_image = cu_mod.get_function('shrink_image')
        self.grow_groups = cu_mod.get_function('grow_groups')
        self.flip_x = cu_mod.get_function('flip_x')

        self.MAX_FILTER_SIZE = 41
        self._gaussian_filter = GpuBuffer((self.MAX_FILTER_SIZE*self.MAX_FILTER_SIZE,), dtype=np.float32)
        self._cached_filter_params = None
        self._gaussian_depth_filter = cu_mod.get_function('gaussian_depth_filter')

        self.make_depth_rgba = cu_mod.get_function('make_depth_rgba')

        self._make_triangles = cu_mod.get_function('make_triangles')
        self._triangle_count = GpuBuffer((1,), dtype=np.uint64)

    
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

    def gaussian_depth_filter(self, d_in: GpuBuffer, d_out: GpuBuffer, sigma:float, k_size:int=5):

        assert k_size <= self.MAX_FILTER_SIZE
        assert len(d_in.shape) == 2
        assert d_in.shape == d_out.shape
        assert d_in.dtype == np.uint16
        assert d_out.dtype == np.uint16

        dim_y, dim_x = d_in.shape

        b = (32, 32, 1)
        g = make_grid((dim_x, dim_y, 1), b)

        if self._cached_filter_params is None or self._cached_filter_params != (sigma, k_size):
            self._cached_filter_params = (sigma, k_size)
            k = gaussian_kernel(k_size, sigma).flatten()
            self._gaussian_filter.cu()[0:k.shape[0]].set(k)

        # could be 'separable', but , fuck it!
        self._gaussian_depth_filter(
            np.array([dim_x, dim_y], dtype=np.int32),
            np.int32(k_size),
            self._gaussian_filter.cu(),
            d_in.cu(),
            d_out.cu(),
            grid=g,
            block=b)
    

    def get_pixel_groups_cpu(self, depth_image: np.array, pct_thresh=0.05):

        dim_y, dim_x = depth_image.shape

        # 0: not seen
        # 1: seen
        seen = np.zeros(depth_image.shape, dtype=np.uint8)
        seen[depth_image == 0] = 1

        d_not0 = np.where(depth_image > 0)

        right_group = None
        left_group = None

        dirs = [(-1,0),(1,0),(0,-1),(0,1)]

        for y,x in zip(d_not0[0], d_not0[1]):

            if seen[y,x]:
                continue
            elif depth_image[y,x] == 0:
                seen[y,x] = 1
            else: # not seen
                # new group!
                seen[y,x] = 1
                to_visit = [(y,x)]
                g = []
                while len(to_visit):
                    c = to_visit.pop()
                    g.append(c)
                    for d in dirs:
                        new_c = (c[0] + d[0], c[1] + d[1])
                        if new_c[0] >= 0 and new_c[1] >= 0 and new_c[0] < dim_y and new_c[1] < dim_x and not seen[new_c[0], new_c[1]]:
                            seen[new_c[0], new_c[1]] = 1
                            if depth_image[new_c[0], new_c[1]]:
                                to_visit.append(new_c)

                if len(g) / (dim_y * dim_x) > pct_thresh:

                    center_y, center_x = np.sum(np.array(g), axis=0) / len(g)
                    if center_x < (dim_x/2):
                        if right_group is None or len(g) > len(right_group):
                            right_group = g
                    else:
                        if left_group is None or len(g) > len(left_group):
                            left_group = g

        return (right_group, left_group)
