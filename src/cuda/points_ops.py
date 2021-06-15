import cuda.py_nvcc_utils as py_nvcc_utils

class PointsOps():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('src/cuda/points_ops.cu')
        self.deproject_points = cu_mod.get_function('deproject_points')
        self.depths_from_points = cu_mod.get_function('depths_from_points')
        self.transform_points = cu_mod.get_function('transform_points')
        self.setup_depth_image_for_forest = cu_mod.get_function('setup_depth_image_for_forest')
        self.apply_point_mapping = cu_mod.get_function('apply_point_mapping')
        self.split_pixels_by_nearest_color = cu_mod.get_function('split_pixels_by_nearest_color')
