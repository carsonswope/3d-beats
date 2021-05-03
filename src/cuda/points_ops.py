import cuda.py_nvcc_utils as py_nvcc_utils

class PointsOps():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('src/cuda/points_ops.cu')
        self.deproject_points = cu_mod.get_function('deproject_points')
        self.transform_points = cu_mod.get_function('transform_points')
        self.make_plane_candidates = cu_mod.get_function('make_plane_candidates')
        self.find_plane_ransac = cu_mod.get_function('find_plane_ransac')
        self.filter_points_by_plane = cu_mod.get_function('filter_points_by_plane')
        self.setup_depth_image_for_forest = cu_mod.get_function('setup_depth_image_for_forest')

