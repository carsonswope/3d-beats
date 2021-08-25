from libc.stdint cimport uintptr_t
# from libcpp cimport b
from cpp_grouping cimport _CppGrouping

import numpy as np

cdef class CppGrouping:
    cdef _CppGrouping* g

    def __cinit(self):
        self.g = new _CppGrouping()
    
    def make_groups(self, in_arr: np.array, coords_arr: np.array, group_info_arr: np.array, pct_thresh):
        cdef uintptr_t in_ptr = in_arr.__array_interface__['data'][0]
        cdef uintptr_t coords_ptr = coords_arr.__array_interface__['data'][0]
        cdef uintptr_t group_info_ptr = group_info_arr.__array_interface__['data'][0]
        self.g.make_groups(
            <void*>in_ptr,
            <int>in_arr.shape[1],
            <int>in_arr.shape[0],
            <void*>coords_ptr,
            <void*>group_info_ptr,
            <float>pct_thresh)
