from libc.stdint cimport uintptr_t
# from libcpp cimport b
from cpp_grouping cimport _CppGrouping

import numpy as np

cdef class CppGrouping:
    cdef _CppGrouping* g

    def __cinit(self):
        self.g = new _CppGrouping()
    
    def make_groups(self, in_arr: np.array):
        cdef uintptr_t in_ptr = in_arr.__array_interface__['data'][0]
        self.g.make_groups(
            <void*>in_ptr,
            <int>in_arr.shape[1],
            <int>in_arr.shape[0])
