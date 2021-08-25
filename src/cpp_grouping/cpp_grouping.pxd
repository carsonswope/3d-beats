cdef extern from "grouping.h":
    cdef cppclass _CppGrouping "CppGrouping":
        _CppGrouping() except+
        void make_groups(void* d, int dim_x, int dim_y) except+
