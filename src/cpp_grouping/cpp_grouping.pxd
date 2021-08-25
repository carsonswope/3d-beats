cdef extern from "grouping.h":
    cdef cppclass _CppGrouping "CppGrouping":
        _CppGrouping() except+
        void make_groups(void*, int, int, void*, void*, float) except+
