#ifndef __CU_UTILS
#define __CU_UTILS

// Wrapper for a 3d array, like the one in numpy!
template <typename T>
class Array3d {
private:
    const int3 dims;
    T* data;
    T default_value; // default return for [] if out of bounds!
    const int max_idx;
public:

    explicit __device__ Array3d(T* _data, int3 _dims, T _default=0) :
        data(_data),
        dims(_dims),
        default_value(_default),
        max_idx(_dims.x * _dims.y * dims.z) {}

    __device__ T get(const int3 i) {
        const int idx = get_idx(i);
        if (idx < 0 || idx >= max_idx) { return default_value; }
        return data[idx];
    }

    __device__ T* get_ptr(const int3 i) {
        const int idx = get_idx(i);
        if (idx < 0 || idx >= max_idx) { return nullptr; }
        return data + idx;
    }

    __device__ void set(const int3 i, T val) {
        const int idx = get_idx(i);
        if (idx < 0 || idx >= max_idx) { assert(false); }
        data[idx] = val;
    }

private:
    __device__ int get_idx(const int3 i) {
        return (i.x * dims.y * dims.z) + (i.y * dims.z) + (i.z);
    }
};

// Wrapper for a 2d array, like the one in numpy!
template <typename T>
class Array2d {
private:
    const int2 dims;
    T* data;
    T default_value; // default return for [] if out of bounds!
    const int max_idx;
public:

    explicit __device__ Array2d(T* _data, int2 _dims, T _default=0) :
        data(_data),
        dims(_dims),
        default_value(_default),
        max_idx(_dims.x * _dims.y) {}

    __device__ T get(const int2 i) {
        const int idx = get_idx(i);
        if (idx < 0 || idx >= max_idx) { return default_value; }
        return data[idx];
    }

    __device__ T* get_ptr(const int2 i) {
        const int idx = get_idx(i);
        if (idx < 0 || idx >= max_idx) { return nullptr; }
        return data + idx;
    }

    __device__ void set(const int2 i, T val) {
        const int idx = get_idx(i);
        if (idx < 0 || idx >= max_idx) { assert(false); }
        data[idx] = val;
    }

private:
    __device__ int get_idx(const int2 i) {
        return (i.x * dims.y) + (i.y);
    }
};

#endif