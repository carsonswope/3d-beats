#ifndef __CU_UTILS
#define __CU_UTILS

template<typename T>
class BinaryTree {
private:
    const int els_per_node;
    T* data;
    const int num_levels;
public:

    explicit __device__ BinaryTree(T* _data, int _els_per_node, int _num_levels):
        data(_data),
        els_per_node(_els_per_node),
        num_levels(_num_levels) {}

    __device__ T* get_ptr(int level, int node) {
        assert(level < num_levels);
        const int MAX_NODES_FOR_LEVEL = 1 << level;
        assert(node < MAX_NODES_FOR_LEVEL);

        const int idx_offset = (1 << level) - 1;
        return data + ((idx_offset + node) * els_per_node);
    }
};

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

// durr...
// https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

#endif

