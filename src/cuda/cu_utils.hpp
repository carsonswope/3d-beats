#ifndef __CU_UTILS
#define __CU_UTILS

// #define GLM_FORCE_CUDA
#define GLM_SWIZZLE 
#include <glm/glm.hpp>

#define MAX_UINT16 65535

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

struct _mat4 {
    float m[16];
};

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
        if (idx == -1) return default_value;
        return data[idx];
    }

    __device__ T* get_ptr(const int3 i) {
        const int idx = get_idx(i);
        if (idx == -1) { return nullptr; }
        return data + idx;
    }

    __device__ void set(const int3 i, T val) {
        const int idx = get_idx(i);
        if (idx == -1) { 
            printf("Set out of bounds!\n");
            assert(false);
        }
        data[idx] = val;
    }

    __device__ int get_idx(const int3 i) {
        const bool out_of_bounds =
            i.x < 0 || i.x >= dims.x ||
            i.y < 0 || i.y >= dims.y ||
            i.z < 0 || i.z >= dims.z;
        if (out_of_bounds) return -1;
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

    __device__ int get_idx(const int2 i) {
        const bool out_of_bounds = 
            i.x < 0 || i.x >= dims.x ||
            i.y < 0 || i.y >= dims.y;
        if (out_of_bounds) return -1;
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

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

// https://stackoverflow.com/questions/12626096/why-has-atomicadd-not-been-implemented-for-doubles
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif


#endif
