#ifndef __DECISION_TREE_COMMON
#define __DECISION_TREE_COMMON

#define MAX_UINT16 65535

#include <cu_utils.hpp>

__device__ float compute_feature(Array3d<uint16>& img_depth, int img_idx, int2 coord, float2 u, float2 v, float uv_scale=1.) {
    // uv_scale: constant scaling factor. set to 0.5 if eval image is 1/2 the size (1/2 side length) of training resolution

    const uint16 d = img_depth.get({img_idx, coord.y, coord.x});
    if (d == 0) { return 0.f; }
    
    const float d_f = d * 1.f;
    const int2 u_coord = int2{
        coord.x + __float2int_rd(uv_scale * u.x / d_f),
        coord.y + __float2int_rd(uv_scale * u.y / d_f)
    };
    const int2 v_coord = int2{
        coord.x + __float2int_rd(uv_scale * v.x / d_f),
        coord.y + __float2int_rd(uv_scale * v.y / d_f)
    };

    const float u_d = img_depth.get({img_idx, u_coord.y, u_coord.x}) * 1.f;
    const float u_v = img_depth.get({img_idx, v_coord.y, v_coord.x}) * 1.f;

    return u_d - u_v;
}

#endif
