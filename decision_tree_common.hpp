#ifndef __DECISION_TREE_COMMON
#define __DECISION_TREE_COMMON

#define MAX_UINT16 65535

#include <cu_utils.hpp>

__device__ float compute_feature(Array3d<uint16>& img_depth, int img_idx, int2 coord, float2 u, float2 v) {

    const uint16 d = img_depth.get({img_idx, coord.y, coord.x});
    if (d == 0) { return 0.f; }
    
    const float d_f = d * 1.f;
    const int2 u_coord = int2{
        coord.x + __float2int_rd(u.x / d_f),
        coord.y + __float2int_rd(u.y / d_f)
    };
    const int2 v_coord = int2{
        coord.x + __float2int_rd(v.x / d_f),
        coord.y + __float2int_rd(v.y / d_f)
    };

    const float u_d = img_depth.get({img_idx, u_coord.y, u_coord.x}) * 1.f;
    const float u_v = img_depth.get({img_idx, v_coord.y, v_coord.x}) * 1.f;

    return u_d - u_v;
}

#endif
