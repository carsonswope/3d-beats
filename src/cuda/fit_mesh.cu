#include <cu_utils.hpp>

extern "C" {__global__
void calc_image_cost(
        int dim_x,
        int dim_y,
        uint16* _d0,
        uint16* _d1,
        uint16* _labels, // for d0 evaluated
        uint16 target_label,
        float* cost) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim_x || y >= dim_y) return;

    Array2d<uint16> d0(_d0, {dim_y, dim_x});
    Array2d<uint16> d1(_d1, {dim_y, dim_x});
    Array2d<uint16> labels(_labels, {dim_y, dim_x});

    const auto d0_val = d0.get({y, x});
    const auto d1_val = d1.get({y, x});
    const auto label_val = labels.get({y, x});

    // Various cost conditions here!

    // First case: original image did not have a value for this pixel.
    // Rendered image is allowed to have a pixel here at no cost
    if (d0_val == 0) {
        return;
    }

    const float BOUNDARY_MISMATCH_COST = 1000.;

    if (label_val == target_label && d1_val == 0) {
        atomicAdd(cost, BOUNDARY_MISMATCH_COST);
        return;
    }

    if (label_val != target_label && d1_val != 0) {
        if (d1_val > d0_val) {
            atomicAdd(cost, BOUNDARY_MISMATCH_COST);
        }
        return;
    }

    if (label_val == target_label && d1_val != 0) {
        const float diff = abs((d0_val * 1.f) - (d1_val * 1.f));
        atomicAdd(cost, 0.01 * diff*diff);        
        return;
    }

    // if (label_val == target_label) {
    //     if (d1_val == 0) {
    //         // expected there to be a pixel here, but rendered image did not have one!
    //         atomicAdd(cost, BOUNDARY_MISMATCH_COST);
    //     } else {
    //         // pixel here is expected. compare depth values!
    //         const float diff = abs((d0_val * 1.f) - (d1_val * 1.f));
    //         atomicAdd(cost, diff);
    //     }
    // } else {
    //     // label does not match
    //     if (d1_val == 0) {
    //         // good, label doesn't match so we don't expect rendered image to have written here
    //     } else {
    //         // bad - we wrote a pixel here that was not expected.
    //         if (d1_val > d0_val) {
    //             // if rendered pixel has depth greater than original depth,
    //             // then no cost, because it would have failed the depth test
    //         } else {
    //             atomicAdd(cost, BOUNDARY_MISMATCH_COST);
    //         }
    //     }
    // }

}}
