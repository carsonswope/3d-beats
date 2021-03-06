#include <cu_utils.hpp>

extern "C" {__global__
void run(
        uint16* _label_image,
        const int num_classes,
        const int dim_x,
        const int dim_y,
        float* variances,
        // current means
        double2* means,
        const int iter_number,
        // numerator and denominator fields computed by atomicSums
        double* _temp_sum) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim_x || y >= dim_y) return;

    Array2d<uint16> label_image(_label_image, {dim_y, dim_x});
    const auto l = label_image.get({y, x});
    if (l == 0 || l == MAX_UINT16) return;

    Array2d<double> temp_sum(_temp_sum, {num_classes, 3});
    auto* const temp_sum_ptr = temp_sum.get_ptr({l-1, 0});

    double2 c{x * 1., y * 1.};


    if (iter_number == 0) {
        atomicAdd(temp_sum_ptr + 0, c.x);
        atomicAdd(temp_sum_ptr + 1, c.y);
        atomicAdd(temp_sum_ptr + 2, 1.);
    } else {
        const double2 diff = {
            c.x - means[l-1].x,
            c.y - means[l-1].y,
        };
        const double dist_sq = (diff.x*diff.x) + (diff.y*diff.y);
        const double v_2 = variances[l - 1] * variances[l - 1];
        const double p = exp(-dist_sq / (2 * v_2));

        atomicAdd(temp_sum_ptr + 0, diff.x * p);
        atomicAdd(temp_sum_ptr + 1, diff.y * p);
        atomicAdd(temp_sum_ptr + 2, p);
    }
}}
    