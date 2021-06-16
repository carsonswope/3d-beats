#include <cu_utils.hpp>

extern "C" {__global__
void make_composite_labels_image(
        uint16** label_images,
        const int num_label_images,
        const int dim_x,
        const int dim_y,
        int2* label_decision_tree,
        uint16* _composite_image) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim_x || y >= dim_y) return;

    int current_tree_offset = 0;

    Array2d<uint16> composite_image(_composite_image, {dim_y, dim_x});

    for (int i = 0; i < num_label_images; i++) {
        Array2d<uint16> img(label_images[i], {dim_y, dim_x});
        const auto l = img.get({y, x});
        if (l == 0 || l == MAX_UINT16) return;
        auto tree_val = label_decision_tree[current_tree_offset + l - 1];
        if (tree_val.x == 0) {
            composite_image.set({y, x}, (uint16)tree_val.y);
            return;
        } else {
            // tree_val == 1. keep looking
            current_tree_offset = tree_val.y;
        }
    }
    composite_image.set({y, x}, 10);

}}
