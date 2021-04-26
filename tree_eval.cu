// code for using a given tree to evaluate images!
#include <cu_utils.hpp>
#include <decision_tree_common.hpp>

__device__ int get_best_pdf_chance(float* pdf, int num_classes) {
        
    float best_pct = 0.f;
    int best_class = 0;

    for (int j = 0; j < num_classes; j++) {
        float pct = pdf[j];
        if (pct > best_pct) {
            best_pct = pct;
            best_class = j;
        }
    }

    return best_class;
}

// uses the trained tree to classify all pixels in N images
extern "C" {__global__
    void evaluate_image_using_tree(
        int NUM_IMAGES,
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_CLASSES,
        int MAX_TREE_DEPTH,
        uint16* _img_in,
        float* decision_tree,
        uint16* _labels_out)
{

    const int2 IMG_DIM{IMG_DIM_X, IMG_DIM_Y};
    const int TOTAL_NUM_PIXELS = NUM_IMAGES * IMG_DIM.x * IMG_DIM.y;
    const int TREE_NODE_ELS = 7 + NUM_CLASSES + NUM_CLASSES; // (ux,uy,vx,vy,thresh,l_next,r_next,{l_pdf},{r_pdf})

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_NUM_PIXELS) { return; }

    const int img_idx = i / (IMG_DIM.x * IMG_DIM.y);
    const int i_rem = i % (IMG_DIM.x * IMG_DIM.y);
    const int img_y = i_rem / IMG_DIM.x;
    const int img_x = i_rem % IMG_DIM.x;

    Array3d<uint16> img_in(_img_in, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});
    Array3d<uint16> labels_out(_labels_out, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});

    BinaryTree<float> decision_tree_w(decision_tree, TREE_NODE_ELS, MAX_TREE_DEPTH);

    // Don't try to evaluate if img in has 0 value!
    const uint16 img_d = img_in.get({img_idx, img_y, img_x});
    if (img_d == 0 || img_d == 65535) { return; } // max uint16 is also considered 'not present'

    // current node ID
    int g = 0;

    // should be unrolled??
    for (int j = 0; j < MAX_TREE_DEPTH; j++) {
        float* d_ptr = decision_tree_w.get_ptr(j, g);
        const float2 u = {d_ptr[0], d_ptr[1]};
        const float2 v = {d_ptr[2], d_ptr[3]};
        const float thresh = d_ptr[4];

        const int l_next = __float2int_rd(d_ptr[5]);
        const int r_next = __float2int_rd(d_ptr[6]);

        float* l_pdf = d_ptr + 7;
        float* r_pdf = d_ptr + 7 + NUM_CLASSES;

        const float f = compute_feature(img_in, img_idx, int2{img_x, img_y}, u, v);

        if (f < thresh) {
            // Left path
            if (l_next == -1) {
                g = (g * 2);
            } else {
                int label = get_best_pdf_chance(l_pdf, NUM_CLASSES);
                labels_out.set({img_idx, img_y, img_x}, (uint16)label);
                return;
            }
        } else {
            // Right path
            if (r_next == -1) {
                g = (g * 2) + 1;
            } else {
                int label = get_best_pdf_chance(r_pdf, NUM_CLASSES);
                labels_out.set({img_idx, img_y, img_x}, (uint16)label);
                return;
            }
        }
    }

}}
