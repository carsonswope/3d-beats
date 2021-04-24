#include <cu_utils.hpp>

typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

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

// img labels: (num_images, image_dim_y, image_dim_x)
// img depths: (num_images, image_dim_y, image_dim_x)
// random features: (num_random_features, 5) -> (ux,uy,vx,vy,thresh)
// groups: (num_images * image_dim_y * image_dim_x)
// next groups: (num_random_features, num_images * image_dim_y * image_dim_x)
// next_groups_counts: (num_random_features, 2**MAX_TREE_DEPTH, NUM_CLASSES)
// 
extern "C" {__global__
void evaluate_random_features(
        int NUM_IMAGES,
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_RANDOM_FEATURES,
        int NUM_CLASSES,
        int MAX_TREE_DEPTH,
        uint16* _img_labels,
        uint16* _img_depth,
        float* _random_features,
        int* groups, // int32?? -1 means not active, 
        int* _next_groups,
        uint64* _next_groups_counts) {

    const int2 IMG_DIM{IMG_DIM_X, IMG_DIM_Y};
    const int MAX_LEAF_NODES = 1 << MAX_TREE_DEPTH; // 2^MAX_TREE_DEPTH (256)
    const int TOTAL_NUM_PIXELS = NUM_IMAGES * IMG_DIM.x * IMG_DIM.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // in bounds..
    if (i >= TOTAL_NUM_PIXELS) { return; }
    if (j >= NUM_RANDOM_FEATURES) { return; }

    const int img_idx = i / (IMG_DIM.x * IMG_DIM.y);
    const int i_rem = i % (IMG_DIM.x * IMG_DIM.y);
    const int img_y = i_rem / IMG_DIM.x;
    const int img_x = i_rem % IMG_DIM.x;

    const int group = groups[i];
    if (group == -1) { return; }

    Array3d<uint16> img_labels(_img_labels, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});
    Array3d<uint16> img_depth(_img_depth, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});
    Array2d<float> random_features(_random_features, {NUM_RANDOM_FEATURES, 5}); // (ux,uy,vx,vy,thresh)
    Array2d<int> next_groups(_next_groups, {NUM_RANDOM_FEATURES, TOTAL_NUM_PIXELS});
    Array3d<uint64> next_groups_counts(_next_groups_counts, {NUM_RANDOM_FEATURES, MAX_LEAF_NODES, NUM_CLASSES});

    const uint16 label = img_labels.get({img_idx, img_y, img_x});

    // load feature
    const float* f = random_features.get_ptr({j, 0});
    const float2 u{f[0], f[1]};
    const float2 v{f[2], f[3]};
    const float f_thresh = f[4];
    // eval feature, split pixel into L or R group of next level
    const float f_val = compute_feature(img_depth, img_idx, {img_x, img_y}, u, v);
    const int next_group = (group * 2) + (f_val < f_thresh ? 0 : 1);
    next_groups.set({j, i}, next_group);

    uint64* next_groups_counts_ptr = next_groups_counts.get_ptr({j, next_group, label});
    atomicAdd(next_groups_counts_ptr, (uint64)1);
};}


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
extern "C" {
__global__ void evaluate_image_using_tree(
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
    const int MAX_LEAF_NODES = 1 << MAX_TREE_DEPTH; // 2^MAX_TREE_DEPTH
    const int TOTAL_NUM_PIXELS = NUM_IMAGES * IMG_DIM.x * IMG_DIM.y;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_NUM_PIXELS) { return; }

    const int img_idx = i / (IMG_DIM.x * IMG_DIM.y);
    const int i_rem = i % (IMG_DIM.x * IMG_DIM.y);
    const int img_y = i_rem / IMG_DIM.x;
    const int img_x = i_rem % IMG_DIM.x;

    Array3d<uint16> img_in(_img_in, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});
    Array3d<uint16> labels_out(_labels_out, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});
    Array3d<float> decision_tree_w(decision_tree, {MAX_TREE_DEPTH,MAX_LEAF_NODES,(7 + NUM_CLASSES + NUM_CLASSES)}); // (ux,uy,vx,vy,thresh,l_next,r_next,{l_pdf},{r_pdf})

    // Don't try to evaluate if img in has 0 value!
    const uint16 img_d = img_in.get({img_idx, img_y, img_x});
    if (img_d == 0) { return; }

    // current node ID
    int g = 0;

    const int TREE_NODE_ELS = 7 + NUM_CLASSES + NUM_CLASSES;

    // should be unrolled??
    for (int j = 0; j < MAX_TREE_DEPTH; j++) {
        float* d_ptr = decision_tree_w.get_ptr({j, g, 0});
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

}
}