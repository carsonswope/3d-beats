typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

// const int NUM_CLASSES = 3;

// const int MAX_TREE_DEPTH = 8;


__device__ int get_i(int img_idx, const int2 IMG_DIM, int img_x, int img_y) {
    return (img_idx * IMG_DIM.x * IMG_DIM.y) + (img_y * IMG_DIM.x) + img_x;
}

__device__ uint16 get_pixel(uint16* img, const int2 IMG_DIM, int img_idx, int img_x, int img_y) {
    if (img_x < 0 || img_y < 0 || img_x >= IMG_DIM.x || img_y >= IMG_DIM.y) {
        return 0;
    } else {
        return img[get_i(img_idx, IMG_DIM, img_x, img_y)];
    }
}

__device__ float compute_feature(uint16* img_depth, const int2 IMG_DIM, int img_idx, int2 coord, float2 u, float2 v) {
    const uint16 d = get_pixel(img_depth, IMG_DIM, img_idx, coord.x, coord.y);
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

    const float u_d = get_pixel(img_depth, IMG_DIM, img_idx, u_coord.x, u_coord.y) * 1.f;
    const float u_v = get_pixel(img_depth, IMG_DIM, img_idx, v_coord.x, v_coord.y) * 1.f;

    return u_d - u_v;
}

// img labels: (num_images, image_dim_y, image_dim_x)
// img depths: (num_images, image_dim_y, image_dim_x)
// random features: (num_random_features, 5) -> (ux,uy,vx,vy,thresh)
// groups: (num_images * image_dim_y * image_dim_x)
// next groups: (num_random_features, num_images * image_dim_y * image_dim_x)
// next_groups_counts: (num_random_features, 2**MAX_TREE_DEPTH, NUM_CLASSES)
// 
__global__ void evaluate_random_features(
        int NUM_IMAGES,
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_RANDOM_FEATURES,
        int NUM_CLASSES,
        int MAX_TREE_DEPTH,
        uint16* img_labels,
        uint16* img_depth,
        float* random_features,
        int* groups, // int32?? -1 means not active, 
        int* next_groups,
        uint64* next_groups_counts) {
    
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

    const uint16 label = get_pixel(img_labels, IMG_DIM, img_idx, img_x, img_y);

    // err... can we get them all at once?
    const float ux = random_features[(j * 5) + 0];
    const float uy = random_features[(j * 5) + 1];
    const float vx = random_features[(j * 5) + 2];
    const float vy = random_features[(j * 5) + 3];
    const float f_thresh = random_features[(j * 5) + 4];

    const float f_val = compute_feature(img_depth, IMG_DIM, img_idx, int2{img_x, img_y}, float2{ux, uy}, float2{vx, vy});

    const int next_group = (group * 2) + (f_val < f_thresh ? 0 : 1);

    const int next_groups_idx = (j * TOTAL_NUM_PIXELS) + i;
    next_groups[next_groups_idx] = next_group;

    const int next_group_counts_idx = (j * MAX_LEAF_NODES * NUM_CLASSES) + (next_group * NUM_CLASSES) + label;
    atomicAdd(next_groups_counts + next_group_counts_idx, (uint64)1);
};

// uses the trained tree to classify all pixels in N images
__global__ void evaluate_image_using_tree(
        int NUM_IMAGES,
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_CLASSES,
        int MAX_TREE_DEPTH,
        uint16* img_in,
        float* decision_tree, // (MAX_TREE_DEPTH * MAX_LEAF_NODES * 7) => (ux,uy,vx,vy,thresh,l_next,r_next)
        float* leaf_pdf, // (MAX_LEAF_NODES * NUM_CLASSES)
        uint16* labels_out)
{
    const int2 IMG_DIM{IMG_DIM_X, IMG_DIM_Y};
    const int MAX_LEAF_NODES = 1 << MAX_TREE_DEPTH; // 2^MAX_TREE_DEPTH (256)
    const int TOTAL_NUM_PIXELS = NUM_IMAGES * IMG_DIM.x * IMG_DIM.y;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_NUM_PIXELS) { return; }

    const int img_idx = i / (IMG_DIM.x * IMG_DIM.y);
    const int i_rem = i % (IMG_DIM.x * IMG_DIM.y);
    const int img_y = i_rem / IMG_DIM.x;
    const int img_x = i_rem % IMG_DIM.x;

    // Don't try to evaluate if img in has 0 value!
    const uint16 img_d = get_pixel(img_in, IMG_DIM, img_idx, img_x, img_y);
    if (img_d == 0) { return; }

    // current node ID
    int g = 0;

    // should be unrolled??
    for (int j = 0; j < MAX_TREE_DEPTH; j++) {
        const int decision_tree_base_idx = (j * MAX_LEAF_NODES * 7) + (g * 7);

        const float ux = decision_tree[decision_tree_base_idx + 0];
        const float uy = decision_tree[decision_tree_base_idx + 1];
        const float vx = decision_tree[decision_tree_base_idx + 2];
        const float vy = decision_tree[decision_tree_base_idx + 3];
        const float thresh = decision_tree[decision_tree_base_idx + 4];
        const int l_next = __float2int_rd(decision_tree[decision_tree_base_idx + 5]);
        const int r_next = __float2int_rd(decision_tree[decision_tree_base_idx + 6]);

        const float f = compute_feature(img_in, IMG_DIM, img_idx, int2{img_x, img_y}, float2{ux, uy}, float2{vx, vy});

        if (f < thresh) {
            // Left path
            if (l_next == -1) {
                g = (g * 2);
            } else {
                labels_out[i] = l_next;
                return;
            }
        } else {
            // Right path
            if (r_next == -1) {
                g = (g * 2) + 1;
            } else {
                labels_out[i] = r_next;
                return;
            }
        }
    }

    // got to end of tree.. pick element in PDF with highest!
    const int leaf_pdf_start_idx = (g * NUM_CLASSES);

    // labels_out[i] = 69;

    // labels_out[i] = (uint16)g;

    
    // float pct_sum = 0.;
    // for (int k = 0; k < NUM_CLASSES; k++) {
    //     float s = leaf_pdf[(g * NUM_CLASSES) + k];
    //     pct_sum += 0.3333f;
    // }

    // if (pct_sum > 1.01f || pct_sum < 0.99f) {
    //     labels_out[i] = 69;
    // } else {
    //     labels_out[i] = 55;
    // }
    // */

    // float s1 = leaf_pdf[(g * NUM_CLASSES) + 1];
    // float s2 = leaf_pdf[(g * NUM_CLASSES) + 2];
    // float s_sum = s1 + s2;

    // if (s_sum < 0.7 || s_sum > 1.3) {
    //     labels_out[i] = 69;
    // } else {
    //     labels_out[i] = 55;
    // }
    
    float best_pct = 0.f;
    uint16 best_class = 0;
    // float pct_sum = 0.;
    for (int j = 0; j < NUM_CLASSES; j++) {
        float pct = leaf_pdf[leaf_pdf_start_idx + j];
        // pct_sum += pct;
        if (pct > best_pct) {
            best_pct = pct;
            best_class = j;
        }
    }

    labels_out[i] = best_class;

    // if (pct_sum > 1.01f || pct_sum < 0.99f) {
    //     labels_out[i] = 69;
    // } else {
    //     labels_out[i] = 55;
    // }
    

    // labels_out[i] = best_class;

}
