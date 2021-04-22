typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

// hard coded for now..
const int NUM_IMAGES = 16;
const int IMG_DIM_X = 424;
const int IMG_DIM_Y = 240;
const int TOTAL_NUM_PIXELS = NUM_IMAGES * IMG_DIM_X * IMG_DIM_Y;

const int NUM_CLASSES = 3;

const int NUM_RANDOM_FEATURES = 16;

const int MAX_TREE_DEPTH = 8;
const int MAX_LEAF_NODES = 256;//2**MAX_TREE_DEPTH;


__device__ int get_i(int img_idx, int img_x, int img_y) {
    return (img_idx * IMG_DIM_X * IMG_DIM_Y) + (img_y * IMG_DIM_X) + img_x;
}



__device__ uint16 get_pixel(uint16* img, int img_idx, int img_x, int img_y) {
    if (img_x < 0 || img_y < 0 || img_x >= IMG_DIM_X || img_y >= IMG_DIM_Y) {
        return 0;
    } else {
        return img[get_i(img_idx, img_x, img_y)];
    }
}

__device__ float compute_feature(uint16* img_depth, int img_idx, int2 coord, float2 u, float2 v) {
    const uint16 d = get_pixel(img_depth, img_idx, coord.x, coord.y);
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

    const float u_d = get_pixel(img_depth, img_idx, u_coord.x, u_coord.y) * 1.f;
    const float u_v = get_pixel(img_depth, img_idx, v_coord.x, v_coord.y) * 1.f;

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
        uint16* img_labels,
        uint16* img_depth,
        float* random_features,
        uint32* groups,
        uint32* next_groups,
        uint64* next_groups_counts) {


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // in bounds..
    if (i >= TOTAL_NUM_PIXELS) { return; }
    if (j >= NUM_RANDOM_FEATURES) { return; }

    const int img_idx = i / (IMG_DIM_X * IMG_DIM_Y);
    const int i_rem = i % (IMG_DIM_X * IMG_DIM_Y);
    const int img_y = i_rem / IMG_DIM_X;
    const int img_x = i_rem % IMG_DIM_X;

    const uint16 label = get_pixel(img_labels, img_idx, img_x, img_y);

    // Dont run training or eval on NONE TYPE.
    if (label == 0) { return; }

    // err... can we get them all at once?
    const float ux = random_features[(j * 5) + 0];
    const float uy = random_features[(j * 5) + 1];
    const float vx = random_features[(j * 5) + 2];
    const float vy = random_features[(j * 5) + 3];
    const float f_thresh = random_features[(j * 5) + 4];

    const float f_val = compute_feature(img_depth, img_idx, int2{img_x, img_y}, float2{ux, uy}, float2{vx, vy});

    
    const uint32 next_group = (groups[i] * 2) + (f_val < f_thresh ? 0 : 1);

    const int next_groups_idx = (j * TOTAL_NUM_PIXELS) + i;
    next_groups[next_groups_idx] = next_group;

    const int next_group_counts_idx = (j * MAX_LEAF_NODES * NUM_CLASSES) + (next_group * NUM_CLASSES) + label;
    atomicAdd(next_groups_counts + next_group_counts_idx, (uint64)1);

    

};
