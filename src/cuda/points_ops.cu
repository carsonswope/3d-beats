#include <cu_utils.hpp>

// given a depth image, 
extern "C" {__global__
void deproject_points(
        int4 imgs_dim, // (num_images, dimx, dimy)
        float2 pp, // (ppx, ppy)
        float f, // focal length
        uint16* _imgs,
        float4* _pts) {
        
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.z * blockDim.z + threadIdx.z;

    const int num_images = imgs_dim.x;
    const int2 img_dim = {imgs_dim.y, imgs_dim.z};

    if (i >= num_images || x >= img_dim.x || y >= img_dim.y) return;

    Array3d<uint16> imgs(_imgs, {num_images,img_dim.y,img_dim.x});
    Array3d<float4> pts(_pts, {num_images,img_dim.y,img_dim.x}, {0., 0., 0., 0.});
    
    const uint16 d = imgs.get({i, y, x});
    if (d > 0) {
        const float d_ = d * 1.f;
        const float4 p{
            d_ * (x - pp.x) / f,
            d_ * (y - pp.y) / f,
            d_,
            1.,
        };

        pts.set({i, y, x}, p);
    }
}}

// given a depth image, 
extern "C" {__global__
void depths_from_points(
        int4 imgs_dim, // (num_images, dimx, dimy)
        // float2 pp, // (ppx, ppy)
        // float f, // focal length
        uint16* _imgs,
        float4* _pts) {
        
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;
    const int y = blockIdx.z * blockDim.z + threadIdx.z;

    const int num_images = imgs_dim.x;
    const int2 img_dim = {imgs_dim.y, imgs_dim.z};

    if (i >= num_images || x >= img_dim.x || y >= img_dim.y) return;

    Array3d<uint16> imgs(_imgs, {num_images,img_dim.y,img_dim.x});
    Array3d<float4> pts(_pts, {num_images,img_dim.y,img_dim.x}, {0., 0., 0., 0.});
    
    const float4 pos = pts.get({i, y, x});
    if (pos.w > 0.f) {
        imgs.set({i, y, x}, (uint16)pos.z);
    }
}}

extern "C" {__global__
void transform_points(int num_pts, glm::vec4* pts, glm::mat4 t) {
    
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= num_pts) return;

    glm::vec4 p = pts[i];
    if (p.w != 1.) return;
    auto new_p = glm::transpose(t) * p;
    pts[i] = new_p;

}}

extern "C" {__global__
void setup_depth_image_for_forest(
        int NUM_PIXELS,
        glm::vec4* pts,
        uint16* depth) {

    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_PIXELS) return;

    const uint16 d = depth[i];
    const glm::vec4 p = pts[i];

    if (d == 0 || p.w == 0) {
        depth[i] = 65535;
    }

}}

extern "C" {__global__
void apply_point_mapping(
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_COLORS,
        uint8* _colors,
        uint8* _color_image) {
    
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM_X || y >= IMG_DIM_Y) return;

    Array2d<uint8> colors(_colors, {NUM_COLORS, 3});
    Array3d<uint8> color_image(_color_image, {IMG_DIM_Y, IMG_DIM_X, 3});

    auto* color_image_pixel = color_image.get_ptr({y, x, 0});
    if (color_image_pixel[0] + color_image_pixel[1] + color_image_pixel[2] == 0) return;

    float best_squared_diff = -1.f;
    uint8* best_colors_ptr = nullptr;

    for (int i = 0; i < NUM_COLORS; i++) {
        auto* test_color = colors.get_ptr({i, 0});
        float squared_diff = 0;
        for (int j = 0; j < 3; j++) {
            const float diff = (color_image_pixel[j] * 1.f) - test_color[j];
            squared_diff += diff * diff;
        }
        if (best_colors_ptr == nullptr || squared_diff < best_squared_diff) {
            best_squared_diff = squared_diff;
            best_colors_ptr = test_color;
        }
    }

    // or memcpy..
    for (int j =0; j < 3; j++) {
        color_image_pixel[j] = best_colors_ptr[j];
    }
}}

extern "C" {__global__
void split_pixels_by_nearest_color(
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_COLORS,
        uint8* _colors,
        uint8* _color_image,
        uint64* _pixel_counts_per_group) {
    
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM_X || y >= IMG_DIM_Y) return;

    Array2d<uint8> colors(_colors, {NUM_COLORS, 3});
    Array3d<uint8> color_image(_color_image, {IMG_DIM_Y, IMG_DIM_X, 3});
    Array2d<uint64> pixel_counts_per_group(_pixel_counts_per_group, {NUM_COLORS, 5}); // (num_pixels, sum_r, sum_g, sum_b, sum_cost)
    
    float best_squared_diff = -1.f;
    int best_colors_idx = -1;

    auto* color_image_pixel = color_image.get_ptr({y, x, 0});
    if (color_image_pixel[0] + color_image_pixel[1] + color_image_pixel[2] == 0) return;

    for (int i = 0; i < NUM_COLORS; i++) {
        auto* test_color = colors.get_ptr({i, 0});
        float squared_diff = 0;
        for (int j = 0; j < 3; j++) {
            const float diff = (color_image_pixel[j] * 1.f) - test_color[j];
            squared_diff += diff * diff;
        }
        if (best_colors_idx == -1 || squared_diff < best_squared_diff) {
            best_squared_diff = squared_diff;
            best_colors_idx = i;
        }
    }


    uint64* p = pixel_counts_per_group.get_ptr({best_colors_idx, 0});
    atomicAdd(p + 0, 1);
    atomicAdd(p + 1, uint64(color_image_pixel[0]));
    atomicAdd(p + 2, uint64(color_image_pixel[1]));
    atomicAdd(p + 3, uint64(color_image_pixel[2]));
    atomicAdd((double*)p+4, (double)best_squared_diff);

    // or memcpy..
    // for (int j =0; j < 3; j++) {
        // color_image_pixel[j] = best_colors_ptr[j];
    // }
}}


extern "C" {__global__
void make_rgba_from_labels(
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int NUM_COLORS,
        uint16* _labels,
        uint8* _colors,
        uint8* _color_image) {
    
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM_X || y >= IMG_DIM_Y) return;

    Array2d<uint16> labels(_labels, {IMG_DIM_Y, IMG_DIM_X});
    Array2d<uint8> colors(_colors, {NUM_COLORS, 4});
    Array3d<uint8> color_image(_color_image, {IMG_DIM_Y, IMG_DIM_X, 4});
    
    const auto l = labels.get({y, x});
    if (l == 0 || l == MAX_UINT16) return;

    auto* color_img_ptr = color_image.get_ptr({y, x, 0});
    auto* color_ptr = colors.get_ptr({l - 1, 0});
    memcpy(color_img_ptr, color_ptr, sizeof(uint8) * 4); // should evaluate to just 4 bytes..
}}
