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
void make_triangles(const int DIM_X, const int DIM_Y, uint64* triangle_count, float4* _pts, uint32* idxes) {
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= (DIM_X - 1) || y >= (DIM_Y - 1)) return;

    Array2d<float4> pts(_pts, {DIM_Y, DIM_X}, float4{0., 0., 0., 0.});
    
    float4 p[4] = {
        pts.get({y,   x  }),
        pts.get({y,   x+1}),
        pts.get({y+1, x  }),
        pts.get({y+1, x+1})
    };

    // atomicAdd(triangle_count, 2);

    
    if (p[0].w > 0. && p[1].w > 0. && p[2].w > 0. && p[3].w > 0. ) {
        int p_idx[4] = {
            pts.get_idx({y,   x  }),
            pts.get_idx({y,   x+1}),
            pts.get_idx({y+1, x  }),
            pts.get_idx({y+1, x+1})
        };

        const auto tri_idx = atomicAdd(triangle_count, 2);
        const auto v_idx = tri_idx * 3;
        idxes[v_idx + 0] = p_idx[0];
        idxes[v_idx + 1] = p_idx[1];
        idxes[v_idx + 2] = p_idx[2];

        idxes[v_idx + 3] = p_idx[1];
        idxes[v_idx + 4] = p_idx[2];
        idxes[v_idx + 5] = p_idx[3];
    }
    
}}

extern "C" {__global__
void convert_0s_to_maxuint(
        int NUM_PIXELS,
        uint16* depth) {

    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_PIXELS) return;
    if (depth[i] == 0) {
        depth[i] = MAX_UINT16;
    }
}}


extern "C" {__global__
void remove_missing_3d_points_from_depth_image(
        int NUM_PIXELS,
        glm::vec4* pts,
        uint16* depth) {

    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_PIXELS) return;

    // const uint16 d = depth[i];
    const glm::vec4 p = pts[i];

    if (p.w == 0.) {
        depth[i] = 0;
    }

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

extern "C" {__global__
void make_depth_rgba(
        int2 IMG_DIM,
        uint16 d_min,
        uint16 d_max,
        uint16* _d,
        uint8* _c) {
    
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM.x || y >= IMG_DIM.y) return;

    const auto d = Array2d<uint16>(_d, {IMG_DIM.y, IMG_DIM.x}).get({y, x});

    Array3d<uint8> c(_c, {IMG_DIM.y, IMG_DIM.x, 4});

    uint8 new_color[4] = {0, 0, 0, 255};

    if (d == 0) {
        new_color[0] = 195;
        new_color[1] = 157;
        new_color[2] = 152;
    } else if (d == MAX_UINT16) {
        new_color[0] = 157;
        new_color[1] = 195;
        new_color[2] = 152; 
    } else if (d < d_min || d > d_max) {
        new_color[0] = 157;
        new_color[1] = 152;
        new_color[2] = 195;
    } else {
        float n_f = ((1.0f * d - d_min) * 255.f) / (d_max - d_min);
        auto n_uint = (uint8)__float2uint_rd(256.f - n_f);
        new_color[0] = n_uint;
        new_color[1] = n_uint;
        new_color[2] = n_uint;
    }

    auto* c_ptr = c.get_ptr({y, x, 0});
    memcpy(c_ptr, new_color, sizeof(uint8)*4);
}}


extern "C" {__global__
void gaussian_depth_filter(
        int2 IMG_DIM,
        int window_size,
        float* _k,
        uint16* _d_in,
        uint16* _d_out) {
    
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM.x || y >= IMG_DIM.y) return;

    // a depth value of d suggests 
    auto k = Array2d<float>(_k, {window_size, window_size});
    auto d_in = Array2d<uint16>(_d_in, {IMG_DIM.y, IMG_DIM.x});
    auto d_out = Array2d<uint16>(_d_out, {IMG_DIM.y, IMG_DIM.x});

    float w_0 = 0.f;
    float w_non0 = 0.f;
    float sum_non0 = 0.f;

    for (int dy = 0; dy < window_size; dy++) {
        for (int dx = 0; dx < window_size; dx++) {
            const int2 c = {
                x + dx - (window_size/2),
                y + dy - (window_size/2)
            };
            if (c.y < 0 || c.x < 0 || c.y >= IMG_DIM.y || c.x >= IMG_DIM.x) continue;

            const auto d = d_in.get({c.y, c.x});
            const auto w = k.get({dy, dx});

            if (d == 0) {
                w_0 += w;
            } else {
                w_non0 += w;
                sum_non0 += (d*w);
            }
        }
    }

    uint16 v = w_0 > w_non0
        ? 0
        : (uint16)__float2uint_rd(sum_non0 / w_non0);

    d_out.set({y, x}, v);

}}

extern "C" {__global__
void shrink_image(
        int2 IMG_DIM_IN,
        int mipmap_level, // 1, 2, 3, 4
        uint16* _d_in,
        uint16* _d_out) {

    const int f = 1 << mipmap_level; // reduction factor
    const int2 IMG_DIM_OUT = {
        IMG_DIM_IN.x / f,
        IMG_DIM_IN.y / f,
    };

    const int x_out = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y_out = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x_out >= IMG_DIM_OUT.x || y_out >= IMG_DIM_OUT.y) return;

    auto d_in = Array2d<uint16>(_d_in, {IMG_DIM_IN.y, IMG_DIM_IN.x});
    auto d_out = Array2d<uint16>(_d_out, {IMG_DIM_OUT.y, IMG_DIM_OUT.x});

    const int x_in = x_out * f;
    const int y_in = y_out * f;

    if (x_in >= IMG_DIM_IN.x || y_in >= IMG_DIM_IN.y) {
        d_out.set({y_out, x_out}, 0);
    } else {
        d_out.set({y_out, x_out}, d_in.get({y_in, x_in}));
    }
}}


extern "C" {__global__
void grow_groups(
        int2 IMG_DIM,
        uint16* _g_in,
        uint16* _g_out) {

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM.x || y >= IMG_DIM.y) return;

    const int2 DIRS[4] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    auto g_in = Array2d<uint16>(_g_in, {IMG_DIM.y, IMG_DIM.x});
    auto g_out = Array2d<uint16>(_g_out, {IMG_DIM.y, IMG_DIM.x});

    const auto g = g_in.get({y, x});
    if (g != 0) {
        g_out.set({y, x}, g);
        return;
    };

    for (int i = 0; i < 4; i++) {
        const auto dir = DIRS[i];
        const auto _g = g_in.get({y + dir.y, x + dir.x});
        if (_g != 0) {
            g_out.set({y, x}, _g);
            return;
        }
    }
}}


extern "C" {__global__
void stencil_depth_image_by_group(
        int2 IMG_DIM,
        int mipmap_level,
        int group,
        uint16* _g_in,
        uint16* _d_in,
        uint16* _d_out) {

    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x >= IMG_DIM.x || y >= IMG_DIM.y) return;

    const int f = 1 << mipmap_level; // reduction factor

    auto g_in = Array2d<uint16>(_g_in, {IMG_DIM.y / f, IMG_DIM.x / f});
    auto d_in = Array2d<uint16>(_d_in, {IMG_DIM.y, IMG_DIM.x});
    auto d_out = Array2d<uint16>(_d_out, {IMG_DIM.y, IMG_DIM.x});

    const auto g = g_in.get({y / f, x / f});

    if (g != group) return;

    d_out.set({y, x}, d_in.get({y, x}));

}}
    