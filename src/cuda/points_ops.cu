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
void find_plane_ransac(
        int NUM_RANDOM_GUESSES,
        float PLANE_Z_OUTLIER_THRESHOLD,
        int NUM_PTS,
        glm::vec4* pts,
        glm::mat4* candidate_planes,
        int* num_inliers) {
    
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_PTS) return;

    const auto pt = pts[i];
    if (pt.w != 1.) return;

    for (int j = 0; j < NUM_RANDOM_GUESSES; j++) {

        const auto t = candidate_planes[j];
        const auto new_pt = glm::transpose(t) * pt;
        if (new_pt.z < PLANE_Z_OUTLIER_THRESHOLD && new_pt.z > -PLANE_Z_OUTLIER_THRESHOLD) {
            atomicAdd(num_inliers + j, 1);
        }
    }
}}


extern "C" {__global__
void filter_points_by_plane(
        int NUM_PTS,
        float PLANE_Z_FILTER_THRESHOLD,
        glm::vec4* pts){

    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_PTS) return;

    const auto pt = pts[i];
    if (pt.w != 1.) return;

    if (pt.z > -PLANE_Z_FILTER_THRESHOLD) {
        pts[i].x = 0.;
        pts[i].y = 0.;
        pts[i].z = 0.;
        pts[i].w = 0.;
    }
    
}}


extern "C" {__global__
void make_plane_candidates(
        int NUM_CANDIDATES,
        int IMG_DIM_X,
        int IMG_DIM_Y,
        float* _rand,
        glm::vec4* pts,
        glm::mat4* plane_candidates){

    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_CANDIDATES) return;

    Array2d<float> rand_arr(_rand, {NUM_CANDIDATES, 32});
    
    glm::vec4 plane_pts[3];
    int plane_pts_set = 0;
    int rand_j = 0;

    while (plane_pts_set < 3 && rand_j < 32) {
        int r = __float2int_rd(rand_arr.get({i, rand_j}) * IMG_DIM_X * IMG_DIM_Y);
        glm::vec4 p = pts[r];
        if (p.z > 0.) {
            plane_pts[plane_pts_set++] = p;
        }
        rand_j++;
    }

    glm::vec3 v0 = glm::normalize((plane_pts[1] - plane_pts[0]).xyz());
    glm::vec3 v1 = glm::normalize((plane_pts[2] - plane_pts[0]).xyz());

    glm::vec3 z_axis = glm::normalize(glm::cross(v0, v1));
    glm::vec3 x_axis = v0;
    glm::vec3 y_axis = glm::normalize(glm::cross(z_axis, x_axis));

    glm::mat4 tf_mat = glm::mat4(1.f);
    tf_mat[0] = glm::vec4{x_axis, 0.f};
    tf_mat[1] = glm::vec4{y_axis, 0.f};
    tf_mat[2] = glm::vec4{z_axis, 0.f};
    tf_mat[3] = glm::vec4{-plane_pts[0].xyz(), 1.f};

    plane_candidates[i] = glm::transpose(tf_mat);

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

// setup_depth_image_for_forest