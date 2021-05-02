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

/*

    np.int32(DIM_X * DIM_Y),
    np.float32(PLANE_Z_FILTER_THRESHOLD),
    pts_cu,
    plane

    */

extern "C" {__global__
void filter_points_by_plane(
        int NUM_PTS,
        float PLANE_Z_FILTER_THRESHOLD,
        glm::vec4* pts){

    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= NUM_PTS) return;

    const auto pt = pts[i];
    if (pt.w != 1.) return;

    // const auto new_pt = glm::transpose(plane_tform) * pt;

    if (pt.z > -PLANE_Z_FILTER_THRESHOLD) {
        pts[i].x = 0.;
        pts[i].y = 0.;
        pts[i].z = 0.;
        pts[i].w = 0.;
    }
    
}}