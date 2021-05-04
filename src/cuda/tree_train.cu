#include <cu_utils.hpp>
#include <decision_tree_common.hpp>

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
        int* nodes_by_pixel, // int32?? -1 means not active,
        uint64* _next_nodes_counts) {

    const int2 IMG_DIM{IMG_DIM_X, IMG_DIM_Y};
    const int MAX_LEAF_NODES = 1 << (MAX_TREE_DEPTH);
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

    const int node = nodes_by_pixel[i];
    if (node == -1) { return; }

    Array3d<uint16> img_labels(_img_labels, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X});
    Array3d<uint16> img_depth(_img_depth, {NUM_IMAGES,IMG_DIM_Y,IMG_DIM_X}, MAX_UINT16);
    Array2d<float> random_features(_random_features, {NUM_RANDOM_FEATURES, 5}); // (ux,uy,vx,vy,thresh)
    Array3d<uint64> next_nodes_counts(_next_nodes_counts, {NUM_RANDOM_FEATURES, MAX_LEAF_NODES, NUM_CLASSES});

    const uint16 label = img_labels.get({img_idx, img_y, img_x});

    // load feature
    const float* f = random_features.get_ptr({j, 0});
    const float2 u{f[0], f[1]};
    const float2 v{f[2], f[3]};
    const float f_thresh = f[4];
    // eval feature, split pixel into L or R node of next level
    const float f_val = compute_feature(img_depth, img_idx, {img_x, img_y}, u, v);
    const int next_node = (node * 2) + (f_val < f_thresh ? 0 : 1);

    assert(next_node < MAX_LEAF_NODES);
    uint64* next_nodes_counts_ptr = next_nodes_counts.get_ptr({j, next_node, label});
    atomicAdd(next_nodes_counts_ptr, (uint64)1);
}}

__device__ uint64 node_counts_sum(uint64* p, const int num_classes) {
    uint64 s = 0;
    for (int i = 0; i < num_classes; i++) s += p[i];
    return s;
}

__device__ float gini_impurity(uint64* c, const int num_classes) {
    const float s = node_counts_sum(c, num_classes) * 1.f;
    float p = 0.f;
    for (int i =0; i < num_classes; i++) {
        const float p_i = c[i] / s;
        p += p_i * p_i;
    }
    return 1 - p;
}

__device__ float gini_gain(uint64* p_counts, uint64* l_counts, uint64* r_counts, const int num_classes) {
    const float p_sum = node_counts_sum(p_counts, num_classes);
    float p_impurity = gini_impurity(p_counts, num_classes);
    float remainder =
        ((node_counts_sum(l_counts, num_classes) / p_sum) * gini_impurity(l_counts, num_classes)) +
        ((node_counts_sum(r_counts, num_classes) / p_sum) * gini_impurity(r_counts, num_classes));
    return p_impurity - remainder;
}

// if any one group has N pct of the sum, return group ID. else return -1
__device__ int count_above_cutoff(uint64* counts, const int num_classes, const uint64 sum, const float cutoff) {
    for (int i =0; i < num_classes; i++) {
        if (counts[i] * 1.f / sum >= cutoff) return i;
    }
    return -1;
}

extern "C" {__global__
void pick_best_features(
        int NUM_ACTIVE_NODES,
        int NUM_RANDOM_FEATURES,
        int MAX_TREE_DEPTH,
        int NUM_CLASSES,
        int CURRENT_TREE_LEVEL,
        int* active_nodes,
        uint64* _parent_node_counts,
        uint64* _child_node_counts_by_feature, // per random feature!
        float* _random_features,
        float* _tree_out,
        uint64* _child_node_counts, // not by feature! - after we picked the best feature!
        int* next_active_nodes,
        int* num_next_active_nodes
    ) {

    // i: active node idx
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUM_ACTIVE_NODES) return;

    const int TREE_NODE_ELS = 7 + (NUM_CLASSES * 2);
    const int MAX_LEAF_NODES = 1 << (MAX_TREE_DEPTH);
    Array2d<uint64> parent_node_counts(_parent_node_counts, {MAX_LEAF_NODES, NUM_CLASSES});
    Array3d<uint64> child_node_counts_by_feature(_child_node_counts_by_feature, {NUM_RANDOM_FEATURES, MAX_LEAF_NODES, NUM_CLASSES});
    Array2d<float> random_features(_random_features, {NUM_RANDOM_FEATURES, 5}); // (ux,uy,vx,vy,thresh)
    BinaryTree<float> tree_out(_tree_out, TREE_NODE_ELS, MAX_TREE_DEPTH);
    Array2d<uint64> child_node_counts(_child_node_counts, {MAX_LEAF_NODES, NUM_CLASSES});

    const int parent_node = active_nodes[i];

    const int left_child_node = (parent_node * 2);
    const int right_child_node = (parent_node * 2) + 1;

    uint64* parent_node_counts_ptr = parent_node_counts.get_ptr({parent_node, 0});
    const uint64 parent_nodes_sum = node_counts_sum(parent_node_counts_ptr, NUM_CLASSES);

    float best_g = -1.;
    int best_g_feature_id = 0;
    uint64* best_left_counts_ptr = nullptr;
    uint64* best_right_counts_ptr = nullptr;

    for (int j = 0; j < NUM_RANDOM_FEATURES; j++) {

        uint64* left_child_counts_ptr = child_node_counts_by_feature.get_ptr({j, left_child_node, 0});
        uint64* right_child_counts_ptr = child_node_counts_by_feature.get_ptr({j, right_child_node, 0});
    
        // debug!
        // verify sums match
        const uint64 left_nodes_sum = node_counts_sum(left_child_counts_ptr, NUM_CLASSES);
        const uint64 right_nodes_sum = node_counts_sum(right_child_counts_ptr, NUM_CLASSES);
        assert(left_nodes_sum + right_nodes_sum == parent_nodes_sum);
        //
        float g = (!left_nodes_sum || !right_nodes_sum) ?
            0. :
            gini_gain(parent_node_counts_ptr, left_child_counts_ptr, right_child_counts_ptr, NUM_CLASSES);

        if (g > best_g) {
            best_g = g;
            best_g_feature_id = j;
            best_left_counts_ptr = left_child_counts_ptr;
            best_right_counts_ptr = right_child_counts_ptr;
        }
    }

    assert(best_g > -1.f);

    const uint64 best_left_counts_sum = node_counts_sum(best_left_counts_ptr, NUM_CLASSES);
    const uint64 best_right_counts_sum  = node_counts_sum(best_right_counts_ptr, NUM_CLASSES);

    // debug again..
    assert(best_left_counts_sum + best_right_counts_sum == parent_nodes_sum);

    // copy selected proposal!
    float* tree_out_ptr = tree_out.get_ptr(CURRENT_TREE_LEVEL, parent_node);
    float* proposal_ptr = random_features.get_ptr({best_g_feature_id, 0});
    memcpy(tree_out_ptr, proposal_ptr, sizeof(float) * 5);

    // no proposal provided any gain..
    // both L and R are end nodes, with learned PDF from parent for both.
    if (best_g <= 0.) {
        tree_out_ptr[5] = 0.;
        tree_out_ptr[6] = 0.;
        for (int k= 0; k < NUM_CLASSES; k++) {
            const float p = (parent_node_counts_ptr[k] * 1.f) / parent_nodes_sum;
            tree_out_ptr[7+k] = p;
            tree_out_ptr[7+NUM_CLASSES+k] = p;
        }
        return;
    }

    const float CUTOFF_THRESH = 0.999f;

    const int l_cutoff = count_above_cutoff(best_left_counts_ptr, NUM_CLASSES, best_left_counts_sum, CUTOFF_THRESH);
    if (l_cutoff > -1) {
        tree_out_ptr[5] = 0.;
        tree_out_ptr[7 + l_cutoff] = 1.;
    } else {
        if (CURRENT_TREE_LEVEL == MAX_TREE_DEPTH - 1) {
            tree_out_ptr[5] = 0.;
            for (int n = 0; n < NUM_CLASSES; n++) {
                tree_out_ptr[7+n] = (best_left_counts_ptr[n] * 1.f) / best_left_counts_sum;
            }
        } else {
            tree_out_ptr[5] = -1;
            // still going! add to next!
            next_active_nodes[atomicAdd(num_next_active_nodes, 1)] = left_child_node;
            memcpy(child_node_counts.get_ptr({left_child_node, 0}), best_left_counts_ptr, sizeof(uint64) * NUM_CLASSES);
        }
    }

    const int r_cutoff = count_above_cutoff(best_right_counts_ptr, NUM_CLASSES, best_right_counts_sum, CUTOFF_THRESH);
    if (r_cutoff > -1) {
        tree_out_ptr[6] = 0.;
        tree_out_ptr[7 + NUM_CLASSES + r_cutoff] = 1.;
    } else {
        if (CURRENT_TREE_LEVEL == MAX_TREE_DEPTH - 1) {
            tree_out_ptr[6] = 0.;
            for (int n = 0; n < NUM_CLASSES; n++) {
                tree_out_ptr[7 + NUM_CLASSES + n] = (best_right_counts_ptr[n] * 1.f) / best_right_counts_sum;
            }
        } else {
            tree_out_ptr[6] = -1;
            // still going! add to next!
            next_active_nodes[atomicAdd(num_next_active_nodes, 1)] = right_child_node;
            memcpy(child_node_counts.get_ptr({right_child_node, 0}), best_right_counts_ptr, sizeof(uint64) * NUM_CLASSES);
        }
    }
}}


extern "C" {__global__
void copy_pixel_groups(
        int NUM_IMAGES,
        int IMG_DIM_X,
        int IMG_DIM_Y,
        int CURRENT_TREE_LEVEL,
        int MAX_TREE_DEPTH,
        int NUM_CLASSES,
        uint16* _img_depth,
        int* nodes_by_pixel, // -1 means not active
        float* _tree_so_far) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int2 IMG_DIM{IMG_DIM_X, IMG_DIM_Y};
    const int TOTAL_NUM_PIXELS = NUM_IMAGES * IMG_DIM.x * IMG_DIM.y;
    if (i >= TOTAL_NUM_PIXELS) return;

    const int i_parent_node = nodes_by_pixel[i];
    // pixel inactive: no need to copy anything
    if (i_parent_node == -1) return;

    const int img_idx = i / (IMG_DIM.x * IMG_DIM.y);
    const int i_rem = i % (IMG_DIM.x * IMG_DIM.y);
    const int img_y = i_rem / IMG_DIM.x;
    const int img_x = i_rem % IMG_DIM.x;

    Array3d<uint16> img_depth(_img_depth, {NUM_IMAGES, IMG_DIM_Y, IMG_DIM_X}, MAX_UINT16);

    // each tree node consists of this many elements..
    const int TREE_NODE_ELS = 7 + (NUM_CLASSES * 2);
    BinaryTree<float> tree_so_far(_tree_so_far, TREE_NODE_ELS, MAX_TREE_DEPTH);
    // if still active, evaluate at tree!
    float* tree_node_ptr = tree_so_far.get_ptr(CURRENT_TREE_LEVEL, i_parent_node);

    const float2 u{tree_node_ptr[0], tree_node_ptr[1]};
    const float2 v{tree_node_ptr[2], tree_node_ptr[3]};
    const float f_thresh = tree_node_ptr[4];

    const float f_val = compute_feature(img_depth, img_idx, {img_x, img_y}, u, v);

    const bool is_left_node = f_val < f_thresh;
    const int node_status = __float2int_rd(tree_node_ptr[is_left_node ? 5 : 6]);
    // If child node is labeled -1, that means the child node is active!
    if (node_status != -1) {
        nodes_by_pixel[i] = -1;
    } else {
        const int next_node = (i_parent_node * 2) + (is_left_node ? 0 : 1);
        nodes_by_pixel[i] = next_node;
    }
}}