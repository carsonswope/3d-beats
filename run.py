import numpy as np
from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array
from pycuda.compiler import SourceModule

import os

np.set_printoptions(suppress=True)

# load/compile cuda kernels..
cu_file = open('tree_train.cu', 'r')
cu_text = cu_file.read()
try:
    cu_mod = SourceModule(cu_text, no_extern_c=True, include_dirs=[os.getcwd()])
    cu_eval_random_features = cu_mod.get_function('evaluate_random_features')
    cu_eval_image = cu_mod.get_function('evaluate_image_using_tree')
    cu_pick_best_features = cu_mod.get_function('pick_best_features')
    cu_copy_pixel_groups = cu_mod.get_function('copy_pixel_groups')
except Exception as e:
    print('error:')
    print(e.msg)
    print(e.stdout)
    print(e.stderr)
    exit()

DEPTH = 'depth'
LABELS = 'labels'
TEST = 'datagen/generated/train_3class'
TRAIN = 'datagen/generated/train_3class'

IMG_DIMS = np.array([424, 240], dtype=np.int)

COLOR_NONE = np.array([0, 0, 0, 0], dtype='uint16')
COLOR_TABLE = np.array([255, 0, 0, 255], dtype='uint16')
COLOR_HAND = np.array([0, 0, 255, 255], dtype='uint16')
COLOR_FINGER = np.array([0, 255, 0, 255], dtype='uint16')

ID_NONE = 0
ID_TABLE = 1
ID_HAND = 2
ID_FINGER = 3
NUM_CLASSES = 4

# DATA_ROOT = 'datagen/generated2'
# s: TEST or TRAIN (set)
# t: DEPTH or LABELS (type)
def data_path(s, t, i):
    return s + '_' + str(i).zfill(8) + '_' + t + '.png'

def load_data(s, num):
    all_depth = np.zeros((num, IMG_DIMS[1], IMG_DIMS[0]), dtype='uint16')
    all_labels = np.zeros((num, IMG_DIMS[1], IMG_DIMS[0]), dtype='uint16')

    for i in range(num):
        depth_img = Image.open(data_path(s, DEPTH, i))
        labels_img = Image.open(data_path(s, LABELS, i))

        assert np.all(depth_img.size == IMG_DIMS)
        assert np.all(labels_img.size == IMG_DIMS)

        all_depth[i,:,:] = depth_img

        l_1 = np.array(labels_img)
        l_2 = np.ones((IMG_DIMS[1], IMG_DIMS[0]), dtype='uint16') * 100 # initialize to 100, make sure all 100s are removed

        l_2[np.all(l_1 == COLOR_NONE, axis=2)] = ID_NONE
        l_2[np.all(l_1 == COLOR_TABLE, axis=2)] = ID_TABLE
        l_2[np.all(l_1 == COLOR_HAND, axis=2)] = ID_HAND
        l_2[np.all(l_1 == COLOR_FINGER, axis=2)] = ID_FINGER

        # all -1s have been replaced
        assert np.min(l_2) >= 0
        # all 100s have been replaced
        assert np.max(l_2) == 3

        all_labels[i,:,:] = l_2

    all_depth[np.where(all_depth == 0)] = np.uint16(65535) # max for 16 bit unsigned
    
    return all_depth, all_labels

FEATURE_MAGNITUDE_MAX = 500000.
FEATURE_THRESHOLD_MAX = 1000. # _MIN = -_MAX

# feature is simply a set of xy offsets
def make_random_offset():
    f_theta = np.random.uniform(0, np.pi*2)
    magnitude = np.random.uniform(0, FEATURE_MAGNITUDE_MAX)
    return np.array([np.cos(f_theta), np.sin(f_theta)]) * magnitude

def make_random_feature():
    return make_random_offset(), make_random_offset()

# how about this threshold?
def make_random_threshold():
    return np.random.uniform(-FEATURE_THRESHOLD_MAX, FEATURE_THRESHOLD_MAX)

def make_random_features(n):
    proposal_features = [(make_random_feature(), make_random_threshold()) for i in range(n)]
    return np.array([(p[0][0][0], p[0][0][1], p[0][1][0], p[0][1][1], p[1]) for p in proposal_features ], dtype=np.float32)

print('loading training data')

NUM_TRAIN = 128
NUM_RANDOM_FEATURES = 256

MAX_TREE_DEPTH = 19
MAX_THREADS_PER_BLOCK = 1024 # cuda constant..

# 2 NP arrays of shape (n, y, x), dtype uint16
train_depth, train_labels = load_data(TRAIN, NUM_TRAIN)

print('allocating GPU memory')

train_depth_cu = cu_array.to_gpu(train_depth)
train_labels_cu = cu_array.to_gpu(train_labels)

num_pixels = NUM_TRAIN * IMG_DIMS[1] * IMG_DIMS[0]

# number of nodes required to specify entire tree
TOTAL_TREE_NODES = (2**MAX_TREE_DEPTH) - 1
# number of nodes required to specify tree at the deepest level + 1 (because at the deepest level, nodes still need to be split into L and R children)
MAX_LEAF_NODES = 2**(MAX_TREE_DEPTH)
# number of (floats) for each node
TREE_NODE_ELS = 7 + (NUM_CLASSES * 2)

# tightly packed binary tree..
tree_out_cu = cu_array.GPUArray((TOTAL_TREE_NODES, TREE_NODE_ELS), dtype=np.float32)
tree_out_cu.fill(np.float32(0.))

node_counts = np.zeros((MAX_LEAF_NODES, NUM_CLASSES), dtype=np.uint64)
node_counts[0,1:] = np.unique(train_labels, return_counts=True)[1][1:]

node_counts_cu = cu_array.to_gpu(node_counts)
next_node_counts_cu = cu_array.to_gpu(node_counts)
next_node_counts_by_feature = np.zeros((NUM_RANDOM_FEATURES, MAX_LEAF_NODES, NUM_CLASSES), dtype=np.uint64)
next_node_counts_by_feature_cu = cu_array.to_gpu(next_node_counts_by_feature)

# start each pixel part of group -1
nodes_by_pixel = np.ones((NUM_TRAIN, IMG_DIMS[1], IMG_DIMS[0]), dtype=np.int32) * -1
# anything that isn't NO_LABEL (not 0 depth) gets put in group 0 to start
nodes_by_pixel[np.where(train_labels > 0)] = 0
nodes_by_pixel_cu = cu_array.to_gpu(nodes_by_pixel)
next_nodes_by_pixel_cu = cu_array.to_gpu(nodes_by_pixel)

active_nodes_cu = cu_array.GPUArray((MAX_LEAF_NODES), dtype=np.int32)
next_active_nodes_cu = cu_array.GPUArray((MAX_LEAF_NODES), dtype=np.int32)

next_num_active_nodes_cu = cu_array.GPUArray((1), dtype=np.int32)
next_num_active_nodes_cu.fill(np.int32(1))
def get_next_num_active_nodes():
    return next_num_active_nodes_cu.get()[0]

# random proposal features
proposal_features_cu = cu_array.GPUArray((NUM_RANDOM_FEATURES, 5), dtype=np.float32)

for current_level in range(MAX_TREE_DEPTH):

    num_active_nodes = get_next_num_active_nodes()
    if num_active_nodes == 0:
        break

    print('training level', current_level)

    # make list of random features
    proposal_features_cu.set(make_random_features(NUM_RANDOM_FEATURES))

    # reset next per-feature node counts
    next_node_counts_by_feature_cu.fill(np.uint64(0))

    BLOCK_DIM_X = int(MAX_THREADS_PER_BLOCK // NUM_RANDOM_FEATURES)
    grid_dim = (int(num_pixels // BLOCK_DIM_X) + 1, 1, 1)
    block_dim = (BLOCK_DIM_X, int(NUM_RANDOM_FEATURES), 1)

    cu_eval_random_features(
        np.int32(NUM_TRAIN),
        np.int32(IMG_DIMS[0]),
        np.int32(IMG_DIMS[1]),
        np.int32(NUM_RANDOM_FEATURES),
        np.int32(NUM_CLASSES),
        np.int32(MAX_TREE_DEPTH),
        train_labels_cu,
        train_depth_cu,
        proposal_features_cu,
        nodes_by_pixel_cu,
        next_node_counts_by_feature_cu,
        grid=grid_dim, block=block_dim)

    node_counts_cu.set(next_node_counts_cu)

    pick_best_grid_dim = (int(num_active_nodes // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
    pick_best_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

    next_node_counts_cu.fill(np.uint64(0))
    next_active_nodes_cu.fill(np.int32(0))
    next_num_active_nodes_cu.fill(np.int32(0))

    cu_pick_best_features(
        np.int32(num_active_nodes),
        np.int32(NUM_RANDOM_FEATURES),
        np.int32(MAX_TREE_DEPTH),
        np.int32(NUM_CLASSES),
        np.int32(current_level),
        active_nodes_cu,
        node_counts_cu,
        next_node_counts_by_feature_cu,
        proposal_features_cu,
        tree_out_cu,
        next_node_counts_cu,
        next_active_nodes_cu,
        next_num_active_nodes_cu,
        grid=pick_best_grid_dim, block=pick_best_block_dim)

    if current_level == MAX_TREE_DEPTH - 1:
        break

    next_nodes_by_pixel_cu.fill(np.int32(-1))

    copy_grid_dim = (int(num_pixels // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
    copy_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

    cu_copy_pixel_groups(
        np.int32(NUM_TRAIN),
        np.int32(IMG_DIMS[0]),
        np.int32(IMG_DIMS[1]),
        np.int32(current_level),
        np.int32(MAX_TREE_DEPTH),
        np.int32(NUM_CLASSES),
        train_depth_cu,
        nodes_by_pixel_cu,
        next_nodes_by_pixel_cu,
        tree_out_cu,
        grid = copy_grid_dim, block=copy_block_dim)

    nodes_by_pixel_cu.set(next_nodes_by_pixel_cu)
    active_nodes_cu.set(next_active_nodes_cu)

print('tree training done..')
print('beginnign evaluation..')
NUM_TEST = 16

# 2 NP arrays of shape (n, y, x), dtype uint16
test_depth, test_labels = load_data(TEST, NUM_TEST) # just use training data for now - generate dedicated test data shortly
test_depth_cu = cu_array.to_gpu(test_depth)

test_output_labels_cu = cu_array.GPUArray((NUM_TEST, IMG_DIMS[1], IMG_DIMS[0]), dtype=np.uint16)

num_test_pixels = NUM_TEST * IMG_DIMS[1] * IMG_DIMS[0]

grid_dim = (int(num_test_pixels // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

cu_eval_image(
    np.int32(NUM_TEST),
    np.int32(IMG_DIMS[0]),
    np.int32(IMG_DIMS[1]),
    np.int32(NUM_CLASSES),
    np.int32(MAX_TREE_DEPTH),
    test_depth_cu,
    tree_out_cu,
    test_output_labels_cu,
    grid=grid_dim, block=block_dim)

test_output_labels = test_output_labels_cu.get()

test_output_labels_render = np.zeros((NUM_TEST, IMG_DIMS[1], IMG_DIMS[0], 4), dtype='uint8')
test_output_labels_render[np.where(test_output_labels == ID_TABLE)] = COLOR_TABLE
test_output_labels_render[np.where(test_output_labels == ID_HAND)] = COLOR_HAND
test_output_labels_render[np.where(test_output_labels == ID_FINGER)] = COLOR_FINGER

test_output_labels[np.where(test_output_labels == 0)] = 65535 # make sure 0s dont match
pct_match =  np.sum(test_output_labels == test_labels) / np.sum(test_labels > 0)

print('pct. matching pixels: ', pct_match)
print('saving renders..')

for i in range(NUM_TEST):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
