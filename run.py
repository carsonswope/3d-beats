import numpy as np
from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array
from pycuda.compiler import SourceModule

np.set_printoptions(suppress=True)

# load/compile cuda kernels..
cu_file = open('tree_train.cu', 'r')
cu_text = cu_file.read()
cu_mod = SourceModule(cu_text)
cu_eval_random_features = cu_mod.get_function('evaluate_random_features')
cu_eval_image = cu_mod.get_function('evaluate_image_using_tree')

DEPTH = 'depth'
LABELS = 'labels'
TEST = 'test'
TRAIN = 'train'

IMG_DIMS = np.array([424, 240], dtype=np.int)

COLOR_NONE = np.array([0, 0, 0, 0], dtype='uint16')
COLOR_TABLE = np.array([255, 0, 0, 255], dtype='uint16')
COLOR_HAND = np.array([0, 0, 255, 255], dtype='uint16')

ID_NONE = 0
ID_TABLE = 1
ID_HAND = 2
NUM_CLASSES = 3

DATA_ROOT = 'datagen/generated'
# s: TEST or TRAIN (set)
# t: DEPTH or LABELS (type)
def data_path(s, t, i):
    return DATA_ROOT + '/' + s + '_' + str(i).zfill(8) + '_' + t + '.png'

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

        # all -1s have been replaced
        assert np.min(l_2) > -1
        assert np.max(l_2) == 2

        all_labels[i,:,:] = l_2
    
    return all_depth, all_labels


# i: np image[row][col]
# x: coords in x,y ?
def read_pixel(i, x):
    x = np.round(x).astype('int')
    if (x[0] < 0 or x[1] < 0):
        return 0
    if (x[0] >= IMG_DIMS[0] or x[1] >= IMG_DIMS[1]):
        return 0
    return i[x[1]][x[0]]

# i: image
# x: pixel coord
# u: offset 1. 
# v: offset 2
def compute_feature(i, x, u, v):
    d_x = read_pixel(i, x)
    u_coord = x + (u / d_x)
    v_coord = x + (v / d_x)
    # convert to floats..
    d_u = read_pixel(i, u_coord) * 1.
    d_v = read_pixel(i, v_coord) * 1.
    return d_u - d_v

# feature is simply a set of xy offsets
def make_random_offset():
    f_theta = np.random.uniform(0, np.pi*2)
    magnitude = np.random.uniform(0, 500000)
    return np.array([np.cos(f_theta), np.sin(f_theta)]) * magnitude

def make_random_feature():
    return make_random_offset(), make_random_offset()

# how about this threshold?
def make_random_threshold():
    return np.random.uniform(-500., 500.)

# pixel idx = single number for image_num, x, y
def get_pixel_info(pixel_id):
    i = pixel_id // (IMG_DIMS[1] * IMG_DIMS[0])
    # remainder..
    pixel_id = pixel_id % (IMG_DIMS[1] * IMG_DIMS[0])
    y = pixel_id // IMG_DIMS[0]
    x = pixel_id % IMG_DIMS[0]
    return i, x, y

# more efficient versions of gini that don't involve linear operation on class list each time
def gini_impurity2(c):
    # c = (num0, num1)
    # counts of class0 and class1 in set
    assert c[0] + c[1] > 0
    p = c[0] / (c[0] + c[1])
    return 2 * p * (1-p)

def gini_gain2(p, cs):
    # p = (num0, num1)
    # cs = [(num0, num1), (num0, num1), ...]
    p_count = p[0] + p[1]
    previous = gini_impurity2(p)
    remainder = sum([((c[0] + c[1]) / p_count) * gini_impurity2(c) for c in cs])
    return previous - remainder

print('loading training data..')

NUM_TRAIN = 64
# 2 NP arrays of shape (n, y, x), dtype uint16
train_depth, train_labels = load_data(TRAIN, NUM_TRAIN)

print('training tree..')

train_depth_cu = cu_array.to_gpu(train_depth)
train_labels_cu = cu_array.to_gpu(train_labels)

num_pixels = NUM_TRAIN * IMG_DIMS[1] * IMG_DIMS[0]

MAX_TREE_DEPTH = 10
max_groups = 2**MAX_TREE_DEPTH # decision tree could have this many leaf nodes!

NUM_RANDOM_FEATURES = 32

group_counts = np.zeros((max_groups, NUM_CLASSES), dtype=np.uint64)
group_counts[0,1:] = np.unique(train_labels, return_counts=True)[1][1:]

# start each pixel part of group -1
groups = np.ones((NUM_TRAIN, IMG_DIMS[1], IMG_DIMS[0]), dtype=np.int32) * -1
# anything that isn't NO_LABEL (not 0 depth) gets put in group 0 to start
groups[np.where(train_labels > 0)] = 0
groups_cu = cu_array.to_gpu(groups)

# empty proposals buffer
proposal_features_flat = np.zeros((NUM_RANDOM_FEATURES, 5), dtype=np.float32)
proposal_features_cu = cu_array.to_gpu(proposal_features_flat)

# empty 'next group counts'
next_group_counts = np.zeros((NUM_RANDOM_FEATURES, max_groups, NUM_CLASSES), dtype=np.uint64)
next_group_counts_cu = cu_array.to_gpu(next_group_counts)

# empty 'next groups': for each random 
next_groups = np.ones((NUM_RANDOM_FEATURES, NUM_TRAIN, IMG_DIMS[1], IMG_DIMS[0]), dtype=np.int32) * -1
next_groups_cu = cu_array.to_gpu(next_groups)

active_groups = [0]
active_next_groups = []

tree_out = np.zeros((MAX_TREE_DEPTH, max_groups, 7), dtype=np.float32) # ux,uy,vx,vy,thresh,l_class,r_class - if class is set to -1, then evaluation continues!
leaf_pdfs = np.zeros((max_groups, NUM_CLASSES), dtype=np.float32)

for current_level in range(MAX_TREE_DEPTH):

    # make list of random features
    proposal_features = [(make_random_feature(), make_random_threshold()) for i in range(NUM_RANDOM_FEATURES)]
    proposal_features_flat = np.array([(p[0][0][0], p[0][0][1], p[0][1][0], p[0][1][1], p[1]) for p in proposal_features ], dtype=np.float32)
    proposal_features_cu.set(proposal_features_flat)

    # reset next group counts
    next_group_counts[:,:,:] = 0
    next_group_counts_cu.set(next_group_counts)

    # reset next groups
    next_groups[:,:,:,:] = -1
    next_groups_cu.set(next_groups)

    grid_dim = (int(num_pixels / 32) + 1, 1, 1)
    block_dim = (32, NUM_RANDOM_FEATURES, 1)

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
        groups_cu,
        next_groups_cu,
        next_group_counts_cu,
        grid=grid_dim, block=block_dim)

    next_groups_cu.get(next_groups)
    next_group_counts_cu.get(next_group_counts)

    # reset all groups. will be re-set when evaluating results of kernel
    groups[:] = -1

    processed_group_counts = np.zeros((max_groups, NUM_CLASSES), dtype=np.uint64)

    for parent_group in active_groups:
        if parent_group == -1:
            continue

        left_child = parent_group * 2
        right_child = (parent_group * 2) + 1

        parent_counts = np.copy(group_counts[parent_group][1:])

        # check reach random feature for the performance of X group ID
        best_g = -1
        best_g_idx = None
        best_left_counts = None
        best_right_counts = None

        for f in range(NUM_RANDOM_FEATURES):
            left_counts = next_group_counts[f][left_child][1:]
            right_counts = next_group_counts[f][right_child][1:]

            assert np.sum(left_counts) + np.sum(right_counts) == np.sum(parent_counts)

            if not np.sum(left_counts) or not np.sum(right_counts):
                g = 0
            else:
                g = gini_gain2(parent_counts, [left_counts, right_counts])

            if g > best_g:
                best_g = g
                best_g_idx = f
                best_left_counts = np.copy(left_counts)
                best_right_counts = np.copy(right_counts)

        assert np.all(best_right_counts + best_left_counts == parent_counts)

        # not a single random feature provided any gain!
        if best_g == 0:
            parent_counts_normalized = parent_counts / np.sum(parent_counts)
            # this edge case still needs to be handled somehow
            print('error!!')
        
        # current tree level / current tree node idx
        tree_out[current_level][parent_group][0:5] = np.copy(proposal_features_flat[best_g_idx])

        left_group_pixels = np.where(next_groups[best_g_idx] == left_child)
        right_group_pixels = np.where(next_groups[best_g_idx] == right_child)
        
        assert left_group_pixels[0].shape[0] + right_group_pixels[0].shape[0] == np.sum(parent_counts)

        CUTOFF_THRESH = 0.999

        # determine left node
        best_left_counts_normalized = best_left_counts / np.sum(best_left_counts)
        if np.any(best_left_counts_normalized > CUTOFF_THRESH):
            # threshold is reached.
            # treat as leaf node
            out_class = np.where(best_left_counts_normalized > CUTOFF_THRESH)[0][0] + 1 # add one because first element in array was removed before
            tree_out[current_level][parent_group][5] = out_class
        else:
            # keep tree going on left!
            tree_out[current_level][parent_group][5] = -1
            groups[left_group_pixels] = left_child
            active_next_groups.append(left_child)
            processed_group_counts[left_child][1:] = np.copy(best_left_counts)
            if current_level == MAX_TREE_DEPTH - 1:
                leaf_pdfs[left_child,0] = 0.
                leaf_pdfs[left_child,1:] = np.copy(best_left_counts_normalized)

        best_right_counts_normalized = best_right_counts / np.sum(best_right_counts)
        if np.any(best_right_counts_normalized > CUTOFF_THRESH):
            # threshold is reached
            # treat as leaf node
            out_class = np.where(best_right_counts_normalized > CUTOFF_THRESH)[0][0] + 1
            tree_out[current_level][parent_group][6] = out_class
        else:
            tree_out[current_level][parent_group][6] = -1
            groups[right_group_pixels] = right_child
            active_next_groups.append(right_child)
            processed_group_counts[right_child][1:] = np.copy(best_right_counts)
            if current_level == MAX_TREE_DEPTH - 1:
                leaf_pdfs[right_child,0] = 0.
                leaf_pdfs[right_child,1:] = np.copy(best_right_counts_normalized)
    
    groups_cu.set(groups)
    group_counts = np.copy(processed_group_counts)
    active_groups = active_next_groups.copy()
    active_next_groups = []

print('tree training done..')

print('beginnign evaluation..')

NUM_TEST = 16
# 2 NP arrays of shape (n, y, x), dtype uint16
test_depth, test_labels = load_data(TRAIN, NUM_TEST) # just use training data for now - generate dedicated test data shortly

test_output_labels = np.zeros((NUM_TEST, IMG_DIMS[1], IMG_DIMS[0]), dtype=np.uint16)
test_output_labels_cu = cu_array.to_gpu(test_output_labels)

test_depth_cu = cu_array.to_gpu(test_depth)
tree_out_cu = cu_array.to_gpu(tree_out)
leaf_pdfs_cu = cu_array.to_gpu(leaf_pdfs)

num_test_pixels = NUM_TEST * IMG_DIMS[1] * IMG_DIMS[0]

grid_dim = (int(num_test_pixels / 256) + 1, 1, 1)
block_dim = (256, 1, 1)

cu_eval_image(
    np.int32(NUM_TEST),
    np.int32(IMG_DIMS[0]),
    np.int32(IMG_DIMS[1]),
    np.int32(NUM_CLASSES),
    np.int32(MAX_TREE_DEPTH),
    test_depth_cu,
    tree_out_cu,
    leaf_pdfs_cu,
    test_output_labels_cu,
    grid=grid_dim, block=block_dim)

test_output_labels_cu.get(test_output_labels)

"""
# tree out, leaf pdfs
def eval_pixel(img, pixel_x, pixel_y, decision_tree, leaf_pdfs):
    group = 0
    for i in range(MAX_TREE_DEPTH):
        criteria = decision_tree[i][group]
        p = np.array([pixel_x, pixel_y])
        u = np.array([criteria[0], criteria[1]])
        v = np.array([criteria[2], criteria[3]])
        f = compute_feature(img, p, u, v)
        thresh = criteria[4]

        if f < thresh:
            # left
            next_group = int(criteria[5])
            if next_group == -1:
                group = (group * 2)
            else:
                # known output!
                out_pdf = np.zeros(NUM_CLASSES, dtype=np.float32)
                out_pdf[next_group] = 1.
                return out_pdf
        else:
            next_group = int(criteria[6])
            if next_group == -1:
                group = (group * 2) + 1
            else:
                # known output!
                out_pdf = np.zeros(NUM_CLASSES, dtype=np.float32)
                out_pdf[next_group] = 1.
                return out_pdf

    return np.copy(leaf_pdfs[group])
"""

test_output_labels_render = np.zeros((NUM_TEST, IMG_DIMS[1], IMG_DIMS[0], 4), dtype='uint8')
test_output_labels_render[np.where(test_output_labels == ID_TABLE)] = COLOR_TABLE
test_output_labels_render[np.where(test_output_labels == ID_HAND)] = COLOR_HAND

matching_pixels = np.sum(test_output_labels == test_labels)
sh = test_output_labels.shape
total_test_pixels = sh[0] * sh[1] * sh[2]
pct_match = matching_pixels / total_test_pixels

print('pct. matching pixels: ', pct_match)
print('saving renders..')

for i in range(NUM_TEST):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')

# px = eval_pixel(train_depth[0], 214, 179, tree_out, leaf_pdfs)
# print('hand', px)

# pz = eval_pixel(train_depth[0], 288, 198, tree_out, leaf_pdfs)
# print('table', pz)

# print('uhh')