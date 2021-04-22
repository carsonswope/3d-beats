import numpy as np
from PIL import Image

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
        assert np.min(l_2) == 0
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
    # if d_x == 0:
        # d_x = 65535
    u_coord = x + (u / d_x)
    v_coord = x + (v / d_x)
    # convert to floats..
    d_u = read_pixel(i, u_coord) * 1.
    d_v = read_pixel(i, v_coord) * 1.
    return d_u - d_v

# feature is simply a set of xy offsets
def make_random_offset():
    f_theta = np.random.uniform(0, np.pi*2)
    magnitude = np.random.uniform(0, 100000)
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
    p = c[0] / (c[0] + c[1])
    return 2 * p * (1-p)

def gini_gain2(p, cs):
    # p = (num0, num1)
    # cs = [(num0, num1), (num0, num1), ...]
    p_count = p[0] + p[1]
    previous = gini_impurity2(p)
    remainder = sum([((c[0] + c[1]) / p_count) * gini_impurity2(c) for c in cs])
    return previous - remainder

NUM_TRAIN = 128
NUM_TEST = 4

# 2 NP arrays of shape (n, y, x), dtype uint16
test_depth, test_labels = load_data(TEST, NUM_TEST)

num_pixels = NUM_TEST * IMG_DIMS[1] * IMG_DIMS[0]

MAX_TREE_DEPTH = 8
max_groups = 2**MAX_TREE_DEPTH # decision tree could have this many leaf nodes!

#
group_counts = np.zeros((max_groups, NUM_CLASSES), dtype=np.int64)
# initialize counts of each class in group 0. Don't keep track of 'NONE' group - we won't even try to classify that group!
group_counts[0,1:3] = np.unique(test_labels, return_counts=True)[1][1:3]

# each round of training should be a per-pixel operation
# buffer keeping track of which group each buffer is in
groups = np.zeros(num_pixels, dtype=np.int32)

NUM_RANDOM_FEATURES_TO_TRY = 8


# proposal_features = [(make_random_feature(), make_random_threshold()) for i in range(NUM_RANDOM_FEATURES_TO_TRY)]
# ((u,v),thresh)
proposal_features = [
    (([-22799.52010017,  31844.44542072], [-23159.7254717 ,  65518.98616169]), 224.99636734542378),
    (([  -692.84064766, -50510.43509532], [-54872.58399917, -18403.33368239]), 377.3484252752879),
    (([42677.79857804, 42241.78039404], [ 12208.86884946, -44481.27711877]), -324.198649832124),
    (([-13328.76273914,  -4973.62915883], [  -521.67342981, -19904.1950021 ]), 256.12471173659424),
    (([ 1063.90107223, 74731.37571924], [-29564.23398385, -57884.01713972]), 144.03060763675796),
    (([ 26590.19023191, -17289.6001013 ], [18140.50055884, 73247.22299132]), 220.399343623307),
    (([52841.11705466, 47489.93466609], [-77901.91054828, -41492.58656817]), -160.12803605488904),
    (([-46233.29062441,  72860.17060032], [-44751.44155779,   2178.14506526]), 289.6863955782877)
]

next_group_counts = np.zeros((NUM_RANDOM_FEATURES_TO_TRY, max_groups, NUM_CLASSES), dtype=np.int64)
next_groups = np.zeros((NUM_RANDOM_FEATURES_TO_TRY, num_pixels), dtype=np.int32)
for p_id in range(num_pixels):
    p_i, p_x, p_y = get_pixel_info(p_id)
    p_label = read_pixel(test_labels[p_i], (p_x, p_y))
    # if 'none' label, no pixel there. don't even try to evaluate!
    if p_label == ID_NONE:
        pass
    else:
        for f in range(NUM_RANDOM_FEATURES_TO_TRY):
            f_offset, f_thresh = proposal_features[f]
            f_result = compute_feature(test_depth[p_i], (p_x, p_y), f_offset[0], f_offset[1])
            if f_result < f_thresh:
                # left path
                next_group = (groups[p_id] * 2)
            else:
                # right path
                next_group = (groups[p_id] * 2) + 1

            next_groups[f, p_id] = next_group
            next_group_counts[f, next_group, p_label] += 1

"""
expected_out_counts = np.array([
    [[244916,  25883], [  5835,    309]],
    [[246830,  24988], [  3921,   1204]],
    [[  8049,   2031], [242702,  24161]],
    [[248269,  25852], [  2482,    340]],
    [[195454,  18763], [ 55297,   7429]],
    [[238832,  24910], [ 11919,   1282]],
    [[ 11582,   4214], [239169,  21978]],
    [[248306,  22469], [  2445,   3723]]])
"""

best_gain = 0
best_gain_f = None

actual_gains = []
for f in range(NUM_RANDOM_FEATURES_TO_TRY):
    start_group_counts = group_counts[0][1:]
    next_group_counts_f = next_group_counts[f,0:2,1:]
    g = gini_gain2(start_group_counts, next_group_counts_f)
    actual_gains.append(g)
    print('gain: ', g)
    if g > best_gain:
        best_gain = g
        best_gain_f = proposal_features[f]

expected_gains = np.array([
    8.898102710525047e-05,
    0.0007428125288906906,
    0.000863496483863968,
    1.3818551075961416e-05,
    0.00033340808218870754,
    6.450454232143077e-07,
    0.0035872638603065277,
    0.011804347342051935])

assert np.allclose(expected_gains, actual_gains)

