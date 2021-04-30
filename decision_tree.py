import py_nvcc_utils
import json
import numpy as np
from PIL import Image

import pycuda.gpuarray as cu_array


MAX_THREADS_PER_BLOCK = 1024 # cuda constant..

MAX_UINT16 = np.uint16(65535) # max for 16 bit unsigned

# configuration for a decision tree dataset.
# image size
# mapping from label colors to int IDs
class DecisionTreeDatasetConfig():

    def __init__(self, dataset_dir):

        self.dataset_dir = dataset_dir
        cfg = json.loads(open(dataset_dir + 'config.json').read())

        self.img_dims = tuple(cfg['img_dims'])
        self.id_to_color = {0: np.array([0, 0, 0, 0], dtype=np.uint8)}
        for i,c in cfg['id_to_color'].items():
            self.id_to_color[int(i)] = np.array(c, dtype=np.uint8)
        
        self.num_train = cfg['num_train']
        self.num_test = cfg['num_test']

        self.train = DecisionTreeDataset(dataset_dir + 'train', self.num_train, self)
        self.test = DecisionTreeDataset(dataset_dir + 'test', self.num_test, self)

    
    def num_classes(self):
        return len(self.id_to_color)

    def convert_colors_to_ids(self, labels_color):
        labels_ids = np.zeros((self.img_dims[1], self.img_dims[0]), dtype=np.uint16)

        labelled_pixels_count = 0
        for class_id,color in self.id_to_color.items():
            pixels_of_color = np.all(labels_color == color, axis=2)
            labels_ids[pixels_of_color] = class_id
            labelled_pixels_count += np.sum(pixels_of_color)

        # make sure every pixel was labelled
        assert(labelled_pixels_count == self.img_dims[0] * self.img_dims[1])
        return labels_ids

    def convert_ids_to_colors(self, labels_ids):
        num_images, y_dim, x_dim = labels_ids.shape

        assert y_dim == self.img_dims[1]
        assert x_dim == self.img_dims[0]

        labels_colors = np.zeros((num_images, y_dim, x_dim, 4), dtype=np.uint8)
        for class_id,color in self.id_to_color.items():
            labels_colors[np.where(labels_ids == class_id)] = color
        return labels_colors

class DecisionTreeDataset():
    def __init__(self, path_root, num_images, config):
        self.num_images = num_images
        self.config = config
        self.depth, self.labels = self.load_data(path_root, num_images)
        self.depth_cu = cu_array.to_gpu(self.depth)
        self.labels_cu = cu_array.to_gpu(self.labels)

    # s: TEST or TRAIN (set)
    # t: DEPTH or LABELS (type)
    @staticmethod
    def __data_path(s, t, i):
        return s + '_' + str(i).zfill(8) + '_' + t + '.png'

    def load_data(self, s, num):
        all_depth = np.zeros((num, self.config.img_dims[1], self.config.img_dims[0]), dtype='uint16')
        all_labels = np.zeros((num, self.config.img_dims[1], self.config.img_dims[0]), dtype='uint16')

        DEPTH = 'depth'
        LABELS = 'labels'

        for i in range(num):
            depth_img = Image.open(DecisionTreeDataset.__data_path(s, DEPTH, i))
            labels_img = Image.open(DecisionTreeDataset.__data_path(s, LABELS, i))

            assert np.all(depth_img.size == self.config.img_dims)
            assert np.all(labels_img.size == self.config.img_dims)

            all_depth[i,:,:] = depth_img
            all_labels[i,:,:] = self.config.convert_colors_to_ids(np.array(labels_img))

        # convert 0 entries in depth image to max value
        all_depth[np.where(all_depth == 0)] = MAX_UINT16
        
        return all_depth, all_labels

    def num_pixels(self):
        return self.num_images * self.config.img_dims[0] * self.config.img_dims[1]

    def images_shape(self):
        return (self.num_images, self.config.img_dims[1], self.config.img_dims[0])

class DecisionTree():
    def __init__(self, max_depth, num_classes):
        self.max_depth = max_depth
        self.num_classes = num_classes

        self.TOTAL_TREE_NODES, self.MAX_LEAF_NODES, self.TREE_NODE_ELS = DecisionTree.get_config(max_depth, num_classes)

        # tightly packed binary tree..
        self.tree_out_cu = cu_array.GPUArray((self.TOTAL_TREE_NODES, self.TREE_NODE_ELS), dtype=np.float32)
        self.tree_out_cu.fill(np.float32(0.))

    @staticmethod
    def get_config(max_depth, num_classes):
        # number of nodes required to specify entire tree
        TOTAL_TREE_NODES = (2**max_depth) - 1
        # number of nodes required to specify tree at the deepest level + 1 (because at the deepest level, nodes still need to be split into L and R children)
        MAX_LEAF_NODES = 2**(max_depth)
        # number of (floats) for each node
        TREE_NODE_ELS = 7 + (num_classes * 2)

        return (TOTAL_TREE_NODES, MAX_LEAF_NODES, TREE_NODE_ELS)

class DecisionForest():
    def __init__(self, num_trees, max_depth, num_classes):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.num_classes = num_classes

        self.TOTAL_TREE_NODES, self.MAX_LEAF_NODES, self.TREE_NODE_ELS = DecisionTree.get_config(max_depth, num_classes)

        self.forest_cu = cu_array.GPUArray((self.num_trees, self.TOTAL_TREE_NODES, self.TREE_NODE_ELS), dtype=np.float32)
        self.forest_cu.fill(np.float32(0.))

    # def eval()
class DecisionTreeEvaluator():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('tree_eval.cu')
        self.cu_eval_image = cu_mod.get_function('evaluate_image_using_tree')
        self.cu_eval_image_forest = cu_mod.get_function('evaluate_image_using_forest')
    
    def get_labels(self, tree, depth_images_in, labels_out):
        num_images, dim_y, dim_x = depth_images_in.shape

        num_test_pixels = num_images * dim_y * dim_x

        grid_dim = (int(num_test_pixels // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
        block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

        self.cu_eval_image(
            np.int32(num_images),
            np.int32(dim_x),
            np.int32(dim_y),
            np.int32(tree.num_classes),
            np.int32(tree.max_depth),
            depth_images_in,
            tree.tree_out_cu,
            labels_out,
            grid=grid_dim, block=block_dim)

        
    def get_labels_forest(self, forest, depth_images_in, labels_out):
        num_images, dim_y, dim_x = depth_images_in.shape

        num_test_pixels = num_images * dim_y * dim_x

        BLOCK_DIM_X = int(MAX_THREADS_PER_BLOCK // forest.num_trees) 
        grid_dim = (int(num_test_pixels // BLOCK_DIM_X) + 1, 1, 1)
        block_dim = (BLOCK_DIM_X, forest.num_trees, 1)

        self.cu_eval_image_forest(
            np.int32(forest.num_trees),
            np.int32(num_images),
            np.int32(dim_x),
            np.int32(dim_y),
            np.int32(forest.num_classes),
            np.int32(forest.max_depth),
            np.int32(BLOCK_DIM_X),
            depth_images_in,
            forest.forest_cu,
            labels_out,
            grid=grid_dim, block=block_dim, shared=(BLOCK_DIM_X * forest.num_classes * 4)) # sizeof(float), right?


# TODO: modify these thresholds during the training process..
# but how?

FEATURE_MAGNITUDE_MAX = 14.
FEATURE_THRESHOLD_MAX = 11. # _MIN = -_MAX

def make_random_offset():
    f_theta = np.random.uniform(0, np.pi*2)
    # sample linearly in log space :)
    magnitude = np.power(np.e, np.random.uniform(0, FEATURE_MAGNITUDE_MAX))
    return np.array([np.cos(f_theta), np.sin(f_theta)]) * magnitude

def make_random_feature():
    return make_random_offset(), make_random_offset()

# how about this threshold?
def make_random_threshold():
    return np.random.choice([-1, 1]) * np.power(np.e, np.random.uniform(0, FEATURE_THRESHOLD_MAX))

def make_random_features(n):
    proposal_features = [(make_random_feature(), make_random_threshold()) for i in range(n)]
    return np.array([(p[0][0][0], p[0][0][1], p[0][1][0], p[0][1][1], p[1]) for p in proposal_features ], dtype=np.float32)

class DecisionTreeTrainer():
    def __init__(self):
        # first load kernels..
        cu_mod = py_nvcc_utils.get_module('tree_train.cu')
        self.cu_eval_random_features = cu_mod.get_function('evaluate_random_features')
        self.cu_pick_best_features = cu_mod.get_function('pick_best_features')
        self.cu_copy_pixel_groups = cu_mod.get_function('copy_pixel_groups')

    def allocate(self, dataset, NUM_RANDOM_FEATURES, MAX_TREE_DEPTH):

        # then,
        self.NUM_RANDOM_FEATURES = NUM_RANDOM_FEATURES
        self.MAX_TREE_DEPTH = MAX_TREE_DEPTH

        _, self.MAX_LEAF_NODES, _ = DecisionTree.get_config(MAX_TREE_DEPTH, dataset.config.num_classes())

        # allocate GPU space needed for training, given the configuration:
        self.node_counts = np.zeros((self.MAX_LEAF_NODES, dataset.config.num_classes()), dtype=np.uint64)

        self.node_counts_cu = cu_array.to_gpu(self.node_counts)
        self.next_node_counts_cu = cu_array.to_gpu(self.node_counts)
        self.next_node_counts_by_feature = np.zeros((self.NUM_RANDOM_FEATURES, self.MAX_LEAF_NODES, dataset.config.num_classes()), dtype=np.uint64)
        self.next_node_counts_by_feature_cu = cu_array.to_gpu(self.next_node_counts_by_feature)

        self.nodes_by_pixel = np.zeros(dataset.images_shape(), dtype=np.int32)
        self.nodes_by_pixel_cu = cu_array.to_gpu(self.nodes_by_pixel)
        self.next_nodes_by_pixel_cu = cu_array.to_gpu(self.nodes_by_pixel)

        self.active_nodes_cu = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.int32)
        self.next_active_nodes_cu = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.int32)

        self.next_num_active_nodes_cu = cu_array.GPUArray((1), dtype=np.int32)
        self.get_next_num_active_nodes = lambda : self.next_num_active_nodes_cu.get()[0]

        # random proposal features
        self.proposal_features_cu = cu_array.GPUArray((self.NUM_RANDOM_FEATURES, 5), dtype=np.float32)

    def train(self, dataset, tree):

        tree.tree_out_cu.fill(np.float(0.))

        # start conditions for iteration
        # original distribution of classes as node counts
        self.node_counts[:] = 0
        self.node_counts[0,1:] = np.unique(dataset.labels, return_counts=True)[1][1:]
        self.node_counts_cu.set(self.node_counts)

        self.nodes_by_pixel[:] = -1
        self.nodes_by_pixel[np.where(dataset.labels > 0)] = 0
        self.nodes_by_pixel_cu.set(self.nodes_by_pixel)

        # 1 node to start, at idx 0
        self.active_nodes_cu.fill(np.int32(0))
        self.next_num_active_nodes_cu.fill(np.int32(1))

        for current_level in range(self.MAX_TREE_DEPTH):

            num_active_nodes = self.get_next_num_active_nodes()
            if num_active_nodes == 0:
                break

            # print('training level', current_level)

            # make list of random features
            self.proposal_features_cu.set(make_random_features(self.NUM_RANDOM_FEATURES))

            # reset next per-feature node counts
            self.next_node_counts_by_feature_cu.fill(np.uint64(0))

            BLOCK_DIM_X = int(MAX_THREADS_PER_BLOCK // self.NUM_RANDOM_FEATURES)
            grid_dim = (int(dataset.num_pixels() // BLOCK_DIM_X) + 1, 1, 1)
            block_dim = (BLOCK_DIM_X, int(self.NUM_RANDOM_FEATURES), 1)

            self.cu_eval_random_features(
                np.int32(dataset.num_images),
                np.int32(dataset.config.img_dims[0]),
                np.int32(dataset.config.img_dims[1]),
                np.int32(self.NUM_RANDOM_FEATURES),
                np.int32(dataset.config.num_classes()),
                np.int32(self.MAX_TREE_DEPTH),
                dataset.labels_cu,
                dataset.depth_cu,
                self.proposal_features_cu,
                self.nodes_by_pixel_cu,
                self.next_node_counts_by_feature_cu,
                grid=grid_dim, block=block_dim)

            pick_best_grid_dim = (int(num_active_nodes // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
            pick_best_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

            self.next_node_counts_cu.fill(np.uint64(0))
            self.next_active_nodes_cu.fill(np.int32(0))
            self.next_num_active_nodes_cu.fill(np.int32(0))

            self.cu_pick_best_features(
                np.int32(num_active_nodes),
                np.int32(self.NUM_RANDOM_FEATURES),
                np.int32(self.MAX_TREE_DEPTH),
                np.int32(dataset.config.num_classes()),
                np.int32(current_level),
                self.active_nodes_cu,
                self.node_counts_cu,
                self.next_node_counts_by_feature_cu,
                self.proposal_features_cu,
                tree.tree_out_cu,
                self.next_node_counts_cu,
                self.next_active_nodes_cu,
                self.next_num_active_nodes_cu,
                grid=pick_best_grid_dim, block=pick_best_block_dim)

            if current_level == self.MAX_TREE_DEPTH - 1:
                break

            self.node_counts_cu.set(self.next_node_counts_cu)

            self.next_nodes_by_pixel_cu.fill(np.int32(-1))

            copy_grid_dim = (int(dataset.num_pixels() // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
            copy_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

            self.cu_copy_pixel_groups(
                np.int32(dataset.num_images),
                np.int32(dataset.config.img_dims[0]),
                np.int32(dataset.config.img_dims[1]),
                np.int32(current_level),
                np.int32(self.MAX_TREE_DEPTH),
                np.int32(dataset.config.num_classes()),
                dataset.depth_cu,
                self.nodes_by_pixel_cu,
                self.next_nodes_by_pixel_cu,
                tree.tree_out_cu,
                grid = copy_grid_dim, block=copy_block_dim)

            self.nodes_by_pixel_cu.set(self.next_nodes_by_pixel_cu)
            self.active_nodes_cu.set(self.next_active_nodes_cu)

        # err, return
        # tree buffer should be filled up now!