import cuda.py_nvcc_utils as py_nvcc_utils
import json
import numpy as np

from pathlib import Path
import os.path

import pycuda.gpuarray as cu_array
import pycuda.driver as cu

from compressed_blocks import *
from engine.buffer import GpuBuffer

from util import sizeof_fmt, MAX_UINT16

MAX_THREADS_PER_BLOCK = 1024 # cuda constant..

# configuration for a decision tree dataset.
# image size
# mapping from label colors to int IDs
class DecisionTreeDatasetConfig():

    # randomly split up dataset into chunks of the requested size
    def multiple(dataset_dir, images):

        total_images = json.loads(open(dataset_dir + 'config.json').read())['num_images']
        num_images_to_fetch = sum([num_images for num_images, _, _ in images])
        assert num_images_to_fetch <= total_images

        images_to_fetch = list(range(total_images))
        np.random.shuffle(images_to_fetch)

        datasets = []
        start_i = 0
        for num_images, images_per_block, imgs_name in images:
            images_per_block = images_per_block or num_images

            datasets.append(DecisionTreeDatasetConfig(dataset_dir, 
                num_images=num_images,
                images_per_block=images_per_block,
                imgs_name=imgs_name))
            start_i += num_images

        return tuple(datasets)

    def __init__(self, dataset_dir, num_images=0, images_per_block=0, imgs_name='data0'):

        self.dataset_dir = dataset_dir
        cfg = json.loads(open(dataset_dir + 'config.json').read())
        self.cfg = cfg
        self.imgs_name = imgs_name

        self.img_dims = tuple(cfg['img_dims'])
        self.id_to_color = {0: np.array([0, 0, 0, 0], dtype=np.uint8)}
        for i,c in cfg['id_to_color'].items():
            self.id_to_color[int(i)] = np.array(c, dtype=np.uint8)

        self.total_available_images = cfg['num_images']

        self.num_images = num_images
        if self.num_images == 0:
            return

        self.images_per_block = images_per_block or self.num_images

        assert self.num_images % self.images_per_block == 0
        self.num_image_blocks = self.num_images // self.images_per_block

        total_images = self.cfg['num_images']
        img_idxes = list(range(total_images))
        np.random.shuffle(img_idxes)
        img_idxes = img_idxes[0:self.num_images]

        def get_image_block(i, arr_out, name):
            assert arr_out.shape == (self.images_per_block, self.img_dims[1], self.img_dims[0])
            assert arr_out.dtype == np.uint16
            for j in range(self.images_per_block):
                img_idx = (i * self.images_per_block) + j
                img_idx = img_idxes[img_idx]
                arr_out[j] = np.array(Image.open(f'{self.dataset_dir}/{str(img_idx).zfill(8)}_{name}.png')).astype(np.uint16)

        self.depth_blocks = CompressedBlocksStatic(self.num_image_blocks, self.images_per_block, self.img_dims, lambda i,a: get_image_block(i,a,'depth'), self.imgs_name + '/depth')
        self.labels_blocks = CompressedBlocksStatic(self.num_image_blocks, self.images_per_block, self.img_dims, lambda i,a: get_image_block(i,a,'labels'), self.imgs_name + '/labels')

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

    def get_depth_block_cu(self, block_num, arr_out):
        self.depth_blocks.get_block_cu(block_num, arr_out)

    def get_labels_block_cu(self, block_num, arr_out):
        self.labels_blocks.get_block_cu(block_num, arr_out)

    def num_pixels(self):
        return self.num_images * self.img_dims[0] * self.img_dims[1]

    def images_shape(self):
        return (self.num_images, self.img_dims[1], self.img_dims[0])

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
    @staticmethod
    def load(model_filename):
        forest_cpu = np.load(model_filename)

        num_trees = forest_cpu.shape[0]
        tree_depth = int(np.log2(forest_cpu.shape[1] + 1))
        num_classes = (forest_cpu.shape[2] - 7) // 2

        f = DecisionForest(num_trees, tree_depth, num_classes)
        f.forest_cu.set(forest_cpu)

        return f

    def __init__(self, num_trees, max_depth, num_classes):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.num_classes = num_classes

        self.TOTAL_TREE_NODES, self.MAX_LEAF_NODES, self.TREE_NODE_ELS = DecisionTree.get_config(max_depth, num_classes)

        self.forest_cu = cu_array.GPUArray((self.num_trees, self.TOTAL_TREE_NODES, self.TREE_NODE_ELS), dtype=np.float32)
        self.forest_cu.fill(np.float32(0.))

# comes with gpu memory 
class LayeredDecisionForest():
    @staticmethod
    def load(config_filename, depth_dims, labels_reduce=1):
        cfg = json.loads(open(config_filename).read())
        # models are loaded 1-by-1 from paths with parent directory as a root
        cfg['root'] = os.path.join(*Path(config_filename).parts[0:-1])
        return LayeredDecisionForest(cfg, depth_dims, labels_reduce)

    def __init__(self, cfg, depth_dims, labels_reduce):

        self.eval = DecisionTreeEvaluator()

        self.depth_dims = depth_dims # y,x !!

        self.labels_reduce = labels_reduce
        self.labels_dims = (depth_dims[0] // labels_reduce, depth_dims[1] // labels_reduce)

        self.m = []
        for l in cfg['layers']:
            # model path is relative to config file itself
            m = DecisionForest.load(os.path.join(cfg['root'], l['model']))
            if 'filter_model' in l and 'filter_model_class in l':
                filter_model = l['filter_model']
                filter_model_class = l['filter_model_class']
            else:
                filter_model = None
                filter_model_class = None
            
            self.m.append((m, filter_model, filter_model_class))

        self.num_models = len(self.m)

        self.label_images = [GpuBuffer(self.labels_dims, dtype=np.uint16) for _ in range(self.num_models)]

        self.labels_images_ptrs_cu = GpuBuffer((self.num_models,), dtype=np.int64)
        label_images_ptrs = np.array([i.cu().__cuda_array_interface__['data'][0] for i in self.label_images], dtype=np.int64)
        self.labels_images_ptrs_cu.cu().set(label_images_ptrs)

        # format of conditions list is as follows.
        # (0, PIXEL_ID)
        #   OR
        # (1, NEXT_IMG_CONDITION_OFFSET)

        # example:
        # conditions = [ (0, 1), (0, 2), (1, 3), (0, 3), (0, 4)]
        # i0 i1 | ID
        # 1  -  | 1
        # 2  -  | 2
        # 3  1  | 3
        # 3  2  | 4
        labels_conditions = np.array(cfg['conditions'], dtype=np.int32)
        self.labels_conditions_cu = GpuBuffer(labels_conditions.shape, dtype=np.int32)
        self.labels_conditions_cu.cu().set(labels_conditions)
        # max class id from conditions config
        self.num_layered_classes = max([c[1] for c in filter(lambda c:c[0] == 0, labels_conditions)])

        label_colors = np.array(cfg['label_colors'], dtype=np.uint8)
        assert label_colors.shape == (self.num_layered_classes, 4)
        self.label_colors = GpuBuffer(label_colors.shape, dtype=np.uint8)
        self.label_colors.cu().set(label_colors)


    def run(self, depth_image, labels_image, scale_factor=1.):

        # TODO: assert depth image dims!

        labels_image.cu().fill(MAX_UINT16)

        for i in self.label_images:
            i.cu().fill(MAX_UINT16)

        # first dim: image id. only one image!
        depth_img_dims = (1,) + self.depth_dims
        label_img_dims = (1,) + self.labels_dims

        for i in range(self.num_models):
            m, filter_model, filter_model_class = self.m[i]
            single_labels_image = self.label_images[i]

            self.eval.get_labels_forest(
                m,
                depth_image.cu().reshape(depth_img_dims),
                single_labels_image.cu().reshape(label_img_dims),
                labels_reduce=self.labels_reduce,
                filter_images=self.label_images[filter_model].cu().reshape(label_img_dims) if (filter_model is not None) else None,
                filter_images_class=filter_model_class,
                scale_factor=scale_factor)

        self.eval.make_composite_labels_image(
            self.labels_images_ptrs_cu.cu(),
            self.labels_dims[1],
            self.labels_dims[0],
            self.labels_conditions_cu.cu(),
            labels_image.cu().reshape(label_img_dims))

    # def eval()
class DecisionTreeEvaluator():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('tree_eval')
        self.cu_eval_image = cu_mod.get_function('evaluate_image_using_tree')
        self.cu_eval_image_forest = cu_mod.get_function('evaluate_image_using_forest')
        self._make_composite_labels_image = cu_mod.get_function('make_composite_labels_image')
    
        # real cu ptr, but should be treated as nullptr
        self.cu_empty_ptr = GpuBuffer((1,), dtype=np.int)

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

        
    # TODO: support filter image for single tree forest! or not??
    def get_labels_forest(self, forest, depth_images_in, labels_out, labels_reduce = 1, filter_images=None, filter_images_class=None, scale_factor=1.):
        num_images, dim_y, dim_x = depth_images_in.shape

        assert labels_out.shape == (num_images, dim_y // labels_reduce, dim_x // labels_reduce)

        if filter_images is not None:
            assert filter_images_class is not None
            assert filter_images.shape == labels_out.shape

        num_test_pixels = num_images * (dim_y // labels_reduce) * (dim_x // labels_reduce)

        BLOCK_DIM_X = int(MAX_THREADS_PER_BLOCK // forest.num_trees) 
        grid_dim = (int(num_test_pixels // BLOCK_DIM_X) + 1, 1, 1)
        block_dim = (BLOCK_DIM_X, forest.num_trees, 1)

        f_img = filter_images if filter_images else self.cu_empty_ptr.cu()

        self.cu_eval_image_forest(
            np.int32(forest.num_trees),
            np.int32(num_images),
            np.int32(dim_x),
            np.int32(dim_y),
            np.int32(forest.num_classes),
            np.int32(forest.max_depth),
            np.int32(BLOCK_DIM_X),
            depth_images_in,
            np.int32(filter_images_class if filter_images else -1),
            f_img,
            forest.forest_cu,
            labels_out,
            np.int32(labels_reduce),
            np.float32(scale_factor),
            grid=grid_dim, block=block_dim, shared=(BLOCK_DIM_X * forest.num_classes * 4)) # sizeof(float), right?


    def make_composite_labels_image(self, images, dim_x, dim_y, labels_decision_tree, composite_image):

        # every point..
        grid_dim = ((dim_x // 32) + 1, (dim_y // 32) + 1, 1)
        block_dim = (32, 32, 1)

        self._make_composite_labels_image(
            images,
            np.int32(images.shape[0]),
            np.int32(dim_x),
            np.int32(dim_y),
            labels_decision_tree,
            composite_image,
            grid=grid_dim,
            block=block_dim)


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

def make_random_features(n, arr):
    proposal_features = [(make_random_feature(), make_random_threshold()) for i in range(n)]
    arr[:] = np.array([(p[0][0][0], p[0][0][1], p[0][1][0], p[0][1][1], p[1]) for p in proposal_features ], dtype=np.float32)

class DecisionTreeTrainer():
    def __init__(self, NUM_IMAGES_PER_IMAGE_BLOCK, NUM_PROPOSALS_PER_PROPOSAL_BLOCK):
        # first load kernels..
        cu_mod = py_nvcc_utils.get_module('tree_train')
        self.cu_eval_random_features = cu_mod.get_function('evaluate_random_features')
        self.cu_pick_best_features = cu_mod.get_function('pick_best_features')
        self.cu_copy_pixel_groups = cu_mod.get_function('copy_pixel_groups')
        self.get_active_nodes_next_level = cu_mod.get_function('get_active_nodes_next_level')

        self.NUM_IMAGES_PER_IMAGE_BLOCK = NUM_IMAGES_PER_IMAGE_BLOCK
        self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK = NUM_PROPOSALS_PER_PROPOSAL_BLOCK

    def allocate(self, dataset, NUM_RANDOM_FEATURES, MAX_TREE_DEPTH,):


        # then,
        self.NUM_RANDOM_FEATURES = NUM_RANDOM_FEATURES
        self.MAX_TREE_DEPTH = MAX_TREE_DEPTH

        # make sure image blocks & feature blocks are all uniform size
        assert dataset.num_images % self.NUM_IMAGES_PER_IMAGE_BLOCK == 0
        assert self.NUM_RANDOM_FEATURES % self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK == 0

        self.NUM_IMAGE_BLOCKS = dataset.num_images // self.NUM_IMAGES_PER_IMAGE_BLOCK
        self.NUM_PROPOSAL_BLOCKS = self.NUM_RANDOM_FEATURES // self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK

        _, self.MAX_LEAF_NODES, _ = DecisionTree.get_config(MAX_TREE_DEPTH, dataset.num_classes())

        self.node_counts = cu.pagelocked_zeros((self.MAX_LEAF_NODES, dataset.num_classes()), dtype=np.uint64)
        self.node_counts_cu = cu_array.to_gpu(self.node_counts)
        self.next_node_counts_cu = cu_array.to_gpu(self.node_counts)

        self.active_nodes_cu = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.int32)
        self.next_active_nodes_cu = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.int32)
        self.next_num_active_nodes_cu = cu_array.GPUArray((1), dtype=np.int32)
        self.get_next_num_active_nodes = lambda : self.next_num_active_nodes_cu.get()[0]

        self.best_gain_seen_per_node = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.float32)

        image_block_dims = (self.NUM_IMAGES_PER_IMAGE_BLOCK, dataset.img_dims[1], dataset.img_dims[0])

        self.current_image_block_depth_cpu = cu.pagelocked_zeros(image_block_dims, dtype=np.uint16)
        self.current_image_block_depth = cu_array.GPUArray(image_block_dims, dtype=np.uint16)

        self.current_image_block_labels_cpu = cu.pagelocked_zeros(image_block_dims, dtype=np.uint16)
        self.current_image_block_labels = cu_array.GPUArray(image_block_dims, dtype=np.uint16)

        self.current_image_block_nodes_by_pixel_cpu = cu.pagelocked_zeros(image_block_dims, dtype=np.int32)
        self.current_image_block_nodes_by_pixel = cu_array.GPUArray(image_block_dims, dtype=np.int32)

        self.current_proposals_block_cpu = cu.pagelocked_zeros((self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, 5), dtype=np.float32)
        self.current_proposals_block = cu_array.GPUArray((self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, 5), dtype=np.float32)

        # due to memory limitations, we can only all tree nodes at once up to a certain depth - training more layers deep than this take twice as long per level because nodes are processed in batches
        self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK = 2**17 
        self.current_next_node_counts_by_feature_cu_block = cu_array.GPUArray((self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK, dataset.num_classes()), dtype=np.uint64)

        print('GPU allocations for tree trainer:')
        print('  node_counts:      ', sizeof_fmt(self.node_counts_cu.size * self.node_counts_cu.itemsize))
        print('  next node_counts: ', sizeof_fmt(self.next_node_counts_cu.size * self.next_node_counts_cu.itemsize))
        print('  active_nodes:     ', sizeof_fmt(self.active_nodes_cu.size * self.active_nodes_cu.itemsize))
        print('  nex_active_nodes: ', sizeof_fmt(self.next_active_nodes_cu.size * self.next_active_nodes_cu.itemsize))
        print('  best gain per nod:', sizeof_fmt(self.best_gain_seen_per_node.size * self.best_gain_seen_per_node.itemsize))
        print('  current depth:    ', sizeof_fmt(self.current_image_block_depth.size * self.current_image_block_depth.itemsize))
        print('  current labels:   ', sizeof_fmt(self.current_image_block_labels.size * self.current_image_block_labels.itemsize))
        print('  cur. nodes by pxel:', sizeof_fmt(self.current_image_block_nodes_by_pixel.size * self.current_image_block_nodes_by_pixel.itemsize))
        print('  cur. proposals blk:', sizeof_fmt(self.current_proposals_block.size * self.current_proposals_block.itemsize))
        print('  next nodes counts by feature block:', sizeof_fmt(self.current_next_node_counts_by_feature_cu_block.size * self.current_next_node_counts_by_feature_cu_block.itemsize))

        self.nodes_by_pixel_compressed_blocks = CompressedBlocksDynamic(self.NUM_IMAGE_BLOCKS, self.NUM_IMAGES_PER_IMAGE_BLOCK, (dataset.img_dims[0], dataset.img_dims[1]), np.int32, 'nodes_by_pixel')

    def train(self, dataset, tree):

        tree.tree_out_cu.fill(np.float(0.))

        # start conditions for iteration
        # original distribution of classes as node counts
        self.node_counts[:] = 0

        for img_block_idx in range(self.NUM_IMAGE_BLOCKS):
 
            dataset.get_labels_block_cu(img_block_idx, self.current_image_block_labels)
            self.current_image_block_labels_cpu = self.current_image_block_labels.get()

            un = np.unique(self.current_image_block_labels_cpu, return_counts=True)
            for label_id, count in zip(un[0], un[1]):
                if label_id > 0:
                    self.node_counts[0][label_id] += count

            self.current_image_block_nodes_by_pixel_cpu.fill(-1)
            self.current_image_block_nodes_by_pixel_cpu[np.where(self.current_image_block_labels_cpu > 0)] = 0
            self.current_image_block_nodes_by_pixel.set(self.current_image_block_nodes_by_pixel_cpu)
            self.nodes_by_pixel_compressed_blocks.write_block(img_block_idx, self.current_image_block_nodes_by_pixel)

        self.node_counts_cu.set(self.node_counts)

        # 1 node to start, at idx 0
        self.active_nodes_cu.fill(np.int32(0))
        self.next_num_active_nodes_cu.fill(np.int32(1))

        

        for current_level in range(self.MAX_TREE_DEPTH):

            print('training level', current_level)

            num_active_nodes = self.get_next_num_active_nodes()
            if num_active_nodes == 0:
                break

            self.best_gain_seen_per_node.fill(np.float32(-1.))

            for proposal_block_idx in range(self.NUM_PROPOSAL_BLOCKS):

                make_random_features(self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, self.current_proposals_block_cpu)
                self.current_proposals_block.set(self.current_proposals_block_cpu)

                # if at 0 level, there is (up to) 1 active node,  so next level could have 2 active nodes
                # if at 1 level, there is (up to) 2 active nodes, so next level could have 4 active nodes
                max_active_nodes_next_level = 2**(current_level+1)

                if max_active_nodes_next_level > self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK:
                    assert max_active_nodes_next_level % self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK == 0 # should just be powers of 2..
                    num_node_blocks = max_active_nodes_next_level // self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK
                    node_blocks = [(i * self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK, (i+1) * self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK) for i in range(num_node_blocks)]
                else:
                    node_blocks = [(0, max_active_nodes_next_level)]

                for node_block_start, node_block_end in node_blocks:

                    self.current_next_node_counts_by_feature_cu_block.fill(np.uint64(0))

                    for img_block_idx in range(self.NUM_IMAGE_BLOCKS):

                        dataset.get_labels_block_cu(img_block_idx, self.current_image_block_labels)
                        dataset.get_depth_block_cu(img_block_idx, self.current_image_block_depth)

                        self.nodes_by_pixel_compressed_blocks.get_block(img_block_idx, self.current_image_block_nodes_by_pixel)

                        bdx = MAX_THREADS_PER_BLOCK // self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK
                        img_block_shape = self.current_image_block_depth.shape
                        num_pixels_in_block = (img_block_shape[0] * img_block_shape[1] * img_block_shape[2])

                        gd = ((num_pixels_in_block // bdx) + 1, 1, 1)
                        bd = (bdx, self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, 1)

                        self.cu_eval_random_features(
                            np.int32(self.NUM_IMAGES_PER_IMAGE_BLOCK),
                            np.int32(dataset.img_dims[0]),
                            np.int32(dataset.img_dims[1]),
                            np.int32(self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK),
                            np.int32(dataset.num_classes()),
                            np.int32(self.MAX_TREE_DEPTH),
                            np.int32(self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK),
                            np.int32(node_block_start), # eligible node offset
                            np.int32(node_block_end),
                            self.current_image_block_labels,
                            self.current_image_block_depth,
                            self.current_proposals_block,
                            self.current_image_block_nodes_by_pixel,
                            self.current_next_node_counts_by_feature_cu_block,
                            grid=gd, block=bd)

                    pick_best_grid_dim = (int(num_active_nodes // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
                    pick_best_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

                    self.cu_pick_best_features(
                        np.int32(num_active_nodes),
                        np.int32(self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK),
                        np.int32(self.MAX_TREE_DEPTH),
                        np.int32(self.MAX_NEXT_NODES_TO_COUNT_PER_BLOCK),
                        np.int32(node_block_start),
                        np.int32(node_block_end),
                        np.int32(dataset.num_classes()),
                        np.int32(current_level),
                        self.active_nodes_cu,
                        self.node_counts_cu,
                        self.current_next_node_counts_by_feature_cu_block,
                        self.current_proposals_block,
                        tree.tree_out_cu,
                        self.next_node_counts_cu,
                        self.best_gain_seen_per_node,
                        grid=pick_best_grid_dim, block=pick_best_block_dim)

            self.next_num_active_nodes_cu.fill(np.int32(0))
            self.next_active_nodes_cu.fill(np.int32(0))

            self.get_active_nodes_next_level(
                np.int32(current_level),
                np.int32(self.MAX_TREE_DEPTH),
                np.int32(dataset.num_classes()),
                tree.tree_out_cu,
                self.active_nodes_cu,
                np.int32(num_active_nodes),
                self.next_active_nodes_cu,
                self.next_num_active_nodes_cu,
                grid=(int(num_active_nodes), 1, 1), block=(1, 1, 1))

            if current_level == self.MAX_TREE_DEPTH - 1:
                break

            self.node_counts_cu.set(self.next_node_counts_cu)

            for img_block_idx in range(self.NUM_IMAGE_BLOCKS):

                dataset.get_depth_block_cu(img_block_idx, self.current_image_block_depth)
                self.nodes_by_pixel_compressed_blocks.get_block(img_block_idx, self.current_image_block_nodes_by_pixel)

                copy_grid_dim = (int((self.NUM_IMAGES_PER_IMAGE_BLOCK * dataset.img_dims[0] * dataset.img_dims[1]) // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
                copy_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

                self.cu_copy_pixel_groups(
                    np.int32(self.NUM_IMAGES_PER_IMAGE_BLOCK),
                    np.int32(dataset.img_dims[0]),
                    np.int32(dataset.img_dims[1]),
                    np.int32(current_level),
                    np.int32(self.MAX_TREE_DEPTH),
                    np.int32(dataset.num_classes()),
                    self.current_image_block_depth,
                    self.current_image_block_nodes_by_pixel,
                    tree.tree_out_cu,
                    grid = copy_grid_dim, block=copy_block_dim)

                self.nodes_by_pixel_compressed_blocks.write_block(img_block_idx, self.current_image_block_nodes_by_pixel)

            self.active_nodes_cu.set(self.next_active_nodes_cu)

        # err, return
        # tree buffer should be filled up now!