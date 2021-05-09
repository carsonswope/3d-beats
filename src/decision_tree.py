import cuda.py_nvcc_utils as py_nvcc_utils
import json
import numpy as np
from PIL import Image
import imageio
from io import BytesIO

import pycuda.gpuarray as cu_array
import pycuda.driver as cu

import nvcomp

MAX_THREADS_PER_BLOCK = 1024 # cuda constant..

MAX_UINT16 = np.uint16(65535) # max for 16 bit unsigned

# configuration for a decision tree dataset.
# image size
# mapping from label colors to int IDs
class DecisionTreeDatasetConfig():

    def __init__(self, dataset_dir, load_train=True, load_test=True, images_per_training_block=0):

        self.dataset_dir = dataset_dir
        cfg = json.loads(open(dataset_dir + 'config.json').read())

        self.img_dims = tuple(cfg['img_dims'])
        self.id_to_color = {0: np.array([0, 0, 0, 0], dtype=np.uint8)}
        for i,c in cfg['id_to_color'].items():
            self.id_to_color[int(i)] = np.array(c, dtype=np.uint8)
        
        self.num_train = cfg['num_train']
        self.num_test = cfg['num_test']

        arr_size = lambda x: x.size * x.itemsize

        if load_train:
            self.train = DecisionTreeDataset(dataset_dir + 'train', self.num_train, self, load_compressed_cu=images_per_training_block > 0, load_compressed_cu_images_per_block=images_per_training_block, load_cu=False)
            # print(self.num_train, 'training images loaded. CPU Pagelocked memory used: ', '{:,}'.format(arr_size(self.train.depth) + arr_size(self.train.labels)))
            # print(self.num_train, 'training images loaded. GPU memory used: ', '{:,}'.format(arr_size(self.train.depth_cu) + arr_size(self.train.labels_cu)))
        else:
            self.train = None

        if load_test:
            # load CU for test..
            self.test = DecisionTreeDataset(dataset_dir + 'test', self.num_test, self, load_cu=True)
            # print(self.num_test, 'test images loaded.   GPU memory used: ', '{:,}'.format(arr_size(self.test.depth_cu) + arr_size(self.test.labels_cu)))
        else:
            self.test = None

    
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
    def __init__(self, path_root, num_images, config, load_cu=True, load_compressed_cu=False, load_compressed_cu_images_per_block=0):
        self.num_images = num_images
        self.config = config
        # raw bytes in PNG format (compressed!)
        self.all_depth_pngs, self.all_labels_pngs = self.load_data(path_root, num_images)

        if load_compressed_cu:
            assert self.num_images % load_compressed_cu_images_per_block == 0
            NUM_IMAGE_BLOCKS = self.num_images // load_compressed_cu_images_per_block
            # IMAGES_PER_IMAGE_BLOCK = 

            compressor = nvcomp.CascadedCompressor('u2', 2, 1, True)

            depth_block_np = np.zeros((load_compressed_cu_images_per_block, self.config.img_dims[1], self.config.img_dims[0]), np.uint16)
            labels_block_np = np.zeros((load_compressed_cu_images_per_block, self.config.img_dims[1], self.config.img_dims[0]), np.uint16)
            uncompressed_block_cu = cu_array.to_gpu(depth_block_np)
            block_size = depth_block_np.size * depth_block_np.itemsize
            # for 
            compressor_temp_size, compressor_output_max_size = compressor.configure(block_size)

            compressor_temp_cu = cu_array.GPUArray((compressor_temp_size,), dtype=np.uint8)
            compressor_output_cu = cu_array.GPUArray((compressor_output_max_size,), dtype=np.uint8)
            compressor_output_size = cu.pagelocked_zeros((1,), np.int64)

            # blocks will all 
            # length_block_indices = [] # (start_idx, length)
            # depth_block_indices = []

            compressed_depth_blocks = []
            compressed_labels_blocks = []

            for i in range(NUM_IMAGE_BLOCKS):
                for j in range(load_compressed_cu_images_per_block):
                    img_idx = (i * load_compressed_cu_images_per_block) + j
                    depth_png = self.all_depth_pngs[img_idx]
                    labels_png = self.all_labels_pngs[img_idx]
                    depth_np = imageio.imread(BytesIO(depth_png)).view(np.uint16)
                    labels_np = imageio.imread(BytesIO(labels_png)).view(np.uint16)
                    depth_block_np[j] = depth_np
                    labels_block_np[j] = labels_np

                # first compress depth
                uncompressed_block_cu.set(depth_block_np)
                compressor.compress(
                    uncompressed_block_cu.ptr,
                    block_size,
                    compressor_temp_cu.ptr,
                    compressor_temp_size,
                    compressor_output_cu.ptr,
                    compressor_output_size.__array_interface__['data'][0])

                cu.Context.synchronize()
                compressed_depth_size = compressor_output_size[0]
                compressed_depth_blocks.append(compressor_output_cu[0:compressed_depth_size].get())

                # then compress labels
                uncompressed_block_cu.set(labels_block_np)
                compressor.compress(
                    uncompressed_block_cu.ptr,
                    block_size,
                    compressor_temp_cu.ptr,
                    compressor_temp_size,
                    compressor_output_cu.ptr,
                    compressor_output_size.__array_interface__['data'][0])
                
                cu.Context.synchronize()
                compressed_labels_size = compressor_output_size[0]
                compressed_labels_blocks.append(compressor_output_cu[0:compressed_labels_size].get())
            
            all_compressed_depth_blocks_size = sum([b.shape[0] for b in compressed_depth_blocks])
            all_compressed_labels_blocks_size = sum([b.shape[0] for b in compressed_labels_blocks])

            self.all_compressed_depth_blocks_cu = cu_array.GPUArray((all_compressed_depth_blocks_size,), dtype=np.uint8)
            self.all_compressed_labels_blocks_cu = cu_array.GPUArray((all_compressed_labels_blocks_size,), dtype=np.uint8)

            self.depth_blocks_idxes = [] # (start_idx, length)
            self.labels_blocks_idxes = []

            depth_idx = 0
            labels_idx = 0

            all_compressed_depth_blocks_cpu = np.zeros((all_compressed_depth_blocks_size,), np.uint8)
            all_compressed_labels_blocks_cpu = np.zeros((all_compressed_labels_blocks_size,), np.uint8)

            for i in range(NUM_IMAGE_BLOCKS):
                compressed_depth_block_size = compressed_depth_blocks[i].shape[0]
                compressed_labels_block_size = compressed_labels_blocks[i].shape[0]
                self.depth_blocks_idxes.append((depth_idx, compressed_depth_block_size))
                self.labels_blocks_idxes.append((labels_idx, compressed_labels_block_size))
                all_compressed_depth_blocks_cpu[depth_idx:depth_idx+compressed_depth_block_size] = compressed_depth_blocks[i]
                all_compressed_labels_blocks_cpu[labels_idx:labels_idx+compressed_labels_block_size] = compressed_labels_blocks[i]
                depth_idx += compressed_depth_block_size
                labels_idx += compressed_labels_block_size

            self.all_compressed_depth_blocks_cu.set(all_compressed_depth_blocks_cpu)
            self.all_compressed_labels_blocks_cu.set(all_compressed_labels_blocks_cpu)

            # compression complete. release memory used for compression
            del uncompressed_block_cu
            del compressor_output_cu
            del compressor_temp_cu

            self.decompressor = nvcomp.CascadedDecompressor()
            self.decompressor_temp_size = cu.pagelocked_zeros((1,), np.int64)
            self.decompressor_output_size = cu.pagelocked_zeros((1,), np.int64)

            max_decompressor_temp_size = 0

            for i in range(NUM_IMAGE_BLOCKS):

                # first depth
                depth_i, depth_block_compressed_size = self.depth_blocks_idxes[i]
                self.decompressor.configure(
                    self.all_compressed_depth_blocks_cu[depth_i].ptr,
                    depth_block_compressed_size,
                    self.decompressor_temp_size.__array_interface__['data'][0],
                    self.decompressor_output_size.__array_interface__['data'][0])
                cu.Context.synchronize()
                print('temp size: ', self.decompressor_temp_size[0])
                assert self.decompressor_output_size[0] == block_size
                max_decompressor_temp_size = max(max_decompressor_temp_size, self.decompressor_temp_size[0])

                # then labels
                labels_i, labels_block_compressed_size = self.labels_blocks_idxes[i]
                self.decompressor.configure(
                    self.all_compressed_labels_blocks_cu[labels_i].ptr,
                    labels_block_compressed_size,
                    self.decompressor_temp_size.__array_interface__['data'][0],
                    self.decompressor_output_size.__array_interface__['data'][0])
                cu.Context.synchronize()
                print('temp size: ', self.decompressor_temp_size[0])
                assert self.decompressor_output_size[0] == block_size
                max_decompressor_temp_size = max(max_decompressor_temp_size, self.decompressor_temp_size[0])

            self.decompressor_temp_cu = cu_array.GPUArray((max_decompressor_temp_size,), dtype=np.uint8)


    def get_depth_block_cu(self, block_num, arr_out):

        depth_i, depth_block_compressed_size = self.depth_blocks_idxes[block_num]

        # allocations already made, still have to configure though..
        self.decompressor.configure(
            self.all_compressed_depth_blocks_cu[depth_i].ptr,
            depth_block_compressed_size,
            self.decompressor_temp_size.__array_interface__['data'][0],
            self.decompressor_output_size.__array_interface__['data'][0])

        self.decompressor.decompress(
            self.all_compressed_depth_blocks_cu[depth_i].ptr,
            depth_block_compressed_size,
            self.decompressor_temp_cu.ptr,
            self.decompressor_temp_cu.size, # itemsize==1, uint8
            arr_out.ptr,
            arr_out.size * arr_out.itemsize)

    def get_labels_block_cu(self, block_num, arr_out):

        labels_i, labels_block_compressed_size = self.labels_blocks_idxes[block_num]

        # allocations already made, still have to configure though..
        self.decompressor.configure(
            self.all_compressed_labels_blocks_cu[labels_i].ptr,
            labels_block_compressed_size,
            self.decompressor_temp_size.__array_interface__['data'][0],
            self.decompressor_output_size.__array_interface__['data'][0])

        self.decompressor.decompress(
            self.all_compressed_labels_blocks_cu[labels_i].ptr,
            labels_block_compressed_size,
            self.decompressor_temp_cu.ptr,
            self.decompressor_temp_cu.size, # itemsize==1, uint8
            arr_out.ptr,
            arr_out.size * arr_out.itemsize)

    # fill up a (np.array(num_images, dimy, dimx)) with labels via decoding the PNG!
    def get_labels(self, start_idx, a):
        num_images = a.shape[0]
        assert a.shape[1] == self.config.img_dims[1]
        assert a.shape[2] == self.config.img_dims[0]

        for i in range(num_images):
            img_pixels = imageio.imread(BytesIO(self.all_labels_pngs[start_idx + i]))
            # assert np.all(img_pixels == self.labels[start_idx + i])
            assert img_pixels.shape[0] == a.shape[1]
            assert img_pixels.shape[1] == a.shape[2]
            a[i] = img_pixels

    # fill up a (np.array(num_images, dimy, dimx)) with depth via decoding the PNG!
    def get_depth(self, start_idx, a):
        num_images = a.shape[0]
        assert a.shape[1] == self.config.img_dims[1]
        assert a.shape[2] == self.config.img_dims[0]

        for i in range(num_images):
            img_pixels = imageio.imread(BytesIO(self.all_depth_pngs[start_idx + i]))
            # assert np.all(img_pixels == self.depth[start_idx + i])
            assert img_pixels.shape[0] == a.shape[1]
            assert img_pixels.shape[1] == a.shape[2]
            a[i] = img_pixels

    # s: TEST or TRAIN (set)
    # t: DEPTH or LABELS (type)
    @staticmethod
    def __data_path(s, t, i):
        return s + '' + str(i).zfill(8) + '_' + t + '.png'

    def load_data(self, s, num):
        # store image data as raw PNG - takes up much less memory!
        all_depth_pngs = []
        all_labels_pngs = []

        DEPTH = 'depth'
        LABELS = 'labels'

        for i in range(num):
            
            with open(DecisionTreeDataset.__data_path(s, DEPTH, i), 'rb') as f_png:
                all_depth_pngs.append(f_png.read())
            with open(DecisionTreeDataset.__data_path(s, LABELS, i), 'rb') as f_png:
                all_labels_pngs.append(f_png.read())

        return all_depth_pngs, all_labels_pngs

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

    # def eval()
class DecisionTreeEvaluator():
    def __init__(self):
        cu_mod = py_nvcc_utils.get_module('src/cuda/tree_eval.cu')
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

# convert np array to png bytes!
def get_png_bytes(a):
    assert a.dtype == np.int32
    _o = BytesIO()
    _i = Image.fromarray(a.view(np.uint8))
    _i.save(_o, format='PNG')
    return _o.getvalue()

# convert multiple images to an array of bytes!
def get_all_png_bytes(a):
    assert len(a.shape) == 3
    return [get_png_bytes(_a) for _a in a]

def decode_all_pngs(pngs, out_a):
    for i in range(len(pngs)):
        img_pixels = imageio.imread(BytesIO(pngs[i]))
        out_a[i] = img_pixels.view(np.int32)


class DecisionTreeTrainer():
    def __init__(self, NUM_IMAGES_PER_IMAGE_BLOCK):
        # first load kernels..
        cu_mod = py_nvcc_utils.get_module('src/cuda/tree_train.cu')
        self.cu_eval_random_features = cu_mod.get_function('evaluate_random_features')
        self.cu_pick_best_features = cu_mod.get_function('pick_best_features')
        self.cu_copy_pixel_groups = cu_mod.get_function('copy_pixel_groups')
        self.get_active_nodes_next_level = cu_mod.get_function('get_active_nodes_next_level')

        self.NUM_IMAGES_PER_IMAGE_BLOCK = NUM_IMAGES_PER_IMAGE_BLOCK
        self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK = 128

    def allocate(self, dataset, NUM_RANDOM_FEATURES, MAX_TREE_DEPTH,):


        # then,
        self.NUM_RANDOM_FEATURES = NUM_RANDOM_FEATURES
        self.MAX_TREE_DEPTH = MAX_TREE_DEPTH

        # make sure image blocks & feature blocks are all uniform size
        assert dataset.num_images % self.NUM_IMAGES_PER_IMAGE_BLOCK == 0
        assert self.NUM_RANDOM_FEATURES % self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK == 0

        _, self.MAX_LEAF_NODES, _ = DecisionTree.get_config(MAX_TREE_DEPTH, dataset.config.num_classes())

        self.node_counts = cu.pagelocked_zeros((self.MAX_LEAF_NODES, dataset.config.num_classes()), dtype=np.uint64)
        self.node_counts_cu = cu_array.to_gpu(self.node_counts)
        self.next_node_counts_cu = cu_array.to_gpu(self.node_counts)

        self.active_nodes_cu = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.int32)
        self.next_active_nodes_cu = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.int32)
        self.next_num_active_nodes_cu = cu_array.GPUArray((1), dtype=np.int32)
        self.get_next_num_active_nodes = lambda : self.next_num_active_nodes_cu.get()[0]

        self.best_gain_seen_per_node = cu_array.GPUArray((self.MAX_LEAF_NODES), dtype=np.float32)

        image_block_dims = (self.NUM_IMAGES_PER_IMAGE_BLOCK, dataset.config.img_dims[1], dataset.config.img_dims[0])

        self.current_image_block_depth_cpu = cu.pagelocked_zeros(image_block_dims, dtype=np.uint16)
        self.current_image_block_depth = cu_array.GPUArray(image_block_dims, dtype=np.uint16)

        self.current_image_block_labels_cpu = cu.pagelocked_zeros(image_block_dims, dtype=np.uint16)
        self.current_image_block_labels = cu_array.GPUArray(image_block_dims, dtype=np.uint16)

        # compressed format..
        self.nodes_by_pixel_png = []
        self.current_image_block_nodes_by_pixel_cpu = cu.pagelocked_zeros(image_block_dims, dtype=np.int32)
        self.current_image_block_nodes_by_pixel = cu_array.GPUArray(image_block_dims, dtype=np.int32)

        self.current_proposals_block = cu_array.GPUArray((self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, 5), dtype=np.float32)
        self.current_next_node_counts_by_feature_cu_block = cu_array.GPUArray((self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, self.MAX_LEAF_NODES, dataset.config.num_classes()), dtype=np.uint64)

    def train(self, dataset, tree):

        NUM_IMAGE_BLOCKS = dataset.num_images // self.NUM_IMAGES_PER_IMAGE_BLOCK
        NUM_PROPOSAL_BLOCKS = self.NUM_RANDOM_FEATURES // self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK

        tree.tree_out_cu.fill(np.float(0.))

        # start conditions for iteration
        # original distribution of classes as node counts
        self.node_counts[:] = 0
        self.nodes_by_pixel_png = []

        # current_image_block_labels_cu__ = cu_arra

        for ii in range(NUM_IMAGE_BLOCKS):
            i_start = ii * self.NUM_IMAGES_PER_IMAGE_BLOCK
            i_end = (ii + 1) * self.NUM_IMAGES_PER_IMAGE_BLOCK
            # dataset.get_labels(i_start, self.current_image_block_labels_cpu)

            dataset.get_labels_block_cu(ii, self.current_image_block_labels)
            self.current_image_block_labels_cpu = self.current_image_block_labels.get()

            un = np.unique(self.current_image_block_labels_cpu, return_counts=True)
            for label_id, count in zip(un[0], un[1]):
                if label_id > 0:
                    self.node_counts[0][label_id] += count

            block_nodes_by_pixel = np.ones(self.current_image_block_labels_cpu.shape, dtype=np.int32) * -1
            block_nodes_by_pixel[np.where(self.current_image_block_labels_cpu > 0)] = 0
            png_bytes = get_all_png_bytes(block_nodes_by_pixel)

            self.nodes_by_pixel_png += png_bytes

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

            for ff in range(NUM_PROPOSAL_BLOCKS):

                print('  proposal block: ', ff)
                
                f_start = ff * self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK
                f_end = (ff + 1) * self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK

                self.current_proposals_block.set(make_random_features(self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK))
                self.current_next_node_counts_by_feature_cu_block.fill(np.uint64(0))

                for ii in range(NUM_IMAGE_BLOCKS):

                    print('    image block', ii)

                    i_start = ii * self.NUM_IMAGES_PER_IMAGE_BLOCK
                    i_end = (ii + 1) * self.NUM_IMAGES_PER_IMAGE_BLOCK

                    dataset.get_labels_block_cu(ii, self.current_image_block_labels)
                    dataset.get_depth_block_cu(ii, self.current_image_block_depth)

                    decode_all_pngs(self.nodes_by_pixel_png[i_start : i_end], self.current_image_block_nodes_by_pixel_cpu)
                    self.current_image_block_nodes_by_pixel.set(self.current_image_block_nodes_by_pixel_cpu)

                    bdx = MAX_THREADS_PER_BLOCK // self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK
                    img_block_shape = self.current_image_block_depth.shape
                    num_pixels_in_block = (img_block_shape[0] * img_block_shape[1] * img_block_shape[2])

                    gd = ((num_pixels_in_block // bdx) + 1, 1, 1)
                    bd = (bdx, self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK, 1)

                    self.cu_eval_random_features(
                        np.int32(self.NUM_IMAGES_PER_IMAGE_BLOCK),
                        np.int32(dataset.config.img_dims[0]),
                        np.int32(dataset.config.img_dims[1]),
                        np.int32(self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK),
                        np.int32(dataset.config.num_classes()),
                        np.int32(self.MAX_TREE_DEPTH),
                        self.current_image_block_labels,
                        self.current_image_block_depth,
                        self.current_proposals_block,
                        self.current_image_block_nodes_by_pixel,
                        self.current_next_node_counts_by_feature_cu_block,
                        grid=gd, block=bd)

                print('    pick best')

                pick_best_grid_dim = (int(num_active_nodes // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
                pick_best_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

                self.cu_pick_best_features(
                    np.int32(num_active_nodes),
                    np.int32(self.NUM_PROPOSALS_PER_PROPOSAL_BLOCK),
                    np.int32(self.MAX_TREE_DEPTH),
                    np.int32(dataset.config.num_classes()),
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

            print('  gen active nodes next level')

            self.get_active_nodes_next_level(
                np.int32(current_level),
                np.int32(self.MAX_TREE_DEPTH),
                np.int32(dataset.config.num_classes()),
                tree.tree_out_cu,
                self.active_nodes_cu,
                np.int32(num_active_nodes),
                self.next_active_nodes_cu,
                self.next_num_active_nodes_cu,
                grid=(int(num_active_nodes), 1, 1), block=(1, 1, 1))

            if current_level == self.MAX_TREE_DEPTH - 1:
                break

            self.node_counts_cu.set(self.next_node_counts_cu)

            for ii in range(NUM_IMAGE_BLOCKS):

                print('  eval pixels', ii)

                i_start = ii * self.NUM_IMAGES_PER_IMAGE_BLOCK
                i_end = (ii+1) * self.NUM_IMAGES_PER_IMAGE_BLOCK

                # dataset.get_depth(i_start, self.current_image_block_depth_cpu)
                # self.current_image_block_depth.set(self.current_image_block_depth_cpu)

                dataset.get_depth_block_cu(ii, self.current_image_block_depth)

                decode_all_pngs(self.nodes_by_pixel_png[i_start : i_end], self.current_image_block_nodes_by_pixel_cpu)
                self.current_image_block_nodes_by_pixel.set(self.current_image_block_nodes_by_pixel_cpu)
                
                copy_grid_dim = (int((self.NUM_IMAGES_PER_IMAGE_BLOCK * dataset.config.img_dims[0] * dataset.config.img_dims[1]) // MAX_THREADS_PER_BLOCK) + 1, 1, 1)
                copy_block_dim = (MAX_THREADS_PER_BLOCK, 1, 1)

                self.cu_copy_pixel_groups(
                    np.int32(self.NUM_IMAGES_PER_IMAGE_BLOCK),
                    np.int32(dataset.config.img_dims[0]),
                    np.int32(dataset.config.img_dims[1]),
                    np.int32(current_level),
                    np.int32(self.MAX_TREE_DEPTH),
                    np.int32(dataset.config.num_classes()),
                    self.current_image_block_depth,
                    self.current_image_block_nodes_by_pixel,
                    tree.tree_out_cu,
                    grid = copy_grid_dim, block=copy_block_dim)

                self.current_image_block_nodes_by_pixel.get(self.current_image_block_nodes_by_pixel_cpu)
                self.nodes_by_pixel_png[i_start : i_end] = get_all_png_bytes(self.current_image_block_nodes_by_pixel_cpu)

            self.active_nodes_cu.set(self.next_active_nodes_cu)

        # err, return
        # tree buffer should be filled up now!