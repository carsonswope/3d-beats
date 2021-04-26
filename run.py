import numpy as np
from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array
from pycuda.compiler import SourceModule

import py_nvcc_utils

from decision_tree import *

import os

np.set_printoptions(suppress=True)
MAX_THREADS_PER_BLOCK = 1024 # cuda constant..
MAX_UINT16 = np.uint16(65535) # max for 16 bit unsigned

print('compiling CUDA kernels..')
decision_tree_trainer = DecisionTreeTrainer()
decision_tree_evaluator = DecisionTreeEvaluator()

print('loading training data')
decision_tree_dataset_config = DecisionTreeDatasetConfig(
    # image dims
    (424, 240),
    # ID to COLOR mapping
    { 1: (255, 0, 0, 255),
    2: (0, 0, 255, 255),
    3: (0, 255, 0, 255) })

TRAIN = 'datagen/generated/train_3class'
NUM_TRAIN = 128
train_data = DecisionTreeDataset(TRAIN, NUM_TRAIN, decision_tree_dataset_config)

TEST = 'datagen/generated/train_3class'
NUM_TEST = 16
test_data = DecisionTreeDataset(TEST, NUM_TEST, decision_tree_dataset_config)

print('allocating GPU memory')
NUM_RANDOM_FEATURES = 128
MAX_TREE_DEPTH = 14
tree1 = DecisionTree(MAX_TREE_DEPTH, decision_tree_dataset_config.num_classes())
decision_tree_trainer.allocate(train_data, NUM_RANDOM_FEATURES, tree1.max_depth)

# allocate space for evaluated classes on test data
test_output_labels_cu = cu_array.GPUArray(test_data.images_shape(), dtype=np.uint16)

TREES_IN_FOREST = 4
best_trees = [None for t in range(TREES_IN_FOREST)]
tree_cpu = np.zeros((tree1.TOTAL_TREE_NODES, tree1.TREE_NODE_ELS), dtype=np.float32)
forest_cpu = np.zeros((TREES_IN_FOREST, tree1.TOTAL_TREE_NODES, tree1.TREE_NODE_ELS), dtype=np.float32)

for i in range(10):
    print('training tree..')
    decision_tree_trainer.train(train_data, tree1)

    print('evaluating..')
    test_output_labels_cu.fill(np.uint16(MAX_UINT16)) # doesnt attempt to reclassify when there is no pixel. makes it easier when computing pct match when 0 != 65535
    decision_tree_evaluator.get_labels(tree1, test_data.depth_cu, test_output_labels_cu)
    test_output_labels = test_output_labels_cu.get()
    # test_output_labels[np.where(test_output_labels == 0)] = 65535 # make sure 0s dont match
    pct_match =  np.sum(test_output_labels == test_data.labels) / np.sum(test_data.labels > 0)
    print('pct. matching pixels: ', pct_match)

    copy_idx = -1
    if None in best_trees:
        copy_idx = best_trees.index(None)
    else:
        worst_pct_match = min(best_trees)
        if pct_match > worst_pct_match:
            copy_idx = best_trees.index(worst_pct_match)

    if copy_idx > -1:
        print('accepted tree at slot ', copy_idx)
        best_trees[copy_idx] = pct_match
        tree1.tree_out_cu.get(tree_cpu)
        forest_cpu[copy_idx] = np.copy(tree_cpu)

print('completing forest..')
forest1 = DecisionForest(TREES_IN_FOREST, MAX_TREE_DEPTH, decision_tree_dataset_config.num_classes())
forest1.forest_cu.set(forest_cpu)

# evaluating forest!
print('evaluating forest..')
test_output_labels_cu.fill(np.uint16(MAX_UINT16))
decision_tree_evaluator.get_labels_forest(forest1, test_data.depth_cu, test_output_labels_cu)

test_output_labels = test_output_labels_cu.get()
pct_match =  np.sum(test_output_labels == test_data.labels) / np.sum(test_data.labels > 0)
print('FOREST pct. matching pixels: ', pct_match)

print('saving forest renders..')
test_output_labels_render = decision_tree_dataset_config.convert_ids_to_colors(test_output_labels)
for i in range(NUM_TEST):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
