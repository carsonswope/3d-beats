import numpy as np
from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array

from decision_tree import *

import os

np.set_printoptions(suppress=True)
MAX_UINT16 = np.uint16(65535) # max for 16 bit unsigned

IMAGES_PER_TRAINING_BLOCK = 256

print('compiling CUDA kernels..')
decision_tree_trainer = DecisionTreeTrainer(IMAGES_PER_TRAINING_BLOCK)
decision_tree_evaluator = DecisionTreeEvaluator()

print('loading training data')
dataset = DecisionTreeDatasetConfig('datagen/sets/set3/', images_per_training_block=IMAGES_PER_TRAINING_BLOCK)

print('allocating GPU memory')
NUM_RANDOM_FEATURES = 1024
MAX_TREE_DEPTH = 18
tree1 = DecisionTree(MAX_TREE_DEPTH, dataset.num_classes())
decision_tree_trainer.allocate(dataset.train, NUM_RANDOM_FEATURES, tree1.max_depth)

# allocate space for evaluated classes on test data
test_output_labels_cu = cu_array.GPUArray(dataset.test.images_shape(), dtype=np.uint16)

TREES_IN_FOREST = 1
best_trees = [None for t in range(TREES_IN_FOREST)]
tree_cpu = np.zeros((tree1.TOTAL_TREE_NODES, tree1.TREE_NODE_ELS), dtype=np.float32)
forest_cpu = np.zeros((TREES_IN_FOREST, tree1.TOTAL_TREE_NODES, tree1.TREE_NODE_ELS), dtype=np.float32)

test_labels_cpu = cu.pagelocked_zeros(dataset.test.images_shape(), dtype=np.uint16)
dataset.test.get_labels(0, test_labels_cpu)

test_depth_cpu = cu.pagelocked_zeros(dataset.test.images_shape(), dtype=np.uint16)
dataset.test.get_depth(0, test_depth_cpu)
test_depth_cu = cu_array.GPUArray(dataset.test.images_shape(), dtype=np.uint16)
test_depth_cu.set(test_depth_cpu)

for i in range(1):
    print('training tree..')
    decision_tree_trainer.train(dataset.train, tree1)

    print('evaluating..')
    test_output_labels_cu.fill(MAX_UINT16) # doesnt attempt to reclassify when there is no pixel. makes it easier when computing pct match when 0 != 65535
    decision_tree_evaluator.get_labels(tree1, test_depth_cu, test_output_labels_cu)
    test_output_labels = test_output_labels_cu.get()
    # test_output_labels[np.where(test_output_labels == 0)] = 65535 # make sure 0s dont match
    pct_match =  np.sum(test_output_labels == test_labels_cpu) / np.sum(test_labels_cpu > 0)
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
forest1 = DecisionForest(TREES_IN_FOREST, MAX_TREE_DEPTH, dataset.num_classes())
forest1.forest_cu.set(forest_cpu)

# evaluating forest!
print('evaluating forest..')
test_output_labels_cu.fill(np.uint16(MAX_UINT16))
decision_tree_evaluator.get_labels_forest(forest1, test_depth_cu, test_output_labels_cu)

test_output_labels = test_output_labels_cu.get()
pct_match =  np.sum(test_output_labels == test_labels_cpu) / np.sum(test_labels_cpu > 0)
print('FOREST pct. matching pixels: ', pct_match)

print('saving model output!')
np.save('models_out/model-filtered.npy', forest_cpu)

"""
print('saving forest renders..')
test_output_labels_render = dataset.convert_ids_to_colors(test_output_labels)
for i in range(dataset.num_test):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
"""