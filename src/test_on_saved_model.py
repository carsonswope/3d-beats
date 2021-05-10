import numpy as np

import json

import pycuda.driver as cu
import pycuda.autoinit

from decision_tree import *

np.set_printoptions(suppress=True)

print('loading forest')
forest = DecisionForest.load('models_out/model-filtered.npy')

print('compiling CUDA kernels..')
decision_tree_evaluator = DecisionTreeEvaluator()

print('loading training data')
dataset = DecisionTreeDatasetConfig('datagen/sets/set1/', load_train=False, load_test=True)

dataset_test_depth = np.zeros(dataset.test.images_shape(), dtype=np.uint16)
dataset.test.get_depth(0, dataset_test_depth)
dataset_test_depth_cu = cu_array.to_gpu(dataset_test_depth)

dataset_test_labels = np.zeros(dataset.test.images_shape(), dtype=np.uint16)
dataset.test.get_labels(0, dataset_test_labels)
# dataset_test_labels_cu = cu_array.to_gpu(dataset_test_labels)

# evaluating forest!
print('evaluating forest..')
test_output_labels_cu = cu_array.GPUArray(dataset.test.images_shape(), dtype=np.uint16)
test_output_labels_cu.fill(MAX_UINT16)
decision_tree_evaluator.get_labels_forest(forest, dataset_test_depth_cu, test_output_labels_cu)

test_output_labels = test_output_labels_cu.get()
pct_match =  np.sum(test_output_labels == dataset_test_labels) / np.sum(dataset_test_labels > 0)
print('FOREST pct. matching pixels: ', pct_match)

print('saving forest renders..')
test_output_labels_render = dataset.convert_ids_to_colors(test_output_labels)
for i in range(dataset.num_test):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
