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
dataset = DecisionTreeDatasetConfig('datagen/sets/flat-hand/', load_train=False, load_test=True)

# evaluating forest!
print('evaluating forest..')
test_output_labels_cu = cu_array.GPUArray(dataset.test.images_shape(), dtype=np.uint16)
test_output_labels_cu.fill(np.uint16(MAX_UINT16))
decision_tree_evaluator.get_labels_forest(forest, dataset.test.depth_cu, test_output_labels_cu)

test_output_labels = test_output_labels_cu.get()
pct_match =  np.sum(test_output_labels == dataset.test.labels) / np.sum(dataset.test.labels > 0)
print('FOREST pct. matching pixels: ', pct_match)

print('saving forest renders..')
test_output_labels_render = dataset.convert_ids_to_colors(test_output_labels)
for i in range(dataset.num_test):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
