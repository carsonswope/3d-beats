import numpy as np

import json

import pycuda.driver as cu
import pycuda.autoinit

from decision_tree import *

np.set_printoptions(suppress=True)

MODEL_OUT_NAME = 'models_out/live10.npy'
DATASET_PATH ='datagen/sets/live1/data9/'

print('loading forest')
forest = DecisionForest.load(MODEL_OUT_NAME)

print('compiling CUDA kernels..')
decision_tree_evaluator = DecisionTreeEvaluator()

print('loading rwar data')
dataset = DecisionTreeDatasetConfig(DATASET_PATH, num_images=16, imgs_name='test', randomize=True)

dataset_test_depth = cu_array.GPUArray(dataset.images_shape(), dtype=np.uint16)
dataset.get_depth_block_cu(0, dataset_test_depth)

dataset_test_labels = cu_array.GPUArray(dataset.images_shape(), dtype=np.uint16)
dataset.get_labels_block_cu(0, dataset_test_labels)
dataset_test_labels_cpu = dataset_test_labels.get()

# evaluating forest!
print('evaluating forest..')
test_output_labels_cu = cu_array.GPUArray(dataset.images_shape(), dtype=np.uint16)
test_output_labels_cu.fill(MAX_UINT16)
decision_tree_evaluator.get_labels_forest(forest, dataset_test_depth, test_output_labels_cu)

test_output_labels = test_output_labels_cu.get()
pct_match =  np.sum(test_output_labels == dataset_test_labels_cpu) / np.sum(dataset_test_labels_cpu > 0)
print('FOREST pct. matching pixels: ', pct_match)

print('saving forest renders..')
test_output_labels_render = dataset.convert_ids_to_colors(test_output_labels)
for i in range(dataset.num_images):
    out_labels_img = test_output_labels_render[i]
    im = Image.fromarray(out_labels_img)
    im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
