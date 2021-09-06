import numpy as np

import argparse

from engine.window import AppBase, run_app

import glfw
from decision_tree import *

np.set_printoptions(suppress=True)

class EvalModel(AppBase):
    def __init__(self):
        super().__init__(title="Eval Model", width=5, height=5)
        
    def splash(self):
        pass

    def tick(self, t):
        main()
        glfw.set_window_should_close(self.window, True)

def main():

    parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
    parser.add_argument('-m', '--model', nargs='?', required=True, type=str, help='Path to .npy model input file')
    parser.add_argument('-d', '--data', nargs='?', required=True, type=str, help='Directory holding data')
    parser.add_argument('-o', '--out', nargs='?', required=True, type=str, help='Directory to save output renderings')
    parser.add_argument('--test', nargs='?', required=True, type=int, help='Num images to evaluate')
    args = parser.parse_args()

    MODEL_PATH = args.model
    DATASET_PATH = args.data
    NUM_IMAGES = args.test
    OUT_PATH = args.out

    print('loading forest')
    forest = DecisionForest.load(MODEL_PATH)

    print('compiling CUDA kernels..')
    decision_tree_evaluator = DecisionTreeEvaluator()

    print('loading data')
    dataset = DecisionTreeDatasetConfig(DATASET_PATH, num_images=NUM_IMAGES, imgs_name='test', randomize=True)

    dataset_test_depth = cu_array.GPUArray(dataset.images_shape(), dtype=np.uint16)
    dataset.get_depth_block_cu(0, dataset_test_depth)

    dataset_test_labels = cu_array.GPUArray(dataset.images_shape(), dtype=np.uint16)
    dataset.get_labels_block_cu(0, dataset_test_labels)
    dataset_test_labels_cpu = dataset_test_labels.get()

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
        im.save(f'{OUT_PATH}/eval_labels_{str(i).zfill(8)}.png')

if __name__ == '__main__':
    run_app(EvalModel)
