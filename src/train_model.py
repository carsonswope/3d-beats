import numpy as np
from PIL import Image

import pycuda.driver as cu
import pycuda.autoinit
import pycuda.gpuarray as cu_array

from decision_tree import *

from util import MAX_UINT16

import argparse

np.set_printoptions(suppress=True)

def main():

    parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
    parser.add_argument('--train', nargs='?', required=True, type=int, help='Num training images')
    parser.add_argument('--train_block', nargs='?', required=False, type=int, help='Num training images in a block. Defaults to num training images')
    parser.add_argument('--test', nargs='?', required=True, type=int, help='Num test images')
    parser.add_argument('--proposals', nargs='?', required=True, type=int, help='Num proposals tested per node')
    parser.add_argument('--proposals_block', nargs='?', required=True, type=int, help='Num proposals tested per proposal block')
    parser.add_argument('--out_trees', nargs='?', required=True, type=int, help='Num trees in final forest')
    parser.add_argument('--trees_to_try', nargs='?', required=False, type=int, help='Num candidate trees generated for forest')
    parser.add_argument('--depth', nargs='?', required=True, type=int, help='Max depth for a tree in the forest')

    parser.add_argument('-o', '--out', nargs='?', required=True, type=str, help='Where to save the output model')
    parser.add_argument('-d', '--data', nargs='?', required=True, type=str, help='Directory containing the training data')

    args = parser.parse_args()

    NUM_TRAIN_IMAGES = args.train
    IMAGES_PER_TRAINING_BLOCK = args.train_block # if none, this will default to NUM_TRAIN_IMAGES
    NUM_TEST_IMAGES = args.test

    NUM_RANDOM_FEATURES = args.proposals
    PROPOSALS_PER_PROPOSAL_BLOCK = args.proposals_block

    MODEL_OUT_NAME = args.out
    DATASET_PATH = args.data

    TREES_IN_FOREST = args.out_trees
    TREES_TO_TRAIN = args.trees_to_try or TREES_IN_FOREST

    MAX_TREE_DEPTH = args.depth

    print('compiling CUDA kernels..')
    decision_tree_trainer = DecisionTreeTrainer(IMAGES_PER_TRAINING_BLOCK, PROPOSALS_PER_PROPOSAL_BLOCK)
    decision_tree_evaluator = DecisionTreeEvaluator()

    print('loading training data')
    train_data, test_data = DecisionTreeDatasetConfig.multiple(DATASET_PATH, [
        # train data
        (NUM_TRAIN_IMAGES, IMAGES_PER_TRAINING_BLOCK, 'train'),
        # test data. None = just one block!
        (NUM_TEST_IMAGES, None, 'test')])

    print('allocating GPU memory')
    tree1 = DecisionTree(MAX_TREE_DEPTH, train_data.num_classes())
    decision_tree_trainer.allocate(train_data, NUM_RANDOM_FEATURES, tree1.max_depth)

    # allocate space for evaluated classes on test data
    test_output_labels_cu = cu_array.GPUArray(test_data.images_shape(), dtype=np.uint16)

    best_trees = [None for t in range(TREES_IN_FOREST)]
    tree_cpu = np.zeros((tree1.TOTAL_TREE_NODES, tree1.TREE_NODE_ELS), dtype=np.float32)
    forest_cpu = np.zeros((TREES_IN_FOREST, tree1.TOTAL_TREE_NODES, tree1.TREE_NODE_ELS), dtype=np.float32)

    test_labels_cu = cu_array.GPUArray(test_data.images_shape(), dtype=np.uint16)
    test_data.get_labels_block_cu(0, test_labels_cu)
    test_labels_cpu = test_labels_cu.get()

    test_depth_cu = cu_array.GPUArray(test_data.images_shape(), dtype=np.uint16)
    test_data.get_depth_block_cu(0, test_depth_cu)

    for i in range(TREES_TO_TRAIN):
        print('training tree..')
        decision_tree_trainer.train(train_data, tree1)

        print('evaluating..')
        test_output_labels_cu.fill(MAX_UINT16) # doesnt attempt to reclassify when there is no pixel. makes it easier when computing pct match when 0 != 65535
        decision_tree_evaluator.get_labels(tree1, test_depth_cu, test_output_labels_cu)
        test_output_labels = test_output_labels_cu.get()
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
    forest1 = DecisionForest(TREES_IN_FOREST, MAX_TREE_DEPTH, test_data.num_classes())
    forest1.forest_cu.set(forest_cpu)

    # evaluating forest!
    print('evaluating forest..')
    test_output_labels_cu.fill(np.uint16(MAX_UINT16))
    decision_tree_evaluator.get_labels_forest(forest1, test_depth_cu, test_output_labels_cu)

    test_output_labels = test_output_labels_cu.get()
    pct_match =  np.sum(test_output_labels == test_labels_cpu) / np.sum(test_labels_cpu > 0)
    print('FOREST pct. matching pixels: ', pct_match)

    print('saving model output!')
    np.save(MODEL_OUT_NAME, forest_cpu)

    """
    print('saving forest renders..')
    test_output_labels_render = dataset.convert_ids_to_colors(test_output_labels)
    for i in range(dataset.num_test):
        out_labels_img = test_output_labels_render[i]
        im = Image.fromarray(out_labels_img)
        im.save('evals/eval_labels_' + str(i).zfill(8) + '.png')
    """

if __name__ == '__main__':
    main()