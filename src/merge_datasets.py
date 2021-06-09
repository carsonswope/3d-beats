import numpy as np

import json
from PIL import Image

import argparse
import shutil

from decision_tree import *
np.set_printoptions(suppress=True)

def main():
    parser = argparse.ArgumentParser(description='Combine multiple datasets')
    parser.add_argument('-m', '--mapping', nargs='?', required=True, type=str, help='Path to JSON file describing merge')
    parser.add_argument('-o', '--out', nargs='?', required=True, type=str, help='Out directory for merged datasets')
    args = parser.parse_args()

    MAPPING_PATH = args.mapping
    OUT_PATH = args.out

    mapping = json.loads(open(MAPPING_PATH).read())

    out_color_mapping = {m:np.array(o['new_color'], dtype=np.uint8) for m, o in mapping['labels'].items()}

    configs = {m[1]: (DecisionTreeDatasetConfig(m[0]), m[0]) for m in mapping['datasets']}

    img_dims = set([c[0].img_dims for c in configs.values()])
    assert len(img_dims) == 1
    img_dims = img_dims.pop()

    num_colors = set([len(c[0].id_to_color) for c in configs.values()])
    assert len(num_colors) == 1
    num_colors = num_colors.pop()

    j = 0

    for c_id, c in configs.items():

        print(f'dataset: {c_id}, num_images: {c[0].total_available_images}')

        for i in range(c[0].total_available_images):
            in_pfx = f'{c[1]}/{str(i).zfill(8)}_'
            out_pfx = f'{OUT_PATH}/{str(j).zfill(8)}_'
            shutil.copy(f'{in_pfx}depth.png', f'{out_pfx}depth.png')
            shutil.copy(f'{in_pfx}depth_rgba.png', f'{out_pfx}depth_rgba.png')

            in_labels_img = np.array(Image.open(f'{in_pfx}labels.png')).astype(np.uint16)
            out_labels_img = np.zeros(in_labels_img.shape, dtype=np.uint16)
            out_labels_rgba = np.zeros((in_labels_img.shape[0], in_labels_img.shape[1], 4), dtype=np.uint8)

            for label_idx in range(num_colors):
                if label_idx > 0:
                    in_label_idx = mapping['labels'][str(label_idx)][c_id]
                    out_labels_img[in_labels_img == in_label_idx] = label_idx
                    out_labels_rgba[in_labels_img == in_label_idx, :] = out_color_mapping[str(label_idx)]

            Image.fromarray(out_labels_img).save(f'{out_pfx}labels.png')
            Image.fromarray(out_labels_rgba).save(f'{out_pfx}labels_rgba.png')
            j += 1

    # write json config as entry point into model
    obj= {}
    obj['img_dims'] = img_dims
    obj['num_images'] = j
    obj['id_to_color'] = {'0': [0, 0, 0, 0]}
    for c_id in range(num_colors):
        if c_id > 0:
            c = out_color_mapping[str(c_id)]
            obj['id_to_color'][str(c_id)] = [int(c[0]), int(c[1]), int(c[2]), 255]

    cfg_json_file = open(f'{OUT_PATH}/config.json', 'w')
    cfg_json_file.write(json.dumps(obj))
    cfg_json_file.close()

if __name__ == '__main__':
    main()
