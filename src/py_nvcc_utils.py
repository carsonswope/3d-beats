from pycuda.compiler import SourceModule

import os

def get_module(filename):
    try:
        # load/compile cuda kernels..
        cu_file = open(filename, 'r')
        cu_text = cu_file.read()
        return SourceModule(cu_text, no_extern_c=True, include_dirs=[
            os.getcwd() + '/src/cuda',
            os.getcwd() + '/src/cuda/deps/glm'])

        # cu_eval_random_features = cu_mod.get_function('evaluate_random_features')
        # cu_pick_best_features = cu_mod.get_function('pick_best_features')
        # cu_copy_pixel_groups = cu_mod.get_function('copy_pixel_groups')
    except Exception as e:
        print('error:')
        print(e.msg)
        print(e.stdout)
        print(e.stderr)
        exit()
