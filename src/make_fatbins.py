import pycuda.driver

import pycuda.autoinit

import os

import pycuda.compiler

GENCODES= [
    # atomicAdd(double*, double) only introduced w/ arch 60.
    # need to put in own implementation if need to backport
    # 'arch=compute_52,code=sm_52',
    # 'arch=compute_53,code=sm_53',
    'arch=compute_60,code=sm_60',
    'arch=compute_60,code=compute_60',
    'arch=compute_61,code=sm_61',
    'arch=compute_62,code=sm_62',
    'arch=compute_70,code=sm_70',
    'arch=compute_72,code=sm_72',
    'arch=compute_75,code=sm_75',
    'arch=compute_80,code=sm_80',
    'arch=compute_86,code=sm_86',
    'arch=compute_86,code=compute_86',


]
CU_FILES = ['calibrated_plane', 'fit_mesh', 'mean_shift', 'points_ops', 'tree_eval', 'tree_train']

def run_command(c):
    result = os.system(c)
    if result != 0:
        raise Exception(f'Command: {c}\nNonzero return code: {result}')

def make_nvcc_fatbin_command(module_name, source_file, include_dirs=[], gencodes=[]):
    c = f'nvcc -m64 --fatbin -o ./cuda_fatbin/{module_name}.fatbin {source_file} '
    for i in include_dirs:
        c += f'-I{i} '
    c += f'-I{pycuda.__path__[0] + "/cuda"} '
    for g in gencodes:
        c += f'-gencode {g} '
    return c

def main():
    for f in CU_FILES:
        c = make_nvcc_fatbin_command(
            f,
            f'./src/cuda/{f}.cu',
            include_dirs=['./src/cuda', './src/cuda/deps/glm'],
            gencodes=GENCODES)
        run_command(c)

if __name__ == '__main__':
    main()
