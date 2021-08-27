from pycuda.compiler import SourceModule
import pycuda.driver

import argparse
import os

def get_module(n):

    parser = argparse.ArgumentParser(description='Train a classifier RDF for depth images')
    parser.add_argument('--cu_use_fatbins', nargs='?', required=False, help='Path to the layered decision forest config file')
    args = parser.parse_known_args()[0]

    try:

        if 'cu_use_fatbins' in args:
            # print('Loading precompiled CUDA binaries')
            filename = f'./cuda_fatbin/{n}.fatbin'
            with open(filename, 'rb') as f:
                return pycuda.driver.module_from_buffer(f.read())

        else:
            # print('Compiling CUDA kernels from source')
            filename = f'./src/cuda/{n}.cu'
            with open(filename, 'r') as f:
                return SourceModule(f.read(), no_extern_c=True, include_dirs=[
                    os.getcwd() + '/src/cuda',
                    os.getcwd() + '/src/cuda/deps/glm'])

    except Exception as e:
        print('error:')
        print(e.msg)
        print(e.stdout)
        print(e.stderr)
        exit()
