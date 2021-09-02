import pycuda.compiler
import pycuda.driver

import argparse
import os

def add_args(parser):
        parser.add_argument('--fatbin_in', nargs='?', required=False, type=str, help='Directory to write cuda fatbins')
        parser.add_argument('--fatbin_out', nargs='?', required=False, type=str, help='Directory to load cuda fatbins')

def config_compiler(args):
    f_in = args.fatbin_in
    f_out = args.fatbin_out

    # cant write out and in! when writing out, those fatbins are automatically loaded back in to execute program
    assert not (f_in and f_out)

    if f_in:
        pycuda.compiler.set_fatbin_in_dir(f_in)
    
    if f_out:
        from cuda.default_gencodes import GENCODES
        pycuda.compiler.set_fatbin_out(f_out, GENCODES)

def get_module(n):

    try:
        if pycuda.compiler.FATBIN_IN_DIR:
            # load precompiled by looking for fatbin with name of hash of module name (n)
            return pycuda.compiler.SourceModule(None, checksum_text=n)

        filename = f'./src/cuda/{n}.cu'
        with open(filename, 'r') as f:
            return pycuda.compiler.SourceModule(f.read(), no_extern_c=True, include_dirs=[
                os.getcwd() + '/src/cuda',
                os.getcwd() + '/src/cuda/deps/glm'],
                checksum_text=n)

    except Exception as e:
        print('error:')
        print(e.msg)
        print(e.stdout)
        print(e.stderr)
        exit()
