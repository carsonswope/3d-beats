
from setuptools import Extension, setup
from Cython.Build import cythonize

# NVCOMP_INCLUDE_DIR = './nvcomp/include'
# NVCOMP_LIB_NAME = 'nvcomp'
# NVCOMP_LIB_DIR = './nvcomp/lib'

# CUDA_INCLUDE_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include'
# CUDA_LIB_NAME = 'cudart'
# CUDA_LIB_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64'

extensions = [Extension(
    'cpp_grouping',
    ['./cpp_grouping.pyx'],
    language='c++',
    include_dirs=['.'],
    libraries=['CppGrouping'],
    library_dirs=['./build'],
)]

setup(name='cpp_grouping', ext_modules=cythonize(extensions, compiler_directives={'language_level' : '3'}))