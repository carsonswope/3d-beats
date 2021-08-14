from OpenGL.GL import *
import pycuda.gl
import pycuda.driver as cu
import pycuda.gpuarray as cu_array

import numpy as np

# Texture object holding OpenGL & CUDA interop texture.

TEX_FORMAT = {
    #(format, dtype): (internal_format, num_components, np_dtype)
    (GL_RGBA, GL_UNSIGNED_BYTE): (GL_RGBA, 4, np.uint8),
    (GL_RGB, GL_UNSIGNED_BYTE): (GL_RGB, 3, np.uint8),
    (GL_RED_INTEGER, GL_UNSIGNED_SHORT): (GL_R16UI, 1, np.uint16)

    # (GL_RGBA, 
}

class GpuTexture:
    def __init__(self, dims, format_dtype):
        self.dims = dims
        self.format, self.dtype = format_dtype
        self.internal_format, self.num_components, self.np_dtype = TEX_FORMAT[format_dtype]

        self._gl = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self._gl)
        # todo: make this configurable
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, self.dims[0], self.dims[1], 0, self.format, self.dtype, None)
        glBindTexture(GL_TEXTURE_2D, 0)

    def gl(self):
        return self._gl

    def set(self, b):
        self.assert_dims_match(b)
        glBindTexture(GL_TEXTURE_2D, self.gl())
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.dims[0], self.dims[1], self.format, self.dtype, b)

    def get(self, b=None):
        if b == None:
            return_b = True
            d = (self.dims[1], self.dims[0])
            if self.num_components > 1:
                d = d + (self.num_components,)
            b = np.zeros(d, dtype=self.np_dtype)
        else:
            return_b = False
        self.assert_dims_match(b)

        glBindTexture(GL_TEXTURE_2D, self.gl())
        glGetTexImage(GL_TEXTURE_2D, 0, self.format, self.dtype, array=b)

        if return_b:
            return b
    
    def assert_dims_match(self, b: np.array):
        assert b.dtype == self.np_dtype, 'datatypes dont match'

        if self.num_components > 1:
            assert (self.dims[1], self.dims[0], self.num_components) == b.shape
        else:
            assert (self.dims[1], self.dims[0]) == b.shape

