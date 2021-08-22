from OpenGL.GL import *
import pycuda.gl
import pycuda.driver as cu
import pycuda.gpuarray as cu_array

import numpy as np

from engine.buffer import GpuBuffer

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

        self.np_dims = (self.dims[1], self.dims[0])
        if self.num_components > 1:
            self.np_dims = self.np_dims + (self.num_components,)

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
            b = np.zeros(self.np_dims, dtype=self.np_dtype)
        else:
            return_b = False
        self.assert_dims_match(b)

        glBindTexture(GL_TEXTURE_2D, self.gl())
        glGetTexImage(GL_TEXTURE_2D, 0, self.format, self.dtype, array=b)

        if return_b:
            return b
    
    def copy_to_gpu_buffer(self, b: GpuBuffer, offset=0, format=None):
        glBindBuffer(GL_PIXEL_PACK_BUFFER, b.gl())
        glBindTexture(GL_TEXTURE_2D, self.gl())

        # allow format override. for example, if texture object is RGBA but only want RGB, can specify format=GL_RGB
        if format is None:
            format = self.format

        # array arg treated as byte offset into buffer object, because GL_PIXEL_PACK_BUFFER is bound
        glGetTexImage(GL_TEXTURE_2D, 0, format, self.dtype, array=offset, outputType=None)

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def copy_from_gpu_buffer(self, b: GpuBuffer):

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, b.gl())
        glBindTexture(GL_TEXTURE_2D, self.gl())

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.dims[0], self.dims[1], self.format, self.dtype, None)

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)


    def assert_dims_match(self, b: np.array):
        assert b.dtype == self.np_dtype, 'datatypes dont match'
        assert b.shape == self.np_dims, 'shape dims dont match'
