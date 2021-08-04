from OpenGL.GL import *
import pycuda.gl
import pycuda.driver as cu
import pycuda.gpuarray as cu_array

import numpy as np

# Texture object holding OpenGL & CUDA interop texture.

class GpuTexture:
    def __init__(self, dims, internalFormat, format, dtype):
        self.dims = dims
        self.internalFormat = internalFormat
        self.format = format
        self.dtype = dtype

        self._gl = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self._gl)
        # todo: make this configurable
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, self.internalFormat, self.dims[0], self.dims[1], 0, self.format, self.dtype, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        # self._cu_handle = pycuda.gl.RegisteredImage(int(self._gl))

        # self._cu = None # (map handle, cu array)

    def gl(self):
        # if self._cu != None:
            # cu_map, _ = self._cu
            # cu_map.unmap()
            # self._cu = None

        return self._gl

    def cu(self):
        print('ERROR! cu interop not implemented yet!')
        return None
    #     if self._cu == None:
    #         cu_map = self._cu_handle.map()
    #         cu_arr = cu_array.GPUArray(self.shape, self.dtype, gpudata=cu_map.device_ptr_and_size()[0])
    #         self._cu = (cu_map, cu_arr)

    #     cu_map, cu_arr = self._cu
    #     return cu_arr
