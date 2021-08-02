from OpenGL.GL import *
import pycuda.gl
import pycuda.driver as cu
import pycuda.gpuarray as cu_array

import numpy as np

# Array-like object holding OpenGL & CUDA interop buffer.

class GpuBuffer:
    def __init__(self, shape, dtype, data_ptr = None, gl_buffer_flag = GL_DYNAMIC_DRAW):
        self.shape = shape
        self.dtype = dtype

        self._gl = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._gl)
        glBufferData(GL_ARRAY_BUFFER, np.prod(shape) * np.dtype(dtype).itemsize, data_ptr, gl_buffer_flag)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._cu_handle = pycuda.gl.RegisteredBuffer(int(self._gl))

        self._cu = None # (map handle, cu array)

    def gl(self):
        if self._cu != None:
            cu_map, _ = self._cu
            cu_map.unmap()
            self._cu = None

        return self._gl

    def cu(self):
        if self._cu == None:
            cu_map = self._cu_handle.map()
            cu_arr = cu_array.GPUArray(self.shape, self.dtype, gpudata=cu_map.device_ptr_and_size()[0])
            self._cu = (cu_map, cu_arr)

        cu_map, cu_arr = self._cu
        return cu_arr
