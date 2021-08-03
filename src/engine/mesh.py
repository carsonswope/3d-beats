from OpenGL.GL import *
from engine.buffer import GpuBuffer
import numpy as np

class GpuMesh:
    def __init__(self, num_idxes, vtxes_shape):
        self.num_idxes = num_idxes
        self.vtxes_shape = vtxes_shape
        # self.num_vtxes = np.prod(vtxes_shape)
        self.idxes = GpuBuffer((self.num_idxes,), dtype=np.uint32)
        self.vtx_pos = GpuBuffer(self.vtxes_shape + (4,), dtype=np.float32)
        self.vtx_color = GpuBuffer(self.vtxes_shape + (3,), dtype=np.uint8)

        self.vao = glGenVertexArrays(1)

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ARRAY_BUFFER, self.vtx_pos.gl())
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * np.dtype(np.float32).itemsize, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, self.vtx_color.gl())
        glVertexAttribPointer(1, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, None)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def draw(self):
        # self.idxes.gl()
        self.vtx_pos.gl()
        self.vtx_color.gl()

        glBindVertexArray(self.vao)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.idxes.gl())
        glDrawElements(GL_TRIANGLES, self.num_idxes, GL_UNSIGNED_INT, None)

