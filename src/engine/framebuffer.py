from OpenGL.GL import *

import numpy as np

# FBO
class GpuFramebuffer:
    def __init__(self, dims):
        self._id = glGenFramebuffers(1)
        self.dims = dims
    
    @property
    def id(self):
        return self._id
    
    def bind(self, rgba_tex, depth_tex):

        if self.dims != rgba_tex.dims or self.dims != depth_tex.dims:
            print('Error. input texture has different dimensions than framebuffer')
            return

        glBindFramebuffer(GL_FRAMEBUFFER, self.id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rgba_tex.gl(), 0)
        # glDrawBuffer(GL_COLOR_ATTACHMENT0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depth_tex.gl(), 0)
        # glDrawBuffer(GL_COLOR_ATTACHMENT1)
        glDrawBuffers(np.array([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1], dtype=np.uint32))
        # glDrawBuffers(1, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('huh?')
        
        glViewport(0, 0, self.dims[0], self.dims[1])
