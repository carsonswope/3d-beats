from OpenGL.GL import *

import numpy as np

# FBO
class GpuFramebuffer:
    def __init__(self, dims):
        self._id = glGenFramebuffers(1)
        self.dims = dims

        self.depth_internal = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_internal)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.dims[0], self.dims[1])

    @property
    def id(self):
        return self._id
    
    def bind(self, rgba_tex, depth_tex):

        if self.dims != rgba_tex.dims or self.dims != depth_tex.dims:
            print('Error. input texture has different dimensions than framebuffer')
            return

        glBindFramebuffer(GL_FRAMEBUFFER, self.id)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_internal);  
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rgba_tex.gl(), 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depth_tex.gl(), 0)
        glDrawBuffers(np.array([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1], dtype=np.uint32))

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('huh?')
        
        glViewport(0, 0, self.dims[0], self.dims[1])
