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
    
    def bind(self, rgba_tex=None, depth_tex=None):

        if rgba_tex is None and depth_tex is None:
            print('must bind either rgba or depth texture to fbo')
            return
        
        if rgba_tex is not None and self.dims != rgba_tex.dims:
            print('Error. rgba texture has different dimensions than framebuffer')
            return

        if depth_tex is not None and self.dims != depth_tex.dims:
            print('Error. depth texture has different dimensions than framebuffer')
            return

        glBindFramebuffer(GL_FRAMEBUFFER, self.id)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_internal);  
        if rgba_tex:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rgba_tex.gl(), 0)
        if depth_tex:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, depth_tex.gl(), 0)
        glDrawBuffers(np.array([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1], dtype=np.uint32))

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('error. gl framebuffer not GL_FRAMEBUFFER_COMPLETE')
            return
        
        glViewport(0, 0, self.dims[0], self.dims[1])
