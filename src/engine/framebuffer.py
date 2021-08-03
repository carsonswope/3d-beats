from OpenGL.GL import *

# FBO
class GpuFramebuffer:
    def __init__(self, dims):
        self._id = glGenFramebuffers(1)
        self.dims = dims
    
    @property
    def id(self):
        return self._id
    
    def bind(self, rgba_tex):

        if self.dims != rgba_tex.dims:
            print('Error. input texture has different dimensions than framebuffer')
            return

        glBindFramebuffer(GL_FRAMEBUFFER, self.id)
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, rgba_tex.gl(), 0)
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print('huh?')
        
        glViewport(0, 0, self.dims[0], self.dims[1])
