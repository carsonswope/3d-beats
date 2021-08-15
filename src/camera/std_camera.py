from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.raw.GL.VERSION.GL_2_0 import GL_FRAGMENT_SHADER, GL_VERTEX_SHADER

import numpy as np

class StdCamera:
    def __init__(self):
        vs_text = open('./src/camera/std_camera.vert', 'r').read()
        fs_text = open('./src/camera/std_camera.frag', 'r').read()
        try:
            vs = shaders.compileShader(vs_text, GL_VERTEX_SHADER)
            fs = shaders.compileShader(fs_text, GL_FRAGMENT_SHADER)
            self._program = shaders.compileProgram(vs, fs)
            glDeleteShader(vs)
            glDeleteShader(fs)
        except shaders.ShaderCompilationError as e:
            print('Error compiling shaders')
            print(e)
            return

    def use(self):
        shaders.glUseProgram(self._program)

    def u_pos(self, name):
        return glGetUniformLocation(self._program, name)

    def u_1ui(self, name, v):
        self.use()
        glUniform1ui(self.u_pos(name), np.uint(v))
    
    def u_4ui(self, name, v):
        self.use()
        glUniform4uiv(self.u_pos(name), 1, v)

    def u_4f(self, name, v):
        self.use()
        glUniform4fv(self.u_pos(name), 1, v)

    def u_mat4(self, name, m):
        self.use()
        glUniformMatrix4fv(self.u_pos(name), 1, GL_TRUE, m)
