from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.raw.GL.VERSION.GL_2_0 import GL_FRAGMENT_SHADER, GL_VERTEX_SHADER

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

    # def uniformMatrix4fv(self, name, )
    def u_pos(self, name):
        return glGetUniformLocation(self._program, name)
