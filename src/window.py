from OpenGL.GL import * 
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

import pycuda.gl

import numpy as np
np.set_printoptions(suppress=True)

import time

class AppBase():
    def __init__(self, width=1920, height=1080, title="My Window"):

        self.width = width
        self.height = height

        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)  
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)

        imgui.create_context()
        self.imgui = GlfwRenderer(self.window)
        self.imgui_io = imgui.get_io()

        # images still very small..
        self.imgui_io.font_global_scale = 2.0

        import pycuda.autoinit
        pycuda.gl.make_context(pycuda.autoinit.device)


    def key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)
        
    def mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)
    
    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def run(self):

        start_time = time.perf_counter()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            glClearColor(0, 0, 0, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            t = time.perf_counter() - start_time
            imgui.new_frame()

            self.tick(t)

            self.imgui.process_inputs()
            imgui.render()
            self.imgui.render(imgui.get_draw_data())
            imgui.end_frame()

            glfw.swap_buffers(self.window)

        glfw.terminate()
