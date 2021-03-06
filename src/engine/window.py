from OpenGL.GL import * 
import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer

import pycuda.gl

import numpy as np
np.set_printoptions(suppress=True)

import gc
import time

class AppBase():
    def __init__(self, width=1920, height=1080, title="My Window", resizeable=True):

        self.width = width
        self.height = height
        self.window_resizeable = resizeable

        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)  
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        if not resizeable:
            glfw.window_hint(glfw.RESIZABLE, GL_FALSE)
        
        # TODO: can this be improved?
        m = glfw.get_primary_monitor()
        self.dpi_scale, _ = glfw.get_monitor_content_scale(m)

        self.window = glfw.create_window(int(self.width * self.dpi_scale), int(self.height * self.dpi_scale), title, None, None)
        glfw.make_context_current(self.window)

        glfw.swap_interval(0)

        imgui.create_context()
        self.imgui = GlfwRenderer(self.window)
        self.imgui_io = imgui.get_io()

        self.imgui_io.font_global_scale = self.dpi_scale

        import pycuda.autoinit
        self.cu_ctx = pycuda.gl.make_context(pycuda.autoinit.device)

        # splash/loading screen..
        glfw.poll_events()

        imgui.new_frame()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())
        imgui.end_frame()

        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        imgui.new_frame()

        self.splash()

        self.imgui.process_inputs()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())
        imgui.end_frame()

        glfw.swap_buffers(self.window)

        glfw.set_key_callback(self.window, self.key_event)

        self.ms_per_frame_log_max = 100
        self.ms_per_frame_log = [0]

        self.key_event_fns = []

        def esc_close(action):
            if action == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)
        self.register_key_event(glfw.KEY_ESCAPE, esc_close)

    def register_key_event(self, key, fn):
        self.key_event_fns.append((key, fn))

    def key_event(self, _window, key, _scancode, action, _mods):
        if not self.imgui_io.want_capture_keyboard:
            for k, fn in self.key_event_fns:
                if k == key:
                    fn(action)

    # 'background' window which is basically like writing directly to the main screen
    def begin_imgui_main(self):
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(self.width * self.dpi_scale, self.height * self.dpi_scale)
        imgui.set_next_window_bg_alpha(0.0)
        imgui.begin('##main',
            flags= imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)

    # def key_event(self, key, action, modifiers):
    #     self.imgui.key_event(key, action, modifiers)
        
    # def mouse_position_event(self, x, y, dx, dy):
    #     self.imgui.mouse_position_event(x, y, dx, dy)
    
    # def mouse_drag_event(self, x, y, dx, dy):
    #     self.imgui.mouse_drag_event(x, y, dx, dy)

    # def mouse_scroll_event(self, x_offset, y_offset):
    #     self.imgui.mouse_scroll_event(x_offset, y_offset)

    # def mouse_press_event(self, x, y, button):
    #     self.imgui.mouse_press_event(x, y, button)

    # def mouse_release_event(self, x: int, y: int, button: int):
    #     self.imgui.mouse_release_event(x, y, button)

    def run(self):

        start_time = time.perf_counter()

        while not glfw.window_should_close(self.window):
            t = time.perf_counter() - start_time

            # should be able to do this via on_resize callback instead!
            if self.window_resizeable:
                w_width, w_height = glfw.get_window_size(self.window)
                self.width = int(w_width / self.dpi_scale)
                self.height = int(w_height / self.dpi_scale)

            glfw.poll_events()

            glClearColor(0, 0, 0, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            imgui.new_frame()

            self.tick(t)

            # make sure to switch back to fill for imgui
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            self.imgui.process_inputs()
            imgui.render()
            draw_data = imgui.get_draw_data()
            self.imgui.render(draw_data)
            imgui.end_frame()

            glfw.swap_buffers(self.window)

            t_end = time.perf_counter() - start_time
            t_frame = t_end - t

            self.ms_per_frame_log.append(t_frame * 1000) # s -> ms
            while len(self.ms_per_frame_log) > self.ms_per_frame_log_max:
                self.ms_per_frame_log.pop(0)

def run_app(A):
    a = A()
    a.run()
    # RAII does not play well w/ python & GC. 
    # This is to force destructors to be called on GPU buffer/texture handles
    for p in dir(a):
        try:
            delattr(a, p)
        except:
            pass
    del a
    pycuda.autoinit.context.pop()
    glfw.terminate()
