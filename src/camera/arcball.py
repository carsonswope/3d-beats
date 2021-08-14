import engine.glm_np as glm_np
import numpy as np
import imgui

class ArcBallCam():
    def __init__(self):
        self.d = 5000.
        self.r0 = 3.1
        self.r1 = 1.0

    def get_cam_inv_tform(self):
        cam_tform = glm_np.rotate((0, 0, 1), self.r0) @ \
            glm_np.rotate((1, 0, 0), (np.pi/2) - self.r1) @ \
            glm_np.translate((0, 0, -self.d))

        return np.linalg.inv(cam_tform)

    def draw_control_gui(self):
        _, self.d = imgui.slider_float("d", self.d, 0, 10000)
        _, self.r0 = imgui.slider_float("r0", self.r0, 0, np.pi*2)
        _, self.r1 = imgui.slider_float("r1", self.r1, 0, np.pi/2)
