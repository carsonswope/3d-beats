from window import AppBase

import imgui

class GlIntegrationTest(AppBase):
    def __init__(self):
        super().__init__(title="Test-icles")

    def tick(self, t):
        imgui.begin("Stuff")
        imgui.text("Cam angles")
        imgui.end()


if __name__ == '__main__':
    a = GlIntegrationTest()
    a.run()
