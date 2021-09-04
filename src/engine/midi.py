import rtmidi
import imgui

class Midi:
    def __init__(self):
        self.m = rtmidi.MidiOut()
        self.load_ports()
        self.open_default_port()
    
    def open_default_port(self):
        # look for LoopBe port, otherwise just open port 0
        self.open_port_idx = 0
        for i,p in self.available_ports:
            if 'LoopBe' in p:
                self.open_port_idx = i

        self.m.open_port(self.open_port_idx)

    def load_ports(self):
        self.available_ports = [(i,p) for i,p in enumerate(self.m.get_ports())]

    def draw_imgui(self):
        imgui.text('MIDI Port:')
        imgui.same_line()
        imgui.push_item_width(300.)
        clicked, current = imgui.combo("##midi-dropdown", self.open_port_idx, [p for _,p in self.available_ports])
        if clicked:
            self.open_port_idx = current
            self.m.close_port()
            self.m.open_port(self.open_port_idx)
        imgui.pop_item_width()

    def send(self, n):
        self.m.send_message(n)
