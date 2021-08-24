import numpy as np
import imgui

class FingertipState:
    def __init__(self, on_fn, off_fn, num_positions = 50, z_thresh = 150, midi_note = 36):
        self.num_positions = num_positions
        self.positions = [0 for _ in range(self.num_positions)]

        self.on_fn = on_fn
        self.off_fn = off_fn

        self.z_thresh = z_thresh
        self.midi_note = midi_note
        self.note_on = False
    
    def reset_positions(self):
        # turn off note if on
        self.positions = [0 for _ in range(self.num_positions)]
        self.set_midi_state(False)
        pass

    def next_z_pos(self, z_pos):

        self.positions.append(z_pos)
        while len(self.positions) > self.num_positions:
            self.positions.pop(0)
            
        if len(self.positions) > 10: # arbitrary..
            last_pos = self.positions[-1]
            if last_pos < self.z_thresh:
                self.set_midi_state(True)
            else:
                self.set_midi_state(False)

    def set_midi_state(self, s):
        if s and not self.note_on:
            self.note_on = True
            self.on_fn(self.midi_note, 127) # todo: velocity!

        elif not s and self.note_on:
            self.note_on = False
            self.off_fn(self.midi_note)


class HandState:
    def __init__(self, defaults, on_fn, off_fn, num_positions = 50):
        self.fingertips = [FingertipState(
            on_fn,
            off_fn,
            num_positions,
            z_thresh,
            midi_note) for z_thresh, midi_note in defaults]
    
    def draw_imgui(self):

        imgui.begin_group()

        c_x, c_y = imgui.get_cursor_pos()
        graph_dim_x = 300.
        slider_dim_x = 35.

        graph_pad = 15.
        dim_y = 150.
        graph_scale_z = 500.

        for i in range(len(self.fingertips)):

            c_x_start = c_x + ((graph_dim_x + graph_pad + slider_dim_x + graph_pad + graph_pad) * i)

            imgui.set_cursor_pos((c_x_start, c_y))

            if len(self.fingertips[i].positions) > 0:
                a = np.array(self.fingertips[i].positions, dtype=np.float32)
            else:
                a = np.array([0], dtype=np.float32)


            cursor_pos = imgui.get_cursor_screen_pos()

            imgui.plot_lines(f'##f{i} pos',
                a,
                scale_max=graph_scale_z,
                scale_min=0.,
                graph_size=(graph_dim_x, dim_y))

            f_threshold = self.fingertips[i].z_thresh

            if self.fingertips[i].note_on:
                thresh_color = imgui.get_color_u32_rgba(0.3,1,0.8,0.30)
            else:
                thresh_color = imgui.get_color_u32_rgba(0.3,1,0.8,0.05)

            imgui.get_window_draw_list().add_rect_filled(
                cursor_pos[0],
                cursor_pos[1] + (dim_y * (1 - (f_threshold / graph_scale_z))),
                cursor_pos[0] + graph_dim_x,
                cursor_pos[1] + dim_y,
                thresh_color)

            imgui.set_cursor_pos((c_x_start + graph_dim_x + graph_pad, c_y))

            _, self.fingertips[i].z_thresh = imgui.v_slider_float(f'##{i}', slider_dim_x, dim_y, self.fingertips[i].z_thresh, 25., 175.)

        imgui.end_group()
