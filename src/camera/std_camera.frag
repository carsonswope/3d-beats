#version 430 core

uniform uint color_mode;
uniform vec4 solid_color;

in vec3 v_color;
in float v_depth;

layout (location = 0) out vec4 rgba_out;
layout (location = 1) out uvec4 depth_out; // really just single uint16 value

void main() {
    if (color_mode == 0) {
        // per-vtx color
        rgba_out = vec4(v_color, 1.);
    } else if (color_mode == 1) {
        // colid color
        rgba_out = solid_color;
    }
    depth_out.x = uint(v_depth);
}
