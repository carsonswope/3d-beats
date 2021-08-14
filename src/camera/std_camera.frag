#version 430 core

in vec3 v_color;
in float v_depth;

layout (location = 0) out vec4 rgba_out;
layout (location = 1) out uvec4 depth_out; // really just single uint16 value

void main() {
    rgba_out = vec4(v_color, 1.);
    depth_out.x = uint(v_depth);
}
