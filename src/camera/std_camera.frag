#version 430 core

in vec3 v_color;
in float v_depth;

layout (location = 0) out vec4 rgba_out;
layout (location = 1) out uvec4 depth_out;
// layout (location = 3) out uint depth_out1;


void main() {

    // if (a_pos_z < 10.) {
        // discard;
        // return;
    // }

    rgba_out = vec4(v_color, 1.);

    depth_out.x = uint(v_depth);
    // depth_out1 = 6904;

    // rgba_out = vec4(0., 0., a_pos_z / 6000., 1.);
}
