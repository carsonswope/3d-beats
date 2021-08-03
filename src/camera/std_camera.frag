#version 430 core

in vec3 v_color;
// in float a_pos_z;

layout (location = 0) out vec4 rgba_out;


void main() {

    // if (a_pos_z < 10.) {
        // discard;
        // return;
    // }

    rgba_out = vec4(v_color, 1.);

    // rgba_out = vec4(0., 0., a_pos_z / 6000., 1.);
}
