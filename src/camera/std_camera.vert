#version 430 core

uniform mat4 tf;

layout(location = 0) in vec4 a_pos;
layout (location = 1) in vec3 a_color;

out vec3 v_color;

// out float a_pos_z;

void main() {
    // a_pos_z = a_pos.z;
    gl_Position = tf * a_pos;
    v_color = a_color;
}