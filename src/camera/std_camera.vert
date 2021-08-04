#version 430 core

uniform mat4 cam_proj;
uniform mat4 obj_tform;

layout(location = 0) in vec4 a_pos;
layout (location = 1) in vec3 a_color;

out vec3 v_color;

void main() {
    gl_Position = cam_proj * obj_tform * a_pos;
    v_color = a_color;
}