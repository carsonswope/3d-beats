#version 430 core

uniform mat4 cam_proj;
uniform mat4 cam_inv_tform;
uniform mat4 obj_tform;

layout(location = 0) in vec4 a_pos;
layout (location = 1) in vec3 a_color;

out vec3 v_color;
out float v_depth;

void main() {
    // position in camera 3d space
    vec4 p = cam_inv_tform * obj_tform * a_pos;
    v_depth = p.z;
    gl_Position = cam_proj * p;
    v_color = a_color;
}