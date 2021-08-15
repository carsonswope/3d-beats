import glm
import numpy as np

# slow but ergonomic wrappers.. why is this so annoying python?
# (library interop for this kind of stuff in c++ really sucks too actually)
def translate(v):
    return np.array(glm.translate(glm.mat4(), (np.float64(v[0]), np.float64(v[1]), np.float64(v[2]))))

def rotate(axis, r):
    return np.array(glm.rotate(glm.mat4(), np.float64(r), (np.float64(axis[0]), np.float64(axis[1]), np.float64(axis[2]))))

def rotate_x(r):
    return rotate((1., 0., 0.), r)

def rotate_y(r):
    return rotate((0., 1., 0.), r)

def rotate_z(r):
    return rotate((0., 0., 1.), r)

def scale(s):
    return np.array(glm.scale(glm.mat4(), (np.float64(s[0]), np.float64(s[1]), np.float64(s[2]))))
