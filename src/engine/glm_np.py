import glm
import numpy as np

# slow but ergonomic wrappers.. why is this so annoying python?
# (library interop for this kind of stuff in c++ really sucks too actually)
def translate(v):
    return np.array(glm.translate(glm.mat4(), (v[0], v[1], v[2])))

def rotate(axis, r):
    return np.array(glm.rotate(glm.mat4(), r, (axis[0], axis[1], axis[2])))

def scale(s):
    return np.array(glm.scale(glm.mat4(), (s[0], s[1], s[2])))
