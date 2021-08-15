from OpenGL.GL import *
from engine.mesh import GpuMesh
import numpy as np

def make_cylinder(num_sections=8):

    vtx_pos = []
    vtx_col = []
    idxes = []

    for s in range(num_sections):
        theta = (s / num_sections) * np.pi * 2
        vtx_pos.append([np.cos(theta), np.sin(theta), -0.5, 1.])
        vtx_col.append([200, 200, 200])
        vtx_pos.append([np.cos(theta), np.sin(theta), 0.5, 1.])
        vtx_col.append([200, 200, 200])

        s2 = s*2

        idxes.append(s2 % (num_sections * 2))
        idxes.append((s2 + 1)% (num_sections * 2))
        idxes.append((s2 + 2)% (num_sections * 2))
        idxes.append((s2 + 1)% (num_sections * 2))
        idxes.append((s2 + 2)% (num_sections * 2))
        idxes.append((s2 + 3)% (num_sections * 2))

    idxes = np.array(idxes, dtype=np.uint32)
    vtx_pos = np.array(vtx_pos, dtype=np.float32)
    vtx_col = np.array(vtx_col, dtype=np.uint8)

    m = GpuMesh(num_idxes=len(idxes), vtxes_shape=(num_sections * 2,))
    m.idxes.cu().set(idxes)
    m.vtx_pos.cu().set(vtx_pos)
    m.vtx_color.cu().set(vtx_col)

    return m
