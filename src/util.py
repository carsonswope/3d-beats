import numpy as np

import pycuda.driver as cu

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)

def rs_projection(f, w, h, ppx, ppy, zmin, zmax):
    return np.array([
        [2*f / w, 0, 0, 0],
        [0, 2*f / h, 0, 0],
        [2*(ppx/w)-1, 2*(ppy/h)-1, (zmax+zmin)/(zmax-zmin), 1],
        [0, 0, 2*zmax*zmin/(zmin-zmax), 0]
    ], dtype=np.float32).T

def make_grid(dims, block_dims):
    assert len(dims) == 3
    assert len(block_dims) == 3
    r = lambda a,b: -(-a // b)
    return (
        r(dims[0], block_dims[0]),
        r(dims[1], block_dims[1]),
        r(dims[2], block_dims[2]))

class PagelockedCounter:
    def __init__(self):
        self.d = cu.pagelocked_zeros((1,), np.int64)

    @property
    def __array_interface__(self):
        return self.d.__array_interface__

    def set(self, new_count):
        self.d[0] = new_count

    def __call__(self):
        return self.d[0]

MAX_UINT16 = np.uint16(65535) # max for 16 bit unsigned
