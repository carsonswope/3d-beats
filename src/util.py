import numpy as np

import pycuda.driver as cu

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)

class PagelockedCounter:
    def __init__(self):
        self.d = cu.pagelocked_zeros((1,), np.int64)
    
    @property
    def ptr(self):
        return self.d.__array_interface__['data'][0]
    
    def set(self, new_count):
        self.d[0] = new_count

    def __call__(self):
        return self.d[0]

MAX_UINT16 = np.uint16(65535) # max for 16 bit unsigned
