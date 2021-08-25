from cpp_grouping import CppGrouping

import numpy as np

a = CppGrouping()
b = np.ones((6, 8), dtype=np.uint16)
b[3,:] = 0
b[:,4] = 0

a.make_groups(b)
