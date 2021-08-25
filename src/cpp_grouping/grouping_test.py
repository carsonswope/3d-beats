from cpp_grouping import CppGrouping

import numpy as np

a = CppGrouping()

b = np.ones((6, 8), dtype=np.uint16)
b[3,:] = 0
b[:,4] = 0
b[5,:] = 2

print(b)

c = np.zeros((np.prod(b.shape), 3), dtype=np.int32)
# 1st row: L(0) or R(1)
# 2nd row: (num coords i.e. group size, center_coord_x, center_coord_y)
g_info = np.zeros((2, 3), dtype=np.float32)

a.make_groups(b, c, g_info, 0.05)

r_group_size = np.int32(g_info[0, 0])
r_group = c[0:r_group_size,:]

l_group_size = np.int32(g_info[1, 0])
l_group = c[r_group_size:r_group_size+l_group_size]

print('r')
print(r_group_size)
print(r_group)

print('l')
print(l_group_size)
print(l_group)
# print(g_info)
