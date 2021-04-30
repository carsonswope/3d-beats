import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

np.set_printoptions(suppress=True)

a = Image.open('./datagen/gen_4class/test_00000000_depth.png')
b = np.array(a)

l_img = cv2.imread('./datagen/xcam_greyscale_l.png')
r_img = cv2.imread('./datagen/xcam_greyscale_r.png')

# greyscales!
l_img_g = l_img[:,:,0]
r_img_g = r_img[:,:,0]

min_disparity = 0
max_disparity = 32
block_size = 11
matcher = cv2.StereoSGBM_create(min_disparity, max_disparity, block_size)

disparity = matcher.compute(l_img, r_img).astype(np.float32) / 16.

# depth = cv2.reprojectImageTo3D(disparity)

# baseline = 0.02
# focal = 0.05
# depth = (baseline * focal) / (disparity / max_disparity)

depth = np.zeros(disparity.shape, dtype=np.float32)
depth[np.where(disparity > 0)] = 100000 / disparity[np.where(disparity > 0)]
depth = depth.astype(np.uint16)


# depth = 11.5 / disparity

depth_img = Image.fromarray(depth)
depth_img.save('./datagen/xcam_calc_depth.png')

# plt.imshow(depth,'gray')


# plt.imshow(orig_gray,'gray')
# plt.imshow(disparity,'gray')
# plt.imshow(depth, "gray")
# plt.show()

"""
_, disparity = cv2.threshold(
        disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)



# depth_img = (baseline * focal) / (disparity / 16)

disparity_scaled = (disparity / 16.).astype(np.uint8)

# baseline

disparity_colour_mapped = cv2.applyColorMap(
    (disparity_scaled * (256. / max_disparity)).astype(np.uint8),
    cv2.COLORMAP_HOT)
cv2.imshow("disp window", disparity_colour_mapped)

# cv2.imshow("disps", (disparity_scaled * (256. / max_disparity)).astype(np.uint8))
"""
# cv2.waitKey(0)
