from PIL import Image
# import rawpy
import numpy as np
import cv2


img_sz = 100

# pts_src = np.array([[867, 404], [976, 569], [1133, 464], [1026, 297]])
pts_src = np.array([[ 884,  417],
 [ 976,  567],
 [1124,  458],
 [1028,  304]])
pts_dest = np.array([[0, 0], [img_sz, 0], [img_sz, img_sz], [0, img_sz]])


im_src = cv2.imread('./datagen/l2_green.png')


current_p = 0

# while True:
    
h, status = cv2.findHomography(pts_src, pts_dest)

im_out = cv2.warpPerspective(im_src, h, (img_sz, img_sz))

im_out_f = im_out.astype(np.float)

im_out_f[80:,:] *= 1.05

im_out = im_out_f.astype(np.uint8)

num_tiles = 20

tile_out = np.zeros((img_sz * num_tiles, img_sz * num_tiles, 3), dtype='uint8')
for u in range(num_tiles):
    for v in range(num_tiles):
        x_start = u * img_sz
        y_start = v * img_sz
        tile_out[y_start:y_start + img_sz, x_start:x_start + img_sz,:] = np.copy(im_out)

# cv2.imshow("t.o.", tile_out)

tile_out_min = tile_out.min()
tile_out_max = tile_out.max()

tile_out_f = tile_out.astype(np.float)
tile_out_f -= tile_out_min
tile_out_f /= (tile_out_max - tile_out_min)
tile_out_f *= 255

tile_out = tile_out_f.astype(np.uint8)

ti = Image.fromarray(tile_out)



ti.save('l2_checker_normalized.png')

# cv2.imsave('l2_checker.png')

    # d = cv2.waitKey(0)
    # if d >= 49 and d <= 52:
    #     current_p = d - 49
    # elif d == 97:
    #     print('x- for p ', current_p)
    #     pts_src[current_p][0] -= 1
    # elif d == 100:
    #     print('x+ for p ', current_p)
    #     pts_src[current_p][0] += 1
    # elif d == 119:
    #     print('y- for p ', current_p)
    #     pts_src[current_p][1] -= 1
    # elif d == 115:
    #     print('y+ for p ', current_p)
    #     pts_src[current_p][1] += 1
    # else:
    #     print('dur')

    # print(pts_src)