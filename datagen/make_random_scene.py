D = bpy.data

# extract depth image
xdim, ydim = D.images['Viewer Node'].size
img = np.array(D.images['Viewer Node'].pixels).reshape((ydim, xdim, 4))
img = img[:,:,0]

D.imagesbpy

cam = D.objects['Camera']

finger_bone = bpy.context.scene.objects['Armature'].pose.bones['palm.03.R.001']

bpy.ops.render.render()

color_labels_node = bpy.data.scenes['Scene'].node_tree.nodes["Color_labels"]
depth_img_node = bpy.data.scenes['Scene'].node_tree.nodes["Depth_img"]

node_tree = bpy.data.scenes['Scene'].node_tree

s = D.scenes['Scene'].node_tree.nodes['switch1']

def get_both_images():
    s = D.scenes['Scene'].node_tree.nodes['switch1']
    s.check = False
    bpy.ops.render.render(write_still=False)
    color_labels_img = np.array(D.images['Viewer Node'].pixels).reshape((ydim, xdim, 4))
    s.check = True
    bpy.ops.render.render(write_still=False)
    depth_img = np.array(D.images['Viewer Node'].pixels).reshape((ydim, xdim, 4))
    depth_img = depth_img[:,:,0]
    return color_labels_img, depth_img
