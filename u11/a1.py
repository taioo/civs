import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header 
'''


def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f')


def get_pointcloud(f, cx, cy, scale, depth):
    rows = depth.shape[0]
    cols = depth.shape[1]
    vector = []
    for i in range(0, rows):
        for j in range(0, cols):
            Z = depth[i, j] / scale
            if Z > 0:
                X = ((j - cx) * Z) / f
                Y = ((i - cy) * Z) / f
                vector.append([X, Y, Z])
    return np.array(vector)


def depth_to_pointcloud(fx, fy, cx, cy, scale, depth_img):
    pointcloud = get_pointcloud(fx, cx, cy, scale, depth_img)
    write_ply("pointclouds/pointcloud.ply", pointcloud)
    print("END")


# < 0 -> with alpha channel (raw)
depth_to_pointcloud(363.58, 363.58, 250.32, 212.55, 5000.5, cv2.imread("images/CoRBS_E1.png", -1))
