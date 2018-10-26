import cv2
import numpy as np	
f = 460
cx = 320
cy = 240
#Aufloesung 640x480
#X1=(10,10,100), X2=(33,22,111), X3=(100,100,1000), X4=(20,-100,100)

x1_3D = np.array(([10.],[10.],[100.]))
x2_3D = np.array(([33.],[22.],[111.]))
x3_3D = np.array(([100.],[100.],[1000.]))
x4_3D = np.array(([20.],[-100.],[100.]))

#Define K
K = np.zeros((3,3))
K[0,0] = f
K[1,1] = f
K[0,2] = cx
K[1,2] = cy
K[2,2] = 1

#Define P
R = np.eye(3)
t = np.array([[ 0.],[ 0.],[ 0.]])
Rt = np.hstack((R,t))
P = K.dot(Rt)

# 3D kartesisch -> 3D homogen
x1_3D_homo = np.vstack ((x1_3D, np.array([[1.]])))
x2_3D_homo = np.vstack ((x2_3D, np.array([[1.]])))
x3_3D_homo = np.vstack ((x3_3D, np.array([[1.]])))
x4_3D_homo = np.vstack ((x4_3D, np.array([[1.]])))

# 3D homogen -> 2D homogen
x1_2D_homo = P.dot(x1_3D_homo)
x2_2D_homo = P.dot(x2_3D_homo)
x3_2D_homo = P.dot(x3_3D_homo)
x4_2D_homo = P.dot(x4_3D_homo)

# 2D homogen -> 2D katesisch
x1_2D = np.array([np.rint(x1_2D_homo[0]/x1_2D_homo[2]),np.rint(x1_2D_homo[1]/x1_2D_homo[2])])
x2_2D = np.array([np.rint(x2_2D_homo[0]/x2_2D_homo[2]),np.rint(x2_2D_homo[1]/x2_2D_homo[2])])
x3_2D = np.array([np.rint(x3_2D_homo[0]/x3_2D_homo[2]),np.rint(x3_2D_homo[1]/x3_2D_homo[2])])
x4_2D = np.array([np.rint(x4_2D_homo[0]/x4_2D_homo[2]),np.rint(x4_2D_homo[1]/x4_2D_homo[2])])

print "Aufgabe 1:"
print ("X1 = " + str(x1_2D))
print ("X2 = " + str(x2_2D))
print ("X3 = " + str(x3_2D))
print ("X4 = " + str(x4_2D))

print "Aufgabe 2:"
points3D = np.array([x1_3D, x2_3D,x3_3D,x4_3D])
points2D = cv2.projectPoints(points3D, np.array([0.,0.,0.]), np.array([0.,0.,0.]), K, None)
print ("X1 = " + str(np.rint(points2D[0][0])))
print ("X2 = " + str(np.rint(points2D[0][1])))
print ("X3 = " + str(np.rint(points2D[0][2])))
print ("X4 = " + str(np.rint(points2D[0][3])))
