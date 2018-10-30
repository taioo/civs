import cv2
import numpy as np	

#Aufloesung 640x480
img = np.zeros([480,640])
img[:] = 255


# 3D points
x1 = np.array(([10.],[10.],[100.]))
x2 = np.array(([33.],[22.],[111.]))
x3 = np.array(([100.],[100.],[1000.]))
x4 = np.array(([20.],[-100.],[100.]))
points3D = np.array([x1,x2,x3,x4])
x_img = cv2.imread("x.png")


# Kalibrierung
f = 460
cx = 320
cy = 240

# K
K = np.zeros((3,3))
K[0,0] = f
K[1,1] = f
K[0,2] = cx
K[1,2] = cy
K[2,2] = 1


#Rotationsmatrix
R = np.eye(3)

#Translationsvektor
t = np.zeros((3,1))

#[R|t]
Rt = np.hstack((R,t))

# P = K[R|t]
P = np.dot(K,Rt)

# katesisch -> homogen
x1 = np.vstack ((x1, np.array([[1.]])))
x2 = np.vstack ((x2, np.array([[1.]])))
x3 = np.vstack ((x3, np.array([[1.]])))
x4 = np.vstack ((x4, np.array([[1.]])))

# x = PX       3D -> 2D
x1 = np.dot(P,x1)
x2 = np.dot(P,x2)
x3 = np.dot(P,x3)
x4 = np.dot(P,x4)

#homogen -> katesisch
x1 = np.array([np.rint(x1[0]/x1[2]),np.rint(x1[1]/x1[2])])
x2 = np.array([np.rint(x2[0]/x2[2]),np.rint(x2[1]/x2[2])])
x3 = np.array([np.rint(x3[0]/x3[2]),np.rint(x3[1]/x3[2])])
x4 = np.array([np.rint(x4[0]/x4[2]),np.rint(x4[1]/x4[2])])

print "Aufgabe 1:"
print ("X1 = " + str(x1))
print ("X2 = " + str(x2))
print ("X3 = " + str(x3))
print ("X4 = " + str(x4))

print "Aufgabe 2:"
points2D = cv2.projectPoints(points3D, np.array([0.,0.,0.]), np.array([0.,0.,0.]), K, None)
print ("X1 = " + str(np.rint(points2D[0][0])))
print ("X2 = " + str(np.rint(points2D[0][1])))
print ("X3 = " + str(np.rint(points2D[0][2])))
print ("X4 = " + str(np.rint(points2D[0][3])))

cv2.circle(img,(x1[0],x1[1]), 5, (0,255,0), -1)
cv2.circle(img,(x2[0],x2[1]), 5, (0,255,0), -1)
cv2.circle(img,(x3[0],x3[1]), 5, (0,255,0), -1)
cv2.circle(img,(x4[0],x4[1]), 5, (0,255,0), -1)
cv2.imwrite('end.png', img)

img = points2D[0][0]
img = points2D[0][2]
img = points2D[0][3]
#img = points2D[0][4]


