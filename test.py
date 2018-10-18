#test programm
import numpy as np
import cv2

#print cv2.__version__

img = cv2.imread("uebungsblatt4/KITTI46_13.png")

px = img[100,100]

print px

print img [100,100,0]