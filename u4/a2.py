import numpy as np
import cv2

image = cv2.imread("KITTI46_13.png")


width = 270
height = 150

x = int(930 - width/2)
y = int(230 - height/2)

cut = image[y:y+height, x:x+width]

print cut.shape
print image.shape

cv2.imwrite('cut.png', cut)
