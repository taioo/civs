import numpy as np
import cv2
import glob


image_pathes = glob.glob("bild/*.jpg")


for i in xrange(0, np.prod(image_pathes.size):
    print i

gray = cv2.cvtColor(cv2.imread(image_pathes[0]),cv2.COLOR_BGR2GRAY)


cv2.findChessboardCorners