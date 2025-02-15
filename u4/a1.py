
import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread("KITTI46_13.png")

    # cv2.imshow("show",img)
    # cv2.waitKey(0)
    #cv2.imwrite("new.png", img)

    blue = img.copy()
    blue[:, :, 1] = 0
    blue[:, :, 2] = 0

    red = img.copy()
    red[:, :, 1] = 0
    red[:, :, 0] = 0

    green = img.copy()
    green[:, :, 0] = 0
    green[:, :, 2] = 0
    # cv2.imshow("blue",blue)
    # cv2.waitKey(0)

    cv2.imwrite(":blue.png", blue)
    cv2.imwrite(":red.png", red)
    cv2.imwrite(":green.png", green)

    height, width, layers = img.shape
    zeroImgMatrix = np.zeros((height, width), dtype="uint8")

    b, g, r = cv2.split(img)
    b = cv2.merge([b, zeroImgMatrix, zeroImgMatrix])
    g = cv2.merge([zeroImgMatrix, g, zeroImgMatrix])
    r = cv2.merge([zeroImgMatrix, zeroImgMatrix, r])

    cv2.imwrite("sblue.png", b)
    cv2.imwrite("sred.png", r)
    cv2.imwrite("sgreen.png", g)
