import glob
import cv2
import numpy as np


images = glob.glob("bild/*.jpg")

# generate object points
objectPoints = np.zeros((9 * 6, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

def project_chessboard_points(fx, cx):
    
    print fx
    print cx
    for i in range(len(images)):
        
        image = cv2.imread(images[i])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        res, corners = cv2.findChessboardCorners(image, (9, 6))

        # Draw the chessboard corners
        cv2.drawChessboardCorners(image, (9, 6), corners, res)

        # Display chessboard corners
        cv2.imshow("Chessboard Corner: " + str(i), image)
        cv2.waitKey(0)

        # get image dimension
        image_x = image.shape[0]
        image_y = image.shape[1]

        #Calibrate use default fx and cx

        res, K, distC, R, t = cv2.calibrateCamera([objectPoints], [corners], (image_x, image_y), None, None)

        # if value not none
        if fx:
            K[0][0] = fx
        if cx:
            K[0][2] = cx

        # Get points
        imgPoints = cv2.projectPoints(objectPoints, np.asanyarray(R), np.asanyarray(t), K, None)[0]

        #print num of the points
        print str(len(imgPoints))

        # new image to see the diffrent 
        image = cv2.imread(images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #print the points
        for j in range(len(imgPoints)):
            # Retrieve point's x & y coordinates
            x = np.rint(imgPoints[j][0][0]).astype(int)
            y = np.rint(imgPoints[j][0][1]).astype(int)
            #print circle on the image
            cv2.circle(image, (x, y), 3, (12, 86, 237), -1)



        cv2.imshow("Projection Point Result: " + str(i+1), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


project_chessboard_points(None, None)

print "input for fx and cx"
fx = input("fx: ")
cx = input("cx: ")

project_chessboard_points(fx, cx)