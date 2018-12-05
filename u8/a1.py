import cv2
import numpy as np
from siftdetector import detect_keypoints
import scipy
import PIL
import glob

imagesKitti11_paths = glob.glob('images/KITTI11*')
imagesKitti14_paths = glob.glob('images/KITTI14*')

#Kitt11
kitti11_94_img = cv2.imread(imagesKitti11_paths[0])
kitti11_96_img = cv2.imread(imagesKitti11_paths[1])
gray_Kitti11_94 = cv2.cvtColor(kitti11_94_img, cv2.COLOR_BGR2GRAY)
gray_Kitti11_96 = cv2.cvtColor(kitti11_96_img ,cv2.COLOR_BGR2GRAY)

#Kitti14
kitti14_left_img = cv2.imread(imagesKitti14_paths[0])
kitti14_right_img = cv2.imread(imagesKitti14_paths[1])
gray_Kitti14_left = cv2.cvtColor(kitti14_left_img, cv2.COLOR_BGR2GRAY)
gray_Kitti14_right = cv2.cvtColor(kitti14_right_img ,cv2.COLOR_BGR2GRAY)


def to_cv2_kplist(kp):
	return list(map(to_cv2_kp, kp))

def to_cv2_kp(kp):
	return cv2.KeyPoint(kp[1], kp[0], kp[2], kp[3]/np.pi*180)

def to_cv2_di(di):
	return np.asarray(di, np.float32)


#[keypoints_kitti11_94_raw, descriptors_kitti11_94_raw] = detect_keypoints(imagesKitti11_paths[0], 5)
#np.savetxt('Kitti11_94_Keypoints', keypoints_kitti11_94_raw, delimiter=',')
#np.savetxt('Kitti11_94_Descriptors', descriptors_kitti11_94_raw, delimiter=',')
keypoints_kitti11_94 = np.loadtxt('Kitti11_94_Keypoints', delimiter=',')
descriptors_kitti11_94 = np.loadtxt('Kitti11_94_Descriptors', delimiter=',')

keypoints_kitti11_94_cv2 = to_cv2_kplist(keypoints_kitti11_94)
descriptors_kitti11_94_cv2 = to_cv2_di(descriptors_kitti11_94)

print ("Anzahl Keypoints in Kitti11_94: " + str(len(keypoints_kitti11_94_cv2)))

kitti11_94_img=cv2.drawKeypoints(gray_Kitti11_94, keypoints_kitti11_94_cv2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Kitti11_94_Keypoints.png',kitti11_94_img)


#[keypoints_kitti11_96_raw, descriptors_kitt11_96_raw] = detect_keypoints(imagesKitti11_paths[1], 5)
#np.savetxt('Kitti11_96_Keypoints', keypoints_kitti11_96_raw, delimiter=',')
#np.savetxt('Kitti11_96_Descriptors', descriptors_kitt11_96_raw, delimiter=',')
keypoints_kitti11_96 = np.loadtxt('Kitti11_96_Keypoints', delimiter=',')
descriptors_kitt11_96 = np.loadtxt('Kitti11_96_Descriptors', delimiter=',')
keypoints_kitti11_96_cv2 = to_cv2_kplist(keypoints_kitti11_96)
descriptors_kitti11_96_cv2 = to_cv2_di(descriptors_kitt11_96)
print ("Anzahl Keypoints in Kitti11_96: " + str(len(keypoints_kitti11_96_cv2)))
kitti11_96_img = cv2.drawKeypoints(gray_Kitti11_96, keypoints_kitti11_96_cv2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Kitti11_96_Keypoints.png',kitti11_96_img)


#[keypoints_kitti14_left_raw, descriptors_kitt14_left_raw] = detect_keypoints(imagesKitti14_paths[0], 5)
#np.savetxt('Kitti14_left_Keypoints', keypoints_kitti14_left_raw, delimiter=',')
#np.savetxt('Kitti14_left_Descriptors', descriptors_kitt14_left_raw, delimiter=',')
keypoints_kitti14_left = np.loadtxt('Kitti14_left_Keypoints', delimiter=',')
descriptors_kitt14_left = np.loadtxt('Kitti14_left_Descriptors', delimiter=',')
keypoints_kitti14_left_cv2 = to_cv2_kplist(keypoints_kitti14_left)
descriptors_kitti14_left_cv2 = to_cv2_di(descriptors_kitt14_left)
print ("Anzahl Keypoints in Kitti14_left: " + str(len(keypoints_kitti14_left_cv2)))
cv2.drawKeypoints(gray_Kitti14_left, keypoints_kitti14_left_cv2, kitti14_left_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('/Kitti14_left_Keypoints.png',kitti14_left_img)


#[keypoints_kitti14_right_raw, descriptors_kitt14_right_raw] = detect_keypoints(imagesKitti14_paths[1], 5)
#np.savetxt('Kitti14_right_Keypoints', keypoints_kitti14_right_raw, delimiter=',')
#np.savetxt('Kitti14_right_Descriptors', descriptors_kitt14_right_raw, delimiter=',')
keypoints_kitti14_right = np.loadtxt('Kitti14_right_Keypoints', delimiter=',')
descriptors_kitt14_right = np.loadtxt('Kitti14_right_Descriptors', delimiter=',')
keypoints_kitti14_right_cv2 = to_cv2_kplist(keypoints_kitti14_right)
descriptors_kitti14_right_cv2 = to_cv2_di(descriptors_kitt14_right)
print ("Anzahl Keypoints in Kitti14_right: " + str(len(keypoints_kitti14_right_cv2)))
cv2.drawKeypoints(gray_Kitti14_right, keypoints_kitti14_right_cv2, kitti14_right_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Kitti14_right_Keypoints.png',kitti14_right_img)


def kitti11(a):
	bf = cv2.BFMatcher()
	matches = bf.match(descriptors_kitti11_94_cv2, descriptors_kitti11_96_cv2)
	kitti11_all_matches = cv2.drawMatches(kitti11_94_img , keypoints_kitti11_94_cv2, kitti11_96_img, keypoints_kitti11_96_cv2, matches, None)
	cv2.imwrite("all_Matches_Kitti11.png",kitti11_all_matches)
	print ("Anzahl aller Matches Kitti11: " + str(len(matches)))

	matches = sorted(matches, key = lambda x:x.distance)
	kitti11_best_matches = cv2.drawMatches(kitti11_94_img, keypoints_kitti11_94_cv2, kitti11_96_img, keypoints_kitti11_96_cv2, matches[:30], None)
	cv2.imwrite("/best_30_Matches_Kitti11.png",kitti11_best_matches)

	matches = bf.knnMatch(descriptors_kitti11_94_cv2, descriptors_kitti11_96_cv2, k=2)
	good = []
	pts1 = []
	pts2 = []
	theshold_matching = a
	for m,n in matches:
		if m.distance < theshold_matching*n.distance:
			good.append([m])
			pts1.append(keypoints_kitti11_94_cv2[m.queryIdx].pt)
			pts2.append(keypoints_kitti11_96_cv2[m.trainIdx].pt)
	kitti11_theshold_matches = cv2.drawMatchesKnn(kitti11_94_img, keypoints_kitti11_94_cv2, kitti11_96_img, keypoints_kitti11_96_cv2, good, np.array([]))
	return pts1, pts2


def kitti14(a):
	bf = cv2.BFMatcher()

	matches = bf.match(descriptors_kitti14_left_cv2, descriptors_kitti14_right_cv2)
	kitti14_all_matches = cv2.drawMatches(kitti14_left_img , keypoints_kitti14_left_cv2, kitti14_right_img, keypoints_kitti14_right_cv2, matches, None)
	cv2.imwrite("/all_Matches_Kitti14.png",kitti14_all_matches)
	print ("Anzahl aller Matches Kitti14: " + str(len(matches)))

	matches = sorted(matches, key = lambda x:x.distance)
	kitti14_best_matches = cv2.drawMatches(kitti14_left_img, keypoints_kitti14_left_cv2, kitti14_right_img, keypoints_kitti14_right_cv2, matches[:30], None)
	cv2.imwrite("/best_30_Matches_Kitti14.png",kitti14_best_matches)

	matches = bf.knnMatch(descriptors_kitti14_left_cv2, descriptors_kitti14_right_cv2, k=2)
	good = []
	pts1 = []
	pts2 = []
	theshold_matching = a
	for m,n in matches:
		if m.distance < theshold_matching*n.distance:
			good.append([m])
			pts1.append(keypoints_kitti14_left_cv2[m.queryIdx].pt)
			pts2.append(keypoints_kitti14_right_cv2[m.trainIdx].pt)
	kitti14_theshold_matches = cv2.drawMatchesKnn(kitti14_left_img, keypoints_kitti14_left_cv2, kitti14_right_img, keypoints_kitti14_right_cv2, good, None)
	return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
		r,c = img1.shape
		img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
		img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
		for r,pt1,pt2 in zip(lines,pts1,pts2):
			color = tuple(np.random.randint(0,255,3).tolist()) 
			x0,y0 = map(int, [0, -r[2]/r[1] ])
			x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
			img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1) 
			img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
			img2 = cv2.circle(img2,tuple(pt2),5,color,-1) 
		return img1,img2
	

pts1,pts2 = kitti14(0.8)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
lines = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2 ,F)
lines = lines.reshape(-1,3)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
img1,img2 = drawlines(gray_Kitti11_94,gray_Kitti11_96,lines,pts1,pts2)

cv2.imwrite('img1a.png',img1)
cv2.imwrite('img2aa.png',img2)


##################A2###################

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


def verify_camerapose(P0,K, R1, R2, t, pts1, pts2):

    P1 = K.dot(np.hstack((R1, t)))
    P2 = K.dot(np.hstack((R1, -t)))
    P3 = K.dot(np.hstack((R2, t)))
    P4 = K.dot(np.hstack((R2, -t)))
    cMs = [P1, P2, P3, P4]
    list = []
    for p in cMs:
        X = cv2.triangulatePoints(P0,p,np.array(pts1,dtype=np.float),np.array(pts2,dtype=np.float))
        X /= X[3]
        list.append(check_3DPoints(X))

    max_value = max(list)
    max_index = list.index(max_value)
    return cMs[max_index]


def check_3DPoints(objectPoints):
    countPositiveDepth = 0
    for o in objectPoints.T:
        if o[2] >= 0:
            countPositiveDepth += 1

    return countPositiveDepth



fx = fy = 721.5
cx = 690.5
cy = 172.8
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
Rt = np.hstack((np.eye(3), np.zeros((3, 1))))
P0 = K.dot(Rt)
E = K.T * np.mat(F) * K
R1, R2, t = cv2.decomposeEssentialMat(E)
P1 = verify_camerapose(P0, K, R1, R2, t, pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2))
pointcloud = cv2.triangulatePoints(P0, P1, np.array(pts1.reshape(-1, 1, 2),dtype=np.float), np.array(pts2.reshape(-1, 1, 2),dtype=np.float))
pointcloud = cv2.convertPointsFromHomogeneous(pointcloud.T)
write_ply("KITTI11" + 'punktwolke.ply', pointcloud)




pts1,pts2 = kitti14(0.8)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
lines = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2 ,F)
lines = lines.reshape(-1,3)
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]
img1,img2 = drawlines(gray_Kitti14_left,gray_Kitti14_right,lines,pts1,pts2)

cv2.imwrite('img1b.png',img1)
cv2.imwrite('img2bb.png',img2)

fx = fy = 721.5
cx = 690.5
cy = 172.8
K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
Rt = np.hstack((np.eye(3), np.zeros((3, 1))))
P0 = K.dot(Rt)
E = K.T * np.mat(F) * K
R1, R2, t = cv2.decomposeEssentialMat(E)
P1 = verify_camerapose(P0, K, R1, R2, t, pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2))
pointcloud = cv2.triangulatePoints(P0, P1, np.array(pts1.reshape(-1, 1, 2),dtype=np.float), np.array(pts2.reshape(-1, 1, 2),dtype=np.float))
pointcloud = cv2.convertPointsFromHomogeneous(pointcloud.T)
write_ply("KITTI14" + 'punktwolke.ply', pointcloud)

# bei 0.8 werden mehr linien angezeigt 


