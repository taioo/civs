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

cv2.drawKeypoints(gray_Kitti11_94, keypoints_kitti11_94_cv2, kitti11_94_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('Kitti11_94_Keypoints.png',kitti11_94_img)


#[keypoints_kitti11_96_raw, descriptors_kitt11_96_raw] = detect_keypoints(imagesKitti11_paths[1], 5)
#np.savetxt('Kitti11_96_Keypoints', keypoints_kitti11_96_raw, delimiter=',')
#np.savetxt('Kitti11_96_Descriptors', descriptors_kitt11_96_raw, delimiter=',')
keypoints_kitti11_96 = np.loadtxt('Kitti11_96_Keypoints', delimiter=',')
descriptors_kitt11_96 = np.loadtxt('Kitti11_96_Descriptors', delimiter=',')
keypoints_kitti11_96_cv2 = to_cv2_kplist(keypoints_kitti11_96)
descriptors_kitti11_96_cv2 = to_cv2_di(descriptors_kitt11_96)
print ("Anzahl Keypoints in Kitti11_96: " + str(len(keypoints_kitti11_96_cv2)))
cv2.drawKeypoints(gray_Kitti11_96, keypoints_kitti11_96_cv2, kitti11_96_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
	kitti11_theshold_matches = cv2.drawMatchesKnn(kitti11_94_img, keypoints_kitti11_94_cv2, kitti11_96_img, keypoints_kitti11_96_cv2, good, None)



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


# bei 0.8 werden mehr linien angezeigt 
kitti11(0.8)
kitti14(0.8)

