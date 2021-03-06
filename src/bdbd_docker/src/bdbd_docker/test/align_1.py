# align images from two different camera sources

import cv2
import numpy as np
import time

GOOD_MATCH_FRACTION = .15

def imgby2(img, factor=2):
    height, width = img.shape[:2]
    return cv2.resize(img, (width//factor, height//factor))

img1 = imgby2(cv2.imread('../data/pantilt.jpg'))
gray1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = imgby2(cv2.imread('../data/sr305.jpg'))
gray2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB

start = time.time()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_FRACTION)
matches = matches[:numGoodMatches]
# Draw good matches.
print('elapsed time', time.time() - start, 'good match count', len(matches))
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('matches', img3)
#sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SIFT_create(nfeatures=50)
#kp = sift.detect(gray,None)
#img=cv2.drawKeypoints(gray, kp, img)
#cv2.imshow('SIFT', img)
  
cv2.waitKey(0) 

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
print('h', h)

# Use homography
height, width = gray2.shape[0:2]
im1Reg = cv2.warpPerspective(img1, h, (width, height))
print('img1, img2, im1Reg', img1.shape, img2.shape, im1Reg.shape)
#cv2.imwrite('../data/sr305_sift.jpg',img)
cv2.imshow('warped', im1Reg)
  
cv2.waitKey(0)

# copy warped source to destination

for j in range(0, width):
    for i in range(0, height):
        docopy = False
        for k in range(3):
            if im1Reg[i, j, k] != 0:
                docopy = True
        if docopy:
            img2[i, j] = im1Reg[i, j]

cv2.imshow('overlay', img2)  
cv2.waitKey(0)
