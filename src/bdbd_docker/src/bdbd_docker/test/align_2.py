# align images from two different camera sources

import cv2
import numpy as np
import time
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import inspect

cvBridge = CvBridge()
CAM1_TOPIC = '/bdbd/pantilt_camera/image_raw/compressed'
CAM2_TOPIC = '/t265/fisheye1/image_rect/compressed'
TYPE='SIFT'
MAX_MATCHES = 100
#TYPE='ORB'

def imgby2(img, factor=2):
    height, width = img.shape[:2]
    return cv2.resize(img, (width//factor, height//factor))

rospy.init_node('align')
img1_msg = rospy.wait_for_message(CAM1_TOPIC, CompressedImage)
img2_msg = rospy.wait_for_message(CAM2_TOPIC, CompressedImage)
img1 = imgby2(cvBridge.compressed_imgmsg_to_cv2(img1_msg, desired_encoding='bgr8'), factor=4)
img2 = cvBridge.compressed_imgmsg_to_cv2(img2_msg, desired_encoding='passthrough')

gray1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gray2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2 = img2

# Initiate detector
if TYPE == 'SIFT':
    detector = cv2.SIFT_create()
    norm = cv2.NORM_L2
elif TYPE == 'ORB':
    detector = cv2.ORB_create()
    norm = cv2.NORM_HAMMING
else:
    print('invalid type')
    exit(1)

start = time.time()
kp1, des1 = detector.detectAndCompute(gray1, None)
kp2, des2 = detector.detectAndCompute(gray2, None)

print('detect time', time.time() - start)
# create BFMatcher object
matcher = cv2.BFMatcher(norm, crossCheck=True)
# Match descriptors.
start = time.time()
matches = matcher.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
print('match time: {}, {} matches found'.format(time.time() - start, len(matches)))

if MAX_MATCHES and len(matches) > MAX_MATCHES:
    matches = matches[0:MAX_MATCHES]

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
print('h', h)

# Draw good matches.
print('good match count', len(matches))
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('matches', img3)
#sift = cv2.xfeatures2d.SIFT_create()
#sift = cv2.SIFT_create(nfeatures=50)
#kp = sift.detect(gray,None)
#img=cv2.drawKeypoints(gray, kp, img)
#cv2.imshow('SIFT', img)
  
cv2.waitKey(0) 

# Use homography
height, width = gray2.shape[0:2]
im1Reg = cv2.warpPerspective(gray1, h, (width, height))
print('img1, img2, im1Reg', img1.shape, img2.shape, im1Reg.shape)
#cv2.imwrite('../data/sr305_sift.jpg',img)
cv2.imshow('warped', im1Reg)
cv2.waitKey(0)

# copy warped source to destination
for j in range(0, width):
    for i in range(0, height):
        docopy = False
        for k in range(3):
            if im1Reg[i, j] != 0:
                docopy = True
        if docopy:
            img2[i, j] = im1Reg[i, j]

cv2.imshow('overlay', img2)  
cv2.waitKey(0)
