# align images from two different camera sources
# 2021-04-08: revising to only consider center of image, assuming known camera calibrations

import cv2
import numpy as np
import time
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from bdbd_common.imageRectify import ImageRectify
from image_geometry import PinholeCameraModel
import inspect
from bdbd_common.utils import fstr, sstr

CAMA = '/bdbd/pantilt_camera'
CAMB = '/t265/fisheye1'
CAMA_TOPIC = CAMA + '/image_raw/compressed'
CAMB_TOPIC = CAMB + '/image_raw/compressed'
TYPE='SIFT'
MAX_MATCHES = 50
MAX_BEST = 10
#TYPE='ORB'

cvBridge = CvBridge()

rospy.init_node('align')
rospy.loginfo('node started')
imageRectifyB = ImageRectify(CAMB, do_publish=True)
imageRectifyA = ImageRectify(CAMA, do_publish=True)
pcmA = imageRectifyA.pcm
pcmB = imageRectifyB.pcm

print('pcmA', pcmA.cx(), pcmA.cy(), pcmA.fx(), pcmA.fy())
print('pcmB', pcmB.cx(), pcmB.cy(), pcmB.fx(), pcmB.fy())
#print('infoA', imageRectifyA.info_msg)
#print('infoB', imageRectifyB.info_msg)

imgA_msg = rospy.wait_for_message(CAMA_TOPIC, CompressedImage)
imgB_msg = rospy.wait_for_message(CAMB_TOPIC, CompressedImage)
#imgA = cvBridge.compressed_imgmsg_to_cv2(imgA_msg)
#imgB = cvBridge.compressed_imgmsg_to_cv2(imgB_msg)
imgA = imageRectifyA.get(imgA_msg)
imgB = imageRectifyB.get(imgB_msg)

grayA = imgA if imgA.ndim == 2 else cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
grayB = imgB if imgB.ndim == 2 else cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# scale A to same pixel dimensions as B
xfactor = pcmA.fx() / pcmB.fx()
yfactor = pcmA.fy() / pcmB.fy()
height, width = grayA.shape[:2]
print('height, width before scaling', height, width)
width, height = int(width/xfactor), int(height/yfactor)
print('height, width after scaling', height, width)

cAx = int(pcmA.cx() / xfactor)
cAy = int(pcmA.cy() / yfactor)

grayA = cv2.resize(grayA, (width, height))
cv2.imshow('scaled', grayA)
cv2.imshow('pinhole', grayB)
cv2.waitKey(0)

#for member in inspect.getmembers(imgA):
#    print('model member {}'.format(member[0]))

# Initiate detector
if TYPE == 'SIFT':
    detector = cv2.SIFT_create(nfeatures=0, nOctaveLayers=6, contrastThreshold=0.01, edgeThreshold=80, sigma=3.0)
    norm = cv2.NORM_L2
elif TYPE == 'ORB':
    detector = cv2.ORB_create()
    norm = cv2.NORM_HAMMING
else:
    print('invalid type')
    exit(1)

start = time.time()
kpA, desA = detector.detectAndCompute(grayA, None)
kpB, desB = detector.detectAndCompute(grayB, None)

print('detect time', time.time() - start)
# create BFMatcher object
matcher = cv2.BFMatcher(norm, crossCheck=True)
# Match descriptors.
start = time.time()
matches = matcher.match(desA, desB)
matches = sorted(matches, key=lambda x: x.distance)
print('match time: {}, {} matches found'.format(time.time() - start, len(matches)))

if MAX_MATCHES and len(matches) > MAX_MATCHES:
    matches = matches[0:MAX_MATCHES]

# Extract location of good matches
pointsA = np.zeros((len(matches), 2), dtype=np.float32)
pointsB = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    pointsA[i, :] = kpA[match.queryIdx].pt
    pointsB[i, :] = kpB[match.trainIdx].pt

# calculate equivalent center for each match
# Use the CV coordinate system (x,y)
centers = []
print(sstr('cAx cAy'))
for i in range(len(matches)):
    pAx = pointsA[i, 0]
    pAy = pointsA[i, 1]
    pBx = pointsB[i, 0]
    pBy = pointsB[i, 1]
    cBx = pBx - (pAx - cAx)
    cBy = pBy - (pAy - cAy)
    centers.append((cBx, cBy))
    print(sstr('pAy pAx pBy pBx cBx cBy'))
print(sstr('centers'))

# plot the centers
grayBctr = grayB.copy()
for center in centers:
    icenter = (int(center[0]), int(center[1]))
    grayBctr = cv2.circle(grayBctr, icenter, 6, 255, 2)

cv2.imshow('centers', grayBctr)
cv2.waitKey(0)

from sklearn.cluster import DBSCAN as clust_method
clust = clust_method(eps=30.0, min_samples=3)
clusters = clust.fit(centers)

print('clusters', clusters)
#for member in inspect.getmembers(clusters):
#    print('clusters member {}'.format(member[0]))
print('labels length: {} values: \n{}'.format(len(clusters.labels_), clusters.labels_))

# filter matches
fmatches = []
fcenters = []
for i in range(len(clusters.labels_)):
    # just keep the MAX_BEST best matches
    if len(fmatches) >= MAX_BEST:
        break
    label = clusters.labels_[i]
    if label == 0:
        fmatches.append(matches[i])
        fcenters.append(centers[i])
# Draw good matches.
print('cluster 0 size {}'.format(len(fmatches)))
#img3 = cv2.drawMatches(grayA, kpA, grayB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = cv2.drawMatches(grayA, kpA, grayB, kpB, fmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('matches', img3)
cv2.waitKey(0)
print(sstr('fcenters'))
(cBx, cBy) = np.mean(fcenters, axis=0).astype(int)
#(cBy, cBx) = (int(np.mean(fcenters[:][0])), int(np.mean(fcenters[:][1])))
print('mean_center', (cBy, cBx))

# copy source to destination using center
img3 = grayB.copy()
for j in range(0, width):
    for i in range(0, height):
        iB = i + cBy - cAy
        jB = j + cBx - cAx
        img3[iB, jB] = grayA[i, j]

cv2.imshow('overlay', img3)
cv2.waitKey(0)

# Find homography
h, mask = cv2.findHomography(pointsA, pointsB, cv2.RANSAC)
print('h', h)

# Use homography
height, width = grayB.shape[0:2]
im1Reg = cv2.warpPerspective(grayA, h, (width, height))

cv2.imshow('warped', im1Reg)  
cv2.waitKey(0)

# copy warped source to destination
print('overlay warped image')
for j in range(0, width):
    for i in range(0, height):
        docopy = False
        for k in range(3):
            if im1Reg[i, j] != 0:
                docopy = True
        if docopy:
            grayB[i, j] = im1Reg[i, j]

cv2.imshow('overlay warp', grayB)  
cv2.waitKey(0)

exit(0)
