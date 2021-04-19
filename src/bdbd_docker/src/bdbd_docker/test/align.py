# align images from two different camera sources
# 2021-04-09: given a point on the fisheye, move pantilt to center on it.

import cv2
import numpy as np
import time
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from bdbd_common.imageRectify import ImageRectify
from bdbd_common.srv import SetPanTilt, SetPanTiltRequest

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

cvBridge = CvBridge()

import math
RADIANS_TO_DEGREES = 180. / math.pi
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

# Initiate detector
if TYPE == 'SIFT':
    detector = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.01, edgeThreshold=40, sigma=1.6)
    norm = cv2.NORM_L2
elif TYPE == 'ORB':
    detector = cv2.ORB_create()
    norm = cv2.NORM_HAMMING
else:
    print('invalid type')
    exit(1)

rospy.init_node('align')
rospy.loginfo('node started')
pantilt_srv = rospy.ServiceProxy('/bdbd/set_pan_tilt', SetPanTilt)
pan = 90
tilt = 45
result = pantilt_srv(SetPanTiltRequest(pan, tilt, False))

imageRectifyB = ImageRectify(CAMB, do_publish=False)
pcmB = imageRectifyB.pcm

#print('pcmA', pcmA.cx(), pcmA.cy(), pcmA.fx(), pcmA.fy())
#print('pcmB', pcmB.cx(), pcmB.cy(), pcmB.fx(), pcmB.fy())
#print('infoA', imageRectifyA.info_msg)
#print('infoB', imageRectifyB.info_msg)

targetB = (500, 200) # target point x, y value
imgB_msg = rospy.wait_for_message(CAMB_TOPIC, CompressedImage)
imgB = imageRectifyB.get(imgB_msg)
grayB = imgB if imgB.ndim == 2 else cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

# draw the center on the image and show
grayBtarget = grayB.copy()
grayBtarget = cv2.circle(grayBtarget, targetB, 20, 0, 2)
grayBtarget = cv2.circle(grayBtarget, targetB, 22, 255, 2)
cv2.imshow('target', grayBtarget)
cv2.waitKey(0)

# move the pan/tilt to roughly this location
anglesB = dot_angles(pcmB, targetB)
print(sstr('anglesB'))
pan = 90 - anglesB[0]
tilt = 45 + anglesB[1]
# add a false error
#pan = pan - 20
#tilt = tilt - 10
result = pantilt_srv(SetPanTiltRequest(pan, tilt, False))

#print(result)
imageRectifyA = ImageRectify(CAMA, do_publish=False)
pcmA = imageRectifyA.pcm
cv2.waitKey(0)

for count in range(10):
    imgA_msg = rospy.wait_for_message(CAMA_TOPIC, CompressedImage)
    imgA = imageRectifyA.get(imgA_msg)
    cv2.imshow('pantilt', imgA)
    cv2.waitKey(1000)

    grayA = imgA if imgA.ndim == 2 else cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
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
    cv2.imshow('scaled pantilt', grayA)
    cv2.waitKey(1000)

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
    cv2.waitKey(1000)

    from sklearn.cluster import DBSCAN as clust_method
    clust = clust_method(eps=50.0, min_samples=4)
    clusters = clust.fit(centers)

    # evaluate the clusters
    clusters_eval = []
    cmatches = []
    for i in range(len(clusters.labels_)):
        label = clusters.labels_[i]
        if label < 0:
            continue
        cmatches.append(matches[i])
        if label >= len(clusters_eval):
            cluster_eval = {
                'centers': [],
                'matches': []
            }
            clusters_eval.append(cluster_eval)
        else:
            cluster_eval = clusters_eval[label]
        cluster_eval['centers'].append(centers[i])
        cluster_eval['matches'].append(matches[i])
    for i in range(len(clusters_eval)):
        cluster_eval = clusters_eval[i]
        print('label {} count {}'.format(i, len(cluster_eval['centers'])))

    # pick the largest cluster
    fmatches = None
    fcenters = None
    best_length = 0
    for cluster_eval in clusters_eval:
        if best_length < len(cluster_eval['centers']):
            fmatches = cluster_eval['matches']
            fcenters = cluster_eval['centers']
            best_length = len(cluster_eval['centers'])
        

    #for member in inspect.getmembers(clusters):
    #    mname = member[0]
    #    if not mname.startswith('_'):
    #        print('clusters member name {} value {}'.format(mname, getattr(clusters, mname)))
    print('labels length: {} values: \n{}'.format(len(clusters.labels_), clusters.labels_))

    # filter matches
    '''
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
    '''
    # Draw good matches.
    print('best cluster size {}'.format(len(fmatches)))
    #img3 = cv2.drawMatches(grayA, kpA, grayB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv2.drawMatches(grayA, kpA, grayB, kpB, fmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img4 = cv2.drawMatches(grayA, kpA, grayB, kpB, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img5 = cv2.drawMatches(grayA, kpA, grayB, kpB, cmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('best matches', img3)
    cv2.imshow('all matches', img4)
    cv2.imshow('cluster matches', img5)
    cv2.waitKey(1000)
    print(sstr('fcenters'))
    (cBx, cBy) = np.mean(fcenters, axis=0).astype(int)
    #(cBy, cBx) = (int(np.mean(fcenters[:][0])), int(np.mean(fcenters[:][1])))
    print('mean_center', (cBx, cBy))

    # calculate correction to pantilt
    anglesBactual = dot_angles(pcmB, (cBx, cBy))

    # actual correction, with limits
    MAXDX = 5.0
    MAXDY = 3.0
    dBx = anglesB[0] - anglesBactual[0]
    dBy = anglesB[1] - anglesBactual[1]
    dBx = max(-MAXDX, min(MAXDX, dBx))
    dBy = max(-MAXDY, min(MAXDY, dBy))
    print(sstr('dBx dBy'))

    pan = pan - dBx
    tilt = tilt + dBy
    result = pantilt_srv(SetPanTiltRequest(pan, tilt, False))
    cv2.waitKey(0)

exit(0)

