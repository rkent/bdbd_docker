# pan tilt camera chases a tag displayed on the t265

import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
import rospy
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
from bdbd_common.msg import PanTilt
import random
import time
import apriltag
from image_geometry import PinholeCameraModel
import math
import numpy as np

RADIANS_TO_DEGREES = 180. / math.pi

# projected angles of image point p given pinhole camera model pcm 
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

centers = []
ir_centers = []

cvBridge = CvBridge()
detector = apriltag.Detector()

rospy.init_node('chase_tag')

print('get it camera info')
pcm_t265 = PinholeCameraModel()
msg_t265 = messageSingle('/t265/fisheye1' + '/camera_info', CameraInfo)
pcm_t265.fromCameraInfo(msg_t265)

print('Get pi camera info')
pcm_pi = PinholeCameraModel()
msg_pi = messageSingle('bdbd/pantilt_camera' + '/camera_info', CameraInfo)
pcm_pi.fromCameraInfo(msg_pi)

pantilt_pub = rospy.Publisher('/bdbd/pantilt', PanTilt, queue_size=1)
'''
while not rospy.is_shutdown():
    inputs = list(map(int, input('Enter pan tilt (negative to exit):').split()))
    if inputs[0] < 0:
        print('All done')
        break
    (pan, tilt) = inputs
    print(pan, tilt)
    pantilt_pub.publish(pan, tilt)
'''

MAX_DPAN = 170
MIN_DPAN = 10
MAX_DTILT = 80
MIN_DTILT = 10
SETTLE_TIME = 1.0
piTopic = '/bdbd/pantilt_camera/image_raw/compressed'
irTopic = '/t265/fisheye1/image_rect/compressed'

qrPub = rospy.Publisher('/camera_align/pi/image_color/compressed', CompressedImage, queue_size=1)

while not rospy.is_shutdown():
    print('loop')
    '''
    rpan = random.randint(MIN_DPAN, MAX_DPAN)
    rtilt = random.randint(MIN_DTILT, MAX_DTILT)
    pantilt_pub.publish(rpan, rtilt)
    rospy.sleep(SETTLE_TIME)
    pantilt_pub.publish(pan, tilt)
    rospy.sleep(SETTLE_TIME)
    '''

    # locate the tag on the IR sensor
    ir_msg = messageSingle(irTopic, CompressedImage)
    ir_image = cvBridge.compressed_imgmsg_to_cv2(ir_msg, desired_encoding='passthrough')
    ir_results = detector.detect(ir_image)
    print('ir center', ir_results and ir_results[0].center)
    if not ir_results:
        continue

    # calculate the required angle on the pi sensor
    t265_angle = dot_angles(pcm_t265, ir_results[0].center)
    print(sstr('t265_angle'))

    pan = 90 - t265_angle[0] - 2.8
    #tilt = 45 + t265_angle[1] - 2.3
    tilt = 45 + t265_angle[1] + 2.3
    print(sstr('pan tilt'))
    pantilt_pub.publish(pan, tilt)
    rospy.sleep(SETTLE_TIME)

    pi_msg = messageSingle(piTopic, CompressedImage)
    imageC = cvBridge.compressed_imgmsg_to_cv2(pi_msg, desired_encoding='bgr8')
    image = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)
    results = detector.detect(image)

    if results:
        result = results[0]
        pi_angle = dot_angles(pcm_pi, result.center)
        print(sstr('result.center pi_angle'))
        corners = result.corners
        color = (127,255, 127)
        thick = 4
        #print(type(corners0), corners0, type(corners2), corners2)
        for i in range(4):
            p0 = tuple(corners[i].astype(int))
            p1 = tuple(corners[(i+1) % 4].astype(int))
            imageC = cv2.line(imageC, p0, p1, color, thick)
        #imageC = cv2.rectangle(imageC, corners0, corners2, (127,255, 127), 4)
        image_msg = cvBridge.cv2_to_compressed_imgmsg(imageC)
        qrPub.publish(image_msg)

    rospy.sleep(1)

pantilt_pub.publish(90, 45)
