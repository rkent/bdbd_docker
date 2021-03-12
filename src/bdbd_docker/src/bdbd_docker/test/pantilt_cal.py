# calibration of pantilt transformations using an Apriltag image seen by both a calibrated t265 and the pantilt.

import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
import rospy
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
from bdbd_common.msg import PanTilt
from bdbd_common.imageRectify import ImageRectify
import random
import time
import apriltag
import numpy as np
centers = []
ir_centers = []

cvBridge = CvBridge()
detector = apriltag.Detector()
pan = 90
tilt = 45

rospy.init_node('camera_align')
pantilt_pub = rospy.Publisher('/bdbd/pantilt', PanTilt, queue_size=1)
imageRectify = ImageRectify(topic_base='/t265/fisheye1', desired_encoding='passthrough')

while not rospy.is_shutdown():
    inputs = list(map(int, input('Enter pan tilt (negative to exit):').split()))
    if inputs[0] < 0:
        print('All done')
        break
    (pan, tilt) = inputs
    print(pan, tilt)
    pantilt_pub.publish(pan, tilt)

MAX_DPAN = 170
MIN_DPAN = 10
MAX_DTILT = 80
MIN_DTILT = 10
SETTLE_TIME = 1.0
TRIALS = 10
piTopic = '/bdbd/pantilt_camera/image_raw/compressed'
irTopic = '/t265/fisheye1/image_raw/compressed'

qrPub = rospy.Publisher('/camera_align/pi/image_color/compressed', CompressedImage, queue_size=1)

cvBridge = CvBridge()

while not rospy.is_shutdown():
    for trial in range(TRIALS):
        rpan = random.randint(MIN_DPAN, MAX_DPAN)
        rtilt = random.randint(MIN_DTILT, MAX_DTILT)
        pantilt_pub.publish(rpan, rtilt)
        rospy.sleep(SETTLE_TIME)
        pantilt_pub.publish(pan, tilt)
        rospy.sleep(SETTLE_TIME)
        
        pi_msg = messageSingle(piTopic, CompressedImage)
        imageC = cvBridge.compressed_imgmsg_to_cv2(pi_msg, desired_encoding='bgr8')
        image = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)
        results = detector.detect(image)

        ir_msg = messageSingle(irTopic, CompressedImage)
        ir_image = imageRectify.get(ir_msg)
        ir_results = detector.detect(ir_image)
        print('pi_center', results and results[0].center, 'ir center', ir_results and ir_results[0].center)
        if ir_results:
            ir_centers.append(ir_results[0].center)

        if results:
            result = results[0]
            centers.append(result.center)
            print('center', result.center)
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
    break

pi_means = np.mean(centers, axis=0)
pi_stds = np.std(centers, axis=0)
print('pi means, stds', pi_means, pi_stds)
ir_means = np.mean(ir_centers, axis=0)
ir_stds = np.std(ir_centers, axis=0)
print('ir means, stds', ir_means, ir_stds)

CAL_DATA = [
#   (IR center), (PI center), (pan, tilt)
    [ir_means, pi_means, [pan, tilt]]
]
print('CAL_DATA\n', CAL_DATA)
# from cam_cal.py
import math
import numpy as np
from image_geometry import PinholeCameraModel

RADIANS_TO_DEGREES = 180. / math.pi

pcm_t265 = PinholeCameraModel()
msg_t265 = messageSingle('/t265/fisheye1' + '/camera_info', CameraInfo)
pcm_t265.fromCameraInfo(msg_t265)

pcm_pi = PinholeCameraModel()
msg_pi = messageSingle('bdbd/pantilt_camera' + '/camera_info', CameraInfo)
pcm_pi.fromCameraInfo(msg_pi)

# projected angles of image point p given pinhole camera model pcm 
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

#calculation of angles from data
pi_angles = []
t265_angles = []
for cal_datum in CAL_DATA:
    print('cal_datum\n', cal_datum)
    t265_pixel = cal_datum[0]
    pi_pixel = cal_datum[1]
    pan_tilt = cal_datum[2]
    t265_ray = pcm_t265.projectPixelTo3dRay(t265_pixel)
    #pi_ray = pcm_pi.projectPixelTo3dRay(pi_pixel)
    #print(sstr('t265_pixel t265_ray pi_pixel pi_ray pan_tilt'))
    #print(sstr('t265_ray pi_ray'))

    # angles of x, y projections
    '''
    t265_xmag = math.sqrt(t265_ray[0]**2 + t265_ray[2]**2)
    t265_ymag = math.sqrt(t265_ray[1]**2 + t265_ray[2]**2)
    pi_xmag = math.sqrt(pi_ray[0]**2 + pi_ray[2]**2)
    pi_ymag = math.sqrt(pi_ray[1]**2 + pi_ray[2]**2)

    dotpx = pi_ray[0] * t265_ray[0] + pi_ray[2] * t265_ray[2]
    dotpy = pi_ray[1] * t265_ray[1] + pi_ray[2] * t265_ray[2]

    theta_x = RADIANS_TO_DEGREES * math.acos(dotpx / (t265_xmag * pi_xmag))
    theta_y = RADIANS_TO_DEGREES * math.acos(dotpy / (t265_ymag * pi_ymag))
    '''
    pa = dot_angles(pcm_pi, pi_pixel)
    pi_angles.append(pa)
    ta = dot_angles(pcm_t265, t265_pixel)
    t265_angles.append(ta)
    #print(sstr('pi_pixel pa t265_pixel ta pan_tilt'))

# error between pan_tilt angles and measured angles
x_dangles = []
y_dangles = []
for i in range(len(CAL_DATA)):
    cal_datum = CAL_DATA[i]
    pan_tilt = cal_datum[2]
    pi_angle = pi_angles[i]
    t265_angle = t265_angles[i]
    x_dangle = pi_angle[0] - t265_angle[0] - pan_tilt[0] + 90
    y_dangle = pi_angle[1] - t265_angle[1] + pan_tilt[1] - 45
    x_dangles.append(x_dangle)
    y_dangles.append(y_dangle)

    t265_pixel = cal_datum[0]
    pi_pixel = cal_datum[1]
    print('\n' + sstr('t265_pixel pi_pixel pan_tilt'))
    print(sstr('x_dangle y_dangle pi_angle t265_angle'))

print('publishing corrected pi to point at center')

(ir_x, ir_y) = dot_angles(pcm_t265, ir_means)
pantilt_pub.publish(90 - x_dangle - ir_x, 45 + y_dangle + ir_y)
