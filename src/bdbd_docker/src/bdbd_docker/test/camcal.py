# testing my understanding of image_geometry and camera calibration

import rospy
import cv2
import os
from sensor_msgs.msg import CameraInfo, CompressedImage
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
from bdbd_common.messageSingle import messageSingle
from bdbd_common.utils import fstr, sstr
import math
import numpy as np

RADIANS_TO_DEGREES = 180. / math.pi

# RKJ 2021-02-24 p 76
CAL_DATA = (
#   (IR center), (PI center), (pan, tilt)
    ((426, 406), (716, 436), (90,45)),
    ((426, 406), (250, 459), (70, 45)),
    ((426, 406), (1064, 397), (105, 45)),
    ((426, 406), (710, 207), (90,52)),
    ((426, 406), (225,215), (70,52)),
    ((426, 406), (1059,161), (105, 52)),
    ((426, 406), (704,529), (90,38)),
    ((426, 406), (253,535), (70,40)),
    ((426, 406), (1075,517), (105,40)),
    ((806, 406), (698,380), (40,45)),
    ((51,  398), (702,479), (140,45)),
    ((414, 591), (673,428), (90,70)),
    ((423, 133), (654,304), (90,10))
)

rospy.init_node('camcal')

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

mean_dangle = np.mean(np.array([x_dangles, y_dangles]), axis=1)
std_dangle = np.std(np.array([x_dangles, y_dangles]), axis=1)
print('\n' + sstr('mean_dangle std_dangle'))
