# t265 camera test node with rectify and (maybe later) stereo

import queue
import rospy
import cv2
import os
import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage
from cv_bridge import CvBridge
from bdbd_common.utils import fstr, sstr
from collections import deque
from image_geometry import PinholeCameraModel, StereoCameraModel
import time

from threading import Lock
frame_mutex = Lock()

try:
    from Queue import Queue
except:
    from queue import Queue

cvBridge = CvBridge()

class T265():
    def __init__(self):
        self.info1_sub = rospy.Subscriber('/t265/fisheye1/camera_info', CameraInfo, self.info1_cb, queue_size=1)
        self.info2_sub = rospy.Subscriber('/t265/fisheye2/camera_info', CameraInfo, self.info2_cb, queue_size=1)
        self.ir1_queue = deque([], 1)
        self.ir2_queue = deque([], 1)
        self.info1 = None
        self.info2 = None
        self.scm = None
        self.pc_right = StereoCameraModel()

    def stereo_model(self):
        # called from the message callbacks with mutex locked
        if not self.info1 or not self.info2:
            return False
        if self.scm:
            return True
        scm = StereoCameraModel()
        scm.fromCameraInfo(self.info1, self.info2)
        self.scm = scm
        return True

    def info1_cb(self, msg):
        rospy.loginfo('Got info1\n{}'.format(msg))
        frame_mutex.acquire()
        self.info1 = msg
        self.info1_sub.unregister()
        self.width_left = msg.width
        self.height_left = msg.height
        self.stereo_model()
        frame_mutex.release()

    def info2_cb(self, msg):
        rospy.loginfo('Got info2\n{}'.format(msg))
        frame_mutex.acquire()
        self.info2 = msg
        self.info2_sub.unregister()
        self.width_right = msg.width
        self.height_right = msg.height
        self.stereo_model()
        frame_mutex.release()

    def cam1_cb(self, msg):
        self.ir1_queue.append(msg)

    def cam2_cb(self, msg):
        self.ir2_queue.append(msg)

    def subscribe_cameras(self):
        self.ir1_sub = rospy.Subscriber('t265/fisheye1/image_raw/compressed', CompressedImage, self.cam1_cb, queue_size=1)
        self.ir2_sub = rospy.Subscriber('t265/fisheye2/image_raw/compressed', CompressedImage, self.cam2_cb, queue_size=1)

def main():
    rospy.init_node('test_t265')
    rospy.loginfo('{} starting with PID {}'.format(__name__, os.getpid()))

    t265 = T265()
    left_rect_pub = rospy.Publisher('/bdbd/t265/fisheye1/image_rect/compressed', CompressedImage, queue_size=1)
    right_rect_pub = rospy.Publisher('/bdbd/t265/fisheye2/image_rect/compressed', CompressedImage, queue_size=1)
    disparity_pub = rospy.Publisher('/bdbd/t265/disparity/compressed', CompressedImage, queue_size=1)

    # Configure the OpenCV stereo algorithm. See
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size = 5
    min_disp = 0
    # must be divisible by 16
    #num_disp = 112 - min_disp
    num_disp = 64 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   #blockSize = 16,
                                   blockSize = 4,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 50,
                                   #speckleWindowSize = 100,
                                   #speckleRange = 32
                                   speckleRange = 2
                                   )

    while True:
        frame_mutex.acquire()
        try:
            if t265.stereo_model():
                break
        finally:
            frame_mutex.release()
        rospy.sleep(.01)

    print('Q\n', t265.scm.Q)

    # Create an undistortion map for the left and right camera which applies the
    # rectification and undoes the camera distortion. This only has to be done
    # once

    K_left = t265.scm.left.K
    D_left = t265.scm.left.D[:4]
    R_left = t265.scm.left.R
    P_left = t265.scm.left.P
    K_right = t265.scm.right.K
    print(K_left, D_left, R_left, P_left)
    D_right = t265.scm.right.D[:4]
    R_right = t265.scm.right.R
    P_right = t265.scm.right.P
    m1type = cv2.CV_32FC1
    #stereo_size = ((4 * t265.width_left)//3, (4 * t265.height_left)//3)
    stereo_size = (t265.width_left, t265.height_left)
    
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)

    rospy.loginfo('Subscribing to cameras')
    t265.subscribe_cameras()

    while not rospy.is_shutdown():
        while not (len(t265.ir1_queue) and len(t265.ir2_queue)):
            rospy.sleep(.01)

        start = time.time()
        cam1_msg = t265.ir1_queue.pop()
        cam2_msg = t265.ir2_queue.pop()

        frame_left = cvBridge.compressed_imgmsg_to_cv2(cam1_msg, desired_encoding='bgr8')
        left_rect = cv2.remap(src=frame_left, map1=lm1, map2=lm2, interpolation = cv2.INTER_LINEAR)
        left_rect_msg = cvBridge.cv2_to_compressed_imgmsg(left_rect)
        left_rect_pub.publish(left_rect_msg)

        frame_right = cvBridge.compressed_imgmsg_to_cv2(cam2_msg, desired_encoding='bgr8')
        right_rect = cv2.remap(src=frame_right, map1=rm1, map2=rm2, interpolation = cv2.INTER_LINEAR)
        right_rect_msg = cvBridge.cv2_to_compressed_imgmsg(right_rect)
        right_rect_pub.publish(right_rect_msg)

        # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
        disparity = stereo.compute(frame_left, frame_right).astype(np.float32) / 16.0

        # re-crop just the valid part of the disparity
        disparity = disparity[:,max_disp:]

        # convert disparity to 0-255 and color it
        disp_vis = 255*(disparity - min_disp)/ num_disp
        disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
        disp_color_msg = cvBridge.cv2_to_compressed_imgmsg(disp_vis)
        disparity_pub.publish(disp_color_msg)
        #color_image = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)

        rospy.loginfo('Processing time {}'.format(time.time() - start))
        #break


if __name__ == '__main__':
    main()
