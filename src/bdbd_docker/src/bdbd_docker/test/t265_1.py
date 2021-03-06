# t265 camera test node with rectify

import rospy
import cv2
import os
from sensor_msgs.msg import CameraInfo, CompressedImage
from cv_bridge import CvBridge
from collections import deque
from image_geometry import PinholeCameraModel, StereoCameraModel
import time

cvBridge = CvBridge()

class T265():
    def __init__(self, topic_base='/t265/fisheye1'):
        self.info_sub = rospy.Subscriber(topic_base + '/camera_info', CameraInfo, self.info_cb, queue_size=1)
        self.ir_queue = deque([], 1)
        self.info = None
        self.topic_base = topic_base
        self.rect_pub = rospy.Publisher(topic_base + '/image_rect/compressed', CompressedImage, queue_size=1)
        rospy.loginfo('Getting camera_info for ' + topic_base)

    def camera_model(self, msg):
        # call with mutex locked
        pcm = PinholeCameraModel()
        pcm.fromCameraInfo(msg)
        (self.m1, self.m2) = cv2.fisheye.initUndistortRectifyMap(
            pcm.K, pcm.D[:4], pcm.R, pcm.P,
            (msg.width, msg.height), cv2.CV_32FC1
        )
        self.subscribe_camera()

    def info_cb(self, msg):
        rospy.loginfo('Got info\n{}'.format(msg))
        self.info_sub.unregister()
        self.info_sub = None
        self.camera_model(msg)

    def cam_cb(self, cam_msg):
        frame = cvBridge.compressed_imgmsg_to_cv2(cam_msg, desired_encoding='bgr8')
        img_rect = cv2.remap(src=frame, map1=self.m1, map2=self.m2, interpolation = cv2.INTER_LINEAR)
        img_rect_msg = cvBridge.cv2_to_compressed_imgmsg(img_rect)
        self.rect_pub.publish(img_rect_msg)

    def subscribe_camera(self):
        rospy.loginfo('Subscribing to camera')
        self.ir_sub = rospy.Subscriber(self.topic_base + '/image_raw/compressed', CompressedImage, self.cam_cb, queue_size=1)

def main():
    rospy.init_node('test_t265')
    rospy.loginfo('{} starting with PID {}'.format(__name__, os.getpid()))

    t265 = T265()

    rospy.spin()

if __name__ == '__main__':
    main()
