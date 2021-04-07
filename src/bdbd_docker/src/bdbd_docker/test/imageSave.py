# grabs an image from a ROS message, and saves it to a file.

import cv2
from sensor_msgs.msg import CompressedImage
from bdbd_common.messageSingle import messageSingle
from cv_bridge import CvBridge
import rospy

TOPIC = '/bdbd/pantilt_camera/image_raw/compressed'
FILE = '../data/pantilt.jpg'
cvBridge = CvBridge()

rospy.init_node('imageSave')

cam_msg = messageSingle(TOPIC, CompressedImage)
frame = cvBridge.compressed_imgmsg_to_cv2(cam_msg, desired_encoding='passthrough')
cv2.imwrite(FILE, frame)
