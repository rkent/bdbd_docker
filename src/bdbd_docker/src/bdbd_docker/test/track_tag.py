# track a tag on IR with the pan/tilt

import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
import rospy
from bdbd_common.messageSingle import messageSingle
from bdbd_common.msg import PanTilt
from bdbd_common.imageRectify import ImageRectify
from image_geometry import PinholeCameraModel

import apriltag
import math

RADIANS_TO_DEGREES = 180. / math.pi

cvBridge = CvBridge()
detector = apriltag.Detector()
x_dangle = -9.
y_dangle = 7.

# projected angles of image point p given pinhole camera model pcm 
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

rospy.init_node('track_tag')
pantilt_pub = rospy.Publisher('/bdbd/pantilt', PanTilt, queue_size=1)
imageRectify = ImageRectify(topic_base='/t265/fisheye1', desired_encoding='passthrough')

irTopic = '/t265/fisheye1/image_raw/compressed'

cvBridge = CvBridge()

pcm_t265 = PinholeCameraModel()
msg_t265 = messageSingle('/t265/fisheye1' + '/camera_info', CameraInfo)
pcm_t265.fromCameraInfo(msg_t265)

while not rospy.is_shutdown():
    ir_msg = messageSingle(irTopic, CompressedImage)
    ir_image = imageRectify.get(ir_msg)
    ir_results = detector.detect(ir_image)
    if ir_results:
        ir_center = ir_results[0].center
        print('ir_center', ir_center)
    (ir_x, ir_y) = dot_angles(pcm_t265, ir_center)
    pantilt_pub.publish(90 - x_dangle - ir_x, 45 + y_dangle + ir_y)

    #rospy.sleep(1)
