# test of the apriltag detect service
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, ObjectDetectResponse
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo

rospy.init_node('apriltag_test')
CAMERA = '/t265/fisheye1/image_raw/compressed'
OBJECT_COUNT = 12
od_srv = rospy.ServiceProxy('/bdbd/apriltagDetect', ObjectDetect)

while not rospy.is_shutdown():
    odr = ObjectDetectRequest()
    odr.image_topic = CAMERA
    odr.max_detections = OBJECT_COUNT
    odr.min_threshold = 0.02
    odr.header.stamp = rospy.Time.now()

    response = od_srv(odr)
    print('Found {} responses'.format(len(response.class_names)))
    for i in range(len(response.class_names)):
        center = ( (response.xmin[i] + response.xmax[i]) / 2., (response.ymin[i] + response.ymax[i]) / 2.)
        print('id {} center {}'.format(response.class_names[i], center))
