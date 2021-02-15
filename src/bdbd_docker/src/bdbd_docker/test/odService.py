import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, ObjectDetectResponse
from bdbd_common.utils import fstr, sstr

#CAMERA = '/bdbd/pi/image_raw/compressed'
#CAMERA = '/sr305/color/image_raw/compressed'
#CAMERA = '/sr305/infra1/image_rect_raw/compressed'
CAMERA = '/t265/fisheye1/image_raw/compressed'
rospy.init_node('test')

od_srv = rospy.ServiceProxy('/bdbd/objectDetect', ObjectDetect)
odr = ObjectDetectRequest()
odr.image_topic = CAMERA
odr.max_detections = 10
odr.min_threshold = 0.2
odr.header.stamp = rospy.Time.now()
print(odr)
print('waiting for service')
rospy.wait_for_service('/bdbd/objectDetect')
print('sending request')
response = od_srv(odr)
print(response)
