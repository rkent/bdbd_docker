# determine depth of object located by pan tilt camera
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, SetPanTilt, SetPanTiltRequest
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import math

PANTILT_CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
SR305RGB_CAMERA = '/sr305/color/image_raw/compressed'
OBJECT_COUNT = 3 # number of objects to attempt to classify
RADIANS_TO_DEGREES = 180. / math.pi

# projected angles of image point p given pinhole camera model pcm 
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

def getCameraTopicBase(imageTopic):
    baseTopic = None
    offset = imageTopic.find('/image_')
    if offset >= 0:
        baseTopic = imageTopic[:offset]
    return baseTopic

def main():
    rospy.init_node('object_scan')

    od_srv = rospy.ServiceProxy('/bdbd/objectDetect', ObjectDetect)
    pantilt_srv = rospy.ServiceProxy('/bdbd/set_pan_tilt', SetPanTilt)
    #print(odr)
    print('waiting for objectDetect service')
    rospy.wait_for_service('/bdbd/objectDetect')
    print('got service')
    depth_info_msg = rospy.wait_for_message(getCameraTopicBase(SR305RGB_CAMERA) + '/camera_info', CameraInfo)
    print(depth_info_msg)
    pcm_depth = PinholeCameraModel()
    pcm_depth.fromCameraInfo(depth_info_msg)

    while not rospy.is_shutdown():

        # depth
        odr = ObjectDetectRequest()
        odr.image_topic = SR305RGB_CAMERA
        odr.max_detections = OBJECT_COUNT
        odr.min_threshold = 0.02
        odr.header.stamp = rospy.Time.now()

        # pantilt
        pdr = ObjectDetectRequest()
        pdr.image_topic = PANTILT_CAMERA
        pdr.max_detections = OBJECT_COUNT
        pdr.min_threshold = 0.02
        pdr.header.stamp = rospy.Time.now()

        try:
            response = od_srv(odr)
            print('top depth scores')
            for i in range(min(3, len(response.scores))):
                print('score {:6.3f} class {}'.format(response.scores[i], response.class_names[i]))

            # point the pantilt camera
            index = 0
            depth_object_center = ((response.xmin[index] + response.xmax[index])/2, (response.ymin[index]+ response.ymax[index])/2)
            (x_depth_angle, y_depth_angle) = dot_angles(pcm_depth, depth_object_center)
            print(sstr('x_depth_angle y_depth_angle'))
            pantilt_request = SetPanTiltRequest()
            pantilt_request.pan = 90. - x_depth_angle
            pantilt_request.tilt = 45 + y_depth_angle
            pantilt_srv(pantilt_request)
            pantilt_objects = od_srv(pdr)
            print('top depth scores')
            for i in range(min(3, len(pantilt_objects.scores))):
                print('score {:6.3f} class {}'.format(pantilt_objects.scores[i], pantilt_objects.class_names[i]))
 
        except rospy.ServiceException as exception:
            if rospy.is_shutdown():
                break
            rospy.logerr('Service Exception {}'.format(exception))
            break
        except Exception as exception:
            rospy.logerr('Exception {}'.format(exception))
            break

if __name__ == '__main__':
    main()
