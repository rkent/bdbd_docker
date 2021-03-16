from numpy.core.fromnumeric import resize
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, ObjectDetectResponse
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
from bdbd_docker.libpy.objectClassifier import ObjectClassifier
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rospy import ServiceException
import cv2
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU

CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
#CAMERA = '/sr305/color/image_raw/compressed'
#CAMERA = '/sr305/infra1/image_rect_raw/compressed'
#CAMERA = '/t265/fisheye1/image_raw/compressed'
#CAMERA = '/t265/fisheye1/image_raw/compressed'

OBJECT_COUNT = 12 # number of objects to attempt to classify
RATE = 1.0

def main():
    rospy.init_node('object_classify')

    cvBridge = CvBridge()
    objectClassifier = ObjectClassifier()

    try:
        image_compressed_pub = rospy.Publisher('/object_classify/image_raw/compressed', CompressedImage, queue_size=1)
        image_objects_pub = rospy.Publisher('/object_classify/objects/image_raw/compressed', CompressedImage, queue_size=1)

        od_srv = rospy.ServiceProxy('/bdbd/objectDetect', ObjectDetect)
        odr = ObjectDetectRequest()
        odr.image_topic = CAMERA
        odr.max_detections = OBJECT_COUNT
        odr.min_threshold = 0.02
        odr.header.stamp = rospy.Time.now()
        #print(odr)
        print('waiting for service')
        rospy.wait_for_service('/bdbd/objectDetect')
        print('got service')

        rate = rospy.Rate(RATE)
        while not rospy.is_shutdown():
            try:
                print('wait for service response')
                response = od_srv(odr)
                print('got service response')
                image_np = cvBridge.compressed_imgmsg_to_cv2(response.image, desired_encoding='bgr8')
                (labels, object_scores, object_names, object_images) = \
                    objectClassifier.classify(
                        image_np, response.class_names, response.scores,
                        response.xmin, response.xmax, response.ymin, response.ymax
                    )
                combined_image = objectClassifier.annotate_image(labels, object_scores, object_names, object_images)
                combined_image_msg = cvBridge.cv2_to_compressed_imgmsg(combined_image)
                image_msg = cvBridge.cv2_to_compressed_imgmsg(image_np)
                image_compressed_pub.publish(image_msg)

                image_objects_pub.publish(combined_image_msg)

            except ServiceException:
                rospy.logwarn('Service Exception. Waiting for service')
                try:
                    rospy.wait_for_service('/bdbd/objectDetect')
                except:
                    print('Exception while waiting for service, exiting')
                    break
                continue
            rate.sleep()
    finally:
        objectClassifier.clear()

if __name__ == '__main__':
    main()
