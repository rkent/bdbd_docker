# locate and identify objects in fisheye FOV using pantilt camera
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, SetPanTilt, SetPanTiltRequest
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo
from bdbd_docker.libpy.objectClassifier import ObjectClassifier, best_label, sort_labels
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import math
import traceback
from bdbd_common.doerRequest import DoerRequest
dr = DoerRequest()

PANTILT_CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
FISHEYE_CAMERA = '/t265/fisheye1/image_raw/compressed'
OBJECT_COUNT = 10 # number of objects to attempt to classify
RADIANS_TO_DEGREES = 180. / math.pi
MIN_SIZE = 100
MAX_SIZE = 10000
MIN_DETECT_SCORE = 0.15
MAX_CLOSE_ANGLE = 30
MAX_FISHEYE_OBJECTS = 4
ASPECT_LIMIT = 2.0
DO_PAD = True
MIN_CLASSIFY_SCORE = 0.70

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
    cvBridge = CvBridge()
    objectClassifier = ObjectClassifier()

    od_srv = dr.ServiceProxy('/bdbd/objectDetect', ObjectDetect, timeout=60.0)
    pantilt_srv = dr.ServiceProxy('/bdbd/set_pan_tilt', SetPanTilt)
    image_compressed_pub = rospy.Publisher('/object_scan/image_raw/compressed', CompressedImage, queue_size=1)
    image_objects_pub = rospy.Publisher('/object_scan/objects/image_raw/compressed', CompressedImage, queue_size=1)
    object_image_pub = rospy.Publisher('/object_scan/object/image_raw/compressed', CompressedImage, queue_size=1)

    #print(odr)
    #print('waiting for objectDetect service', flush=True, end='... ')
    #rospy.wait_for_service('/bdbd/objectDetect')
    #print('done')
    fisheye_info_msg = dr.wait_for_message(getCameraTopicBase(FISHEYE_CAMERA) + '/camera_info', CameraInfo)
    pcm_fisheye = PinholeCameraModel()
    pcm_fisheye.fromCameraInfo(fisheye_info_msg)

    while not rospy.is_shutdown():
        try:
            # Fisheye image objects
            odr = ObjectDetectRequest()
            odr.image_topic = FISHEYE_CAMERA
            odr.max_detections = OBJECT_COUNT
            odr.min_threshold = MIN_DETECT_SCORE
            odr.header.stamp = rospy.Time.now()

            print('wait for service response', end='... ', flush=True)
            fe_response = od_srv(odr)
            print('done')
            best_score = 0.0
            best_i = 0
            reject_names = ['book']
            object_indicies = []
            for i in range(len(fe_response.scores)):
                name = fe_response.class_names[i]
                print('name {} score {}'.format(name, fe_response.scores[i]))
                fisheye_object_center = ((fe_response.xmin[i] + fe_response.xmax[i])/2, (fe_response.ymin[i]+ fe_response.ymax[i])/2)
                xd = fe_response.xmax[i] - fe_response.xmin[i]
                yd = fe_response.ymax[i] - fe_response.ymin[i]
                (x_fisheye_angle, y_fisheye_angle) = dot_angles(pcm_fisheye, fisheye_object_center)
                if name in reject_names:
                    continue
                if max(xd, yd) > MAX_SIZE:
                    continue
                if y_fisheye_angle > MAX_CLOSE_ANGLE:
                    continue
                score = fe_response.scores[i]
                if score > best_score:
                    best_score = score
                    best_i = i
                object_indicies.append(i)
                print(sstr('i score name fisheye_object_center xd yd x_fisheye_angle y_fisheye_angle'))
                if len(object_indicies) >= MAX_FISHEYE_OBJECTS:
                    break

            # point the pantilt camera
            for index in object_indicies:
                fisheye_object_center = ((fe_response.xmin[index] + fe_response.xmax[index])/2, (fe_response.ymin[index]+ fe_response.ymax[index])/2)
                (x_fisheye_angle, y_fisheye_angle) = dot_angles(pcm_fisheye, fisheye_object_center)
                print(sstr('index fisheye_object_center x_fisheye_angle y_fisheye_angle'))
                pantilt_request = SetPanTiltRequest()
                pantilt_request.pan = 90. - x_fisheye_angle
                pantilt_request.tilt = 45 + y_fisheye_angle
                print('waiting for pantilt motion', flush=True, end='... ')
                pantilt_srv(pantilt_request)
                print('done')

                # get the pantilt image
                pantilt_msg = dr.wait_for_message(PANTILT_CAMERA, CompressedImage)

                # locate objects in this image
                # Pantilt image objects
                odr = ObjectDetectRequest()
                odr.image = pantilt_msg
                odr.max_detections = OBJECT_COUNT
                odr.min_threshold = MIN_DETECT_SCORE
                odr.header.stamp = rospy.Time.now()
                print('wait for pantilt detection response ...', flush=True, end=' ')
                pt_response = od_srv(odr)
                print('done')
    
                # extract regions for object classification
                image_np = cvBridge.compressed_imgmsg_to_cv2(pantilt_msg, desired_encoding='bgr8')
                (MAXY, MAXX, _) = image_np.shape
                (labels, object_scores, object_names, object_images) = \
                    objectClassifier.classify(
                        image_np, pt_response.class_names, pt_response.scores,
                        pt_response.xmin, pt_response.xmax, pt_response.ymin, pt_response.ymax,
                        aspect_limit=ASPECT_LIMIT, do_pad=DO_PAD
                    )
                print('pantilt: ' + sstr('object_scores object_names'))
                combined_image = objectClassifier.annotate_image(labels, object_scores, object_names, object_images)
                if (image_compressed_pub.get_num_connections() > 0):
                    image_msg = cvBridge.cv2_to_compressed_imgmsg(image_np)
                    image_compressed_pub.publish(image_msg)
                if (image_objects_pub.get_num_connections() > 0):
                    combined_image_msg = cvBridge.cv2_to_compressed_imgmsg(combined_image)
                    image_objects_pub.publish(combined_image_msg)

                (best_i, best_score, best_name) = best_label(labels)
                sorted_labels = sort_labels(labels)

                # choose regions to examine
                for label in sorted_labels:
                    score = label['score']
                    name = label['name']
                    if name in reject_names:
                        continue
                    if score < MIN_DETECT_SCORE:
                        continue
                    best_i = label['index']
                    xmin = pt_response.xmin[best_i]
                    xmax = pt_response.xmax[best_i]
                    xd = xmax - xmin
                    ymin = pt_response.ymin[best_i]
                    ymax = pt_response.ymax[best_i]
                    yd = ymax - ymin                   
                    if xd < MIN_SIZE and yd < MIN_SIZE:
                        continue
                    xd = max(MIN_SIZE, xd)
                    yd = max(MIN_SIZE, yd)
                    xc = (xmin + xmax) / 2
                    yc = (ymin + ymax) / 2

                    print('detect ' + sstr('score name xc yc xd yd '))

                    # refine the object area
                    # refine an image with multiple calls to classifier
                    
                    delta_factor = 0.20
                    for count in range(2):
                        (best_score, best_name, best_image, xd, yd, xc, yc, combined_image) = \
                            objectClassifier.refine_class(
                                image_np, xd, yd, xc, yc,
                                delta_factor=delta_factor,
                                first_name=name, first_score=score)
                        print('refine ' + sstr('best_score best_name xc yc xd yd'))
                        if (image_objects_pub.get_num_connections() > 0):
                            combined_image_msg = cvBridge.cv2_to_compressed_imgmsg(combined_image)
                            image_objects_pub.publish(combined_image_msg)
                        delta_factor /= 2
                    if (best_score > MIN_CLASSIFY_SCORE):
                        if (object_image_pub.get_num_connections() > 0):
                            object_image_msg = cvBridge.cv2_to_compressed_imgmsg(best_image)
                            object_image_pub.publish(object_image_msg)
                    
        except rospy.ServiceException as exception:
            if rospy.is_shutdown():
                break
            rospy.logerr('Service Exception {}'.format(exception))
            break
        except Exception as exception:
            rospy.logerr('Exception {}'.format(exception))
            rospy.logwarn(traceback.format_exc())
            break
    objectClassifier.clear()

if __name__ == '__main__':
    main()
