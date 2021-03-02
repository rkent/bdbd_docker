from numpy.core.fromnumeric import resize
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, ObjectDetectResponse
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rospy import ServiceException
import cv2
import time
import os
from numba import cuda

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # stop grabbing all memory

# adapted from https://www.tutorialspoint.com/keras/keras_applications.htm
import tensorflow.keras as keras 
import numpy as np 
from tensorflow.keras.applications.imagenet_utils import decode_predictions 
#from tensorflow.keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input
from tensorflow.keras.applications.xception import Xception, decode_predictions, preprocess_input

#Load the model 
#resnet_model = ResNet50V2(weights = 'imagenet') 
model = Xception(weights = 'imagenet')

#CAMERA = '/bdbd/pi/image_raw/compressed'
#CAMERA = '/sr305/color/image_raw/compressed'
#CAMERA = '/sr305/infra1/image_rect_raw/compressed'
#CAMERA = '/t265/fisheye1/image_raw/compressed'
CAMERA = '/t265/fisheye1/image_rect/compressed'

OBJECT_COUNT = 12 # number of objects to attempt to classify
REPORT_COUNT = 9 # number of objects to report
#XSIZE = 224 # size of classifier image
#YSIZE = 224
XSIZE = 299
YSIZE = 299
RATE = 1.0
rospy.init_node('test')

cvBridge = CvBridge()

image_compressed_pub = rospy.Publisher('/odService/image_raw/compressed', CompressedImage, queue_size=1)
image_objects_pub = rospy.Publisher('/odService/objects/image_raw/compressed', CompressedImage, queue_size=1)

od_srv = rospy.ServiceProxy('/bdbd/objectDetect', ObjectDetect)
odr = ObjectDetectRequest()
odr.image_topic = CAMERA
odr.max_detections = OBJECT_COUNT
odr.min_threshold = 0.02
odr.header.stamp = rospy.Time.now()
#print(odr)
print('waiting for service')
rospy.wait_for_service('/bdbd/objectDetect')

from image_geometry import PinholeCameraModel
class T265():
    def __init__(self, topic_base='/t265/fisheye1'):
        #self.info_sub = rospy.Subscriber(topic_base + '/camera_info', CameraInfo, self.info_cb, queue_size=1)
        self.topic_base = topic_base
        rospy.loginfo('Getting camera_info for ' + topic_base)
        pcm = PinholeCameraModel()
        pcm.fromCameraInfo(messageSingle(topic_base + '/camera_info', CameraInfo))
        self.camera_model = pcm

    def info_cb(self, msg):
        #rospy.loginfo('Got info\n{}'.format(msg))
        rospy.loginfo('Get info')
        self.info_sub.unregister()
        self.info_sub = None
        pcm = PinholeCameraModel()
        pcm.fromCameraInfo(msg)
        self.camera_model = pcm

def resize_with_pad(im, desired_size=(YSIZE, XSIZE), color=[0, 0, 0]):
    # adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = max(*desired_size)/max(*old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size[1] - new_size[1]
    delta_h = desired_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

rate = rospy.Rate(RATE)
try:
    t265 = T265()
    pi_model = PinholeCameraModel()
    pi_model.fromCameraInfo(messageSingle('bdbd/pantilt_camera/camera_info', CameraInfo))
    while not rospy.is_shutdown():
        # get the camera model

        try:
            response = od_srv(odr)
            #if t265.camera_model:
            #    pcm = t265.camera_model
            #    print('K\n{}\nD\n{}\nR\n{}\nP\n{}'.format(pcm.K, pcm.D[:4], pcm.R, pcm.P))

            # Process frame from camera
            image_np = cvBridge.compressed_imgmsg_to_cv2(response.image, desired_encoding='bgr8')
            #print('image_np array shape', image_np.shape)
            # gray to color if needed
            if len(image_np.shape) == 2 or image_np.shape[2] != 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                #print('image_np array re-shaped to', image_np.shape)
            # find something that is not a book

            xmin = 0
            ymin = 0
            xmax, ymax, _ = image_np.shape
            object_images = []
            object_scores = []
            object_names = []
            class_names = response.class_names
            if len(class_names) == 0:
                continue
            for index in range(len(response.class_names)):
                if len(object_images) >= REPORT_COUNT:
                    break
                if response.class_names[index] == 'book':
                    continue
                xmin = response.xmin[index]
                ymin = response.ymin[index]
                xmax = response.xmax[index]
                ymax = response.ymax[index]
                nname = response.class_names[index]
                print(nname)

                # project point
                t265_ray = t265.camera_model.projectPixelTo3dRay(((xmin + xmax)//2, (ymin + ymax)//2))
                t265_point = t265.camera_model.project3dToPixel(t265_ray)
                pi_point = pi_model.project3dToPixel(t265_ray)
                # sstr debug
                '''
                import inspect

                vs = 't265_ray t265_point pi_point name'
                def reach(name) :
                    # https://stackoverflow.com/questions/15608987/access-variables-of-caller-function-in-python
                    for f in inspect.stack() :
                        if name in f[0].f_locals : return f[0].f_locals[name]
                    return None 
                dd = dict()
                for v in vs.split(' '):
                    dd[v] = reach(v)
                print(fstr(dd))
                '''

                print(sstr('t265_ray t265_point pi_point nname'))
                # with padding
                image_slice = image_np[ymin:ymax, xmin:xmax, :]

                # without padding
                '''
                (ysize, xsize, _) = image_np.shape
                square = min(xsize, ysize, max(ymax-ymin+1, xmax-xmin+1))
                if ymin + square > ysize:
                    ymin = max(0, ymax - square)
                if xmin + square > xsize:
                    xmin = max(0, xmax - square)
                image_slice = image_np[ymin:(ymin + square), xmin:(xmin + square), :]
                print(sstr('ymin xmin square') + fstr({'image_slice.shape': image_slice.shape}))
                '''
                image_224 = resize_with_pad(image_slice, (XSIZE, YSIZE))

                object_images.append(image_224)
                object_scores.append(response.scores[index])
                object_names.append(response.class_names[index])

            print(fstr({'object detector classes': object_names}))
            count = min(9, len(object_images))
            # Convert the image / images into batch format
            #image_batch = np.expand_dims(object_images[0], axis = 0)
            image_batch = np.array([*object_images])
            start = time.time()
            processed_image = preprocess_input(image_batch.copy())
            # get the predicted probabilities for each class 
            predictions = model.predict(processed_image) 

            # convert the probabilities to class labels 
            labels = decode_predictions(predictions)
            print('classification time {:6.3f}'.format(time.time() - start))
            for i in range(len(labels)):
                print(labels[i][0])
            #id, name, score = labels[0][0]
            #print('imagenet identified {} with score {}'.format(name, score))

            xpositions = (1, 2, 2, 2, 3, 3, 3, 3, 3)[count - 1]
            ypositions = (1, 1, 2, 2, 2, 2, 3, 3, 3)[count - 1]
            combined_image = np.zeros((YSIZE * ypositions, XSIZE * xpositions, 3), dtype=np.uint8)
            #print('combined_image shape', combined_image.shape)
            for i in range(count):
                xpos = i % xpositions
                ypos = i // xpositions
                #if object_scores[i] > labels[i][0][2]:
                color1 = (128, 255, 255)
                score1 = object_scores[i]
                name1 = object_names[i]
                text1 = "{:5.2f} {}".format(score1, name1)
                #else:
                color2 = (255, 128, 255)
                score2 = labels[i][0][2]
                name2 = labels[i][0][1]
                text2 = "{:5.2f} {}".format(score2, name2)
                display_image = cv2.putText(object_images[i], text1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, color1, 2, cv2.LINE_AA)
                display_image = cv2.putText(display_image, text2, (5, XSIZE-20), cv2.FONT_HERSHEY_SIMPLEX, .8, color2, 2, cv2.LINE_AA)
                combined_image[YSIZE * ypos:YSIZE * (ypos + 1), XSIZE * xpos: XSIZE * (xpos + 1), :] = object_images[i]

            combined_image_msg = cvBridge.cv2_to_compressed_imgmsg(combined_image)
            image_compressed_pub.publish(combined_image_msg)

            bordered_image_msg = cvBridge.cv2_to_compressed_imgmsg(image_np)
            image_objects_pub.publish(bordered_image_msg)

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
    # clear memory
    cuda.select_device(0)
    cuda.close()
    print('CUDA memory release: GPU0')
