#!/usr/bin/env python
# coding: utf-8

import os
import rospy
from collections import deque
from os.path import expanduser
import tarfile
import urllib.request
import cv2
import numpy as np
import time
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import traceback
from bdbd_common.srv import ObjectDetect, ObjectDetectResponse
from queue import Queue

# adapted from
#  https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/object_detection_camera.html#sphx-glr-auto-examples-object-detection-camera-py

#CAMERA_TOPIC = '/bdbd/pi/image_raw/compressed'
CAMERA_TOPIC = ''
#CAMERA_TOPIC = '/t265/fisheye1/image_raw/compressed'
SAMPLE_NAME = 'od_sample.jpg'
LIMIT_GROWTH = True
USE_GPU = False
DATA_DIR = os.path.join(expanduser('~'), 'bdbd', 'data')
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', SAMPLE_NAME)
MODEL_DATE = '20200711'
MODELS_DIR = os.path.join(DATA_DIR, 'models')
MODELS_DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/tf2/'
#MODEL_NAME = 'efficientdet_d3_coco17_tpu-32'
#MODEL_NAME = 'efficientdet_d5_coco17_tpu-32'
#MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
#MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
#MODEL_NAME = 'centernet_resnet50_v2_512x512_coco17_tpu-8'
#MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = os.path.join(MODELS_DIR, LABEL_FILENAME)
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
LABEL_ID_OFFSET = 1

for dir in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(dir):
        os.mkdir(dir)

class ObjectDetectClass():
    def __init__(self, use_gpu, model_level):
        self.ensure_model(model_level)
        self.load_tf(use_gpu)
        self.make_detect_fn()
        # these are delayed because they might prematurely load tf
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils

        self.category_index = label_map_util.create_category_index_from_labelmap(
            PATH_TO_LABELS, use_display_name=True)
        print(fstr({'category_index': self.category_index}))

        self.label_map_util = label_map_util
        self.viz_utils = viz_utils
        self.cv_bridge = CvBridge()
        self.image_compressed_pub = rospy.Publisher('object_detect/image_raw/compressed', CompressedImage, queue_size=1)

    # Download and extract model
    def ensure_model(self, model_level):

        from object_detection.utils import config_util

        model_name = 'efficientdet_d' + str(model_level) + '_coco17_tpu-32'
        rospy.loginfo('Using object_detection model {}'.format(model_name))
        path_to_cfg = os.path.join(MODELS_DIR, os.path.join(model_name, 'pipeline.config'))
        self.path_to_ckpt = os.path.join(MODELS_DIR, os.path.join(model_name, 'checkpoint/'))
        tar_filename = model_name + '.tar.gz'
        model_download_link = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + tar_filename
        path_to_model_tar = os.path.join(MODELS_DIR, tar_filename)

        if not os.path.exists(self.path_to_ckpt):
            print('Downloading model. This may take a while... ', end='')
            urllib.request.urlretrieve(model_download_link, path_to_model_tar)
            tar_file = tarfile.open(path_to_model_tar)
            tar_file.extractall(MODELS_DIR)
            tar_file.close()
            os.remove(path_to_model_tar)
            print('Done')

        # Download labels file
        if not os.path.exists(PATH_TO_LABELS):
            print('Downloading label file... ', end='')
            urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
            print('Done')
        
        self.configs = config_util.get_configs_from_pipeline_file(path_to_cfg)


    def load_tf(self, use_gpu):
        #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU

        if LIMIT_GROWTH:
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # stop grabbing all memory
        import tensorflow as tf
        self.tf = tf

    def make_detect_fn(self):
        tf = self.tf
        from object_detection.builders import model_builder

        #tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

        # Load pipeline config and build a detection model
        model_config = self.configs['model']
        detection_model = model_builder.build(model_config=model_config, is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(self.path_to_ckpt, 'ckpt-0')).expect_partial()

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""
            image, shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])
        self.detect_fn = detect_fn

    def evaluate_image(self, image_np):
        tf = self.tf
        print('image size: ', image_np.shape)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        input_tensor = tf.convert_to_tensor(image_np_expanded, dtype=tf.float32)
        detections, predictions_dict, shapes = self.detect_fn(input_tensor)
        return((detections, predictions_dict, shapes))

    def annotate_image(
            self, image_np, detections, classes,
            max_boxes_to_draw=5,
            min_score_thresh=.50
        ):
        image_np_with_detections = image_np.copy()

        self.viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            classes,
            detections['detection_scores'][0].numpy(),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False
        )
        return image_np_with_detections;

    def detection(self, image, min_threshold, max_boxes):
        start = time.time()
        boxes = []
        class_ids = []
        class_names = []
        scores = []

        # Read frame from camera
        image_np = self.cv_bridge.compressed_imgmsg_to_cv2(image, desired_encoding='bgr8')
        print('image_np array shape', image_np.shape)
        # gray to color if needed
        if len(image_np.shape) == 2 or image_np.shape[2] != 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            print('image_np array shape', image_np.shape)
        #sample_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'od_sample.jpg')
        #cv2.imwrite(sample_path, image_np)
        (detections, predictions_dict, shapes) = self.evaluate_image(image_np)
        classes = (detections['detection_classes'][0].numpy() + LABEL_ID_OFFSET).astype(int)
        for i in range(len(detections['detection_scores'][0])):
            score = float(detections['detection_scores'][0][i])
            if score < min_threshold or i >= max_boxes:
                break
            index = classes[i]
            od_box = detections['detection_boxes'][0][i].numpy().tolist()
            (ymin, xmin, ymax, xmax) = od_box
            # convert to float pixels
            im_height, im_width = image_np.shape[:2]
            (pxmin, pxmax, pymin, pymax) = map(float,
                (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            )
            box = (pymin, pxmin, pymax, pxmax)

            id = int(self.category_index[index]['id'])
            name = str(self.category_index[index]['name'])
            #ibox = map(int, (xmin * im_width, xmax * im_width,
            #                            ymin * im_height, ymax * im_height))
            #(left, right, top, bottom) = ibox
            #print(fstr({'score': score, 'id': id, 'name': name, 'box': (left, right, top, bottom)}))
            boxes.append(box)
            class_ids.append(id)
            class_names.append(name)
            scores.append(score)

        # annotated output
        if self.image_compressed_pub.get_num_connections() > 0:
            image_np_with_detections = self.annotate_image(
                image_np, detections, classes,
                max_boxes_to_draw=max_boxes,
                min_score_thresh=min_threshold
            )
            self.image_compressed_pub.publish(self.cv_bridge.cv2_to_compressed_imgmsg(image_np_with_detections))
        detection_time_ms = (time.time() - start) * 1000.
        print(sstr('detection_time_ms'))
        return ((boxes, class_ids, class_names, scores))

PARAMETERS = (
    ('use_gpu', USE_GPU),
    ('max_detections', 10),
    ('min_threshold', .20),
    ('input_topic', CAMERA_TOPIC),
    ('model_level', 2)
)

def retrieve_parms():
    parms = dict()
    for parm in PARAMETERS:
        parms[parm[0]] = rospy.get_param('/object_detect/' + parm[0], parm[1])
    return parms

service_queue = Queue()
msg_queue = deque([], 1)

def on_service_call(req):
    responseQueue = Queue()
    service_queue.put((req, responseQueue))
    response = responseQueue.get()
    return response

def msg_cb(msg):
    # callback just queues messages for processing
    msg_queue.append(msg)

def main():
    rospy.init_node('object_detect')
    name = rospy.get_name()
    rospy.loginfo(name + ' starting')
    
    # retrieve parameters
    parms = retrieve_parms()
    print('parms' + fstr(parms))

    od = ObjectDetectClass(parms['use_gpu'], parms['model_level'])

    # read and process the sample file to prime the pump
    # TODO The main call is still slow initially
    rospy.loginfo('Processing sample image to prime the pump')
    sample_im = cv2.imread(SAMPLE_PATH)
    sample_msg = od.cv_bridge.cv2_to_compressed_imgmsg(sample_im)
    msg_queue.append(sample_msg)
    #(boxes, class_ids, class_names, scores) = od.detection(sample_msg, .50, 1)

    image_sub = None
    objectDetectService = None
    while not rospy.is_shutdown():
        try:
            image = None
            service_request = None
            response_queue = None
            if not service_queue.empty():
                service_request = service_queue.get()
                service_msg = service_request[0]
                response_queue = service_request[1]
                if service_msg.image_topic:
                    print(fstr({'service request\n': service_request}))
                    oneshot_queue = Queue()
                    oneshot_cb = lambda image_msg: oneshot_queue.put(image_msg);print('got image_msg')
                    oneshot_sub = rospy.Subscriber(service_msg.image_topic, CompressedImage, oneshot_cb)
                    image = oneshot_queue.get()
                    print('got image')
                    oneshot_sub.unregister()
                else:
                    image = service_msg.image
                max_detections = service_msg.max_detections
                min_threshold = service_msg.min_threshold
            elif len(msg_queue):
                image = msg_queue.pop()
                min_threshold = parms['min_threshold']
                max_detections = parms['max_detections']
            else:
                rospy.sleep(.02)

            if image:
                print('asking for detections')
                (boxes, class_ids, class_names, scores) = od.detection(image, min_threshold, max_detections)
                print(fstr({'num_detected': len(boxes)}))
                for i in range(len(boxes)):
                    print(fstr({'score': scores[i], 'id': class_ids[i], 'name': class_names[i], 'box': boxes[i]}))
                if response_queue:
                    response = ObjectDetectResponse()
                    # match the response header to the request header for synchronization
                    response.header.stamp = service_msg.header.stamp
                    response.header.seq = service_msg.header.seq
                    for i in range(len(boxes)):
                        (pymin, pxmin, pymax, pxmax) = map(int, boxes[i])
                        response.xmin.append(pxmin)
                        response.ymin.append(pymin)
                        response.xmax.append(pxmax)
                        response.ymax.append(pymax)
                    response.scores = scores
                    response.class_names = class_names
                    response.class_ids = class_ids
                    print('full response\n', response)
                    response_queue.put(response)
                # we've delayed subscriptions until we are ready for action
                if not objectDetectService:
                    rospy.loginfo('Finished processing sample image, enabling subscriptions')
                    if parms['input_topic']:
                        image_sub = rospy.Subscriber(parms['input_topic'], CompressedImage, msg_cb)
                    objectDetectService = rospy.Service('/bdbd/objectDetect', ObjectDetect, on_service_call)

        except (SystemExit, KeyboardInterrupt):
            break;
        except:
            rospy.logwarn(traceback.format_exc())
            break
if __name__ == '__main__':
    main()
