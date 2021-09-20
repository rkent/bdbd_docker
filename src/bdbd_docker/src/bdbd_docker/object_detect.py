#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import rospy
import cv2
import time
from bdbd_common.utils import fstr, sstr, getShape, gstr
from bdbd_docker.libpy.objectDetector import ObjectDetector
from bdbd_common.messageSingle import messageSingle
from bdbd_common.imageRectify import ImageRectify
from bdbd_common.doerRequest import DoerRequest
from sensor_msgs.msg import CompressedImage
import traceback
from bdbd_common.srv import ObjectDetect, ObjectDetectResponse
from queue import Queue

dr = DoerRequest()

# adapted from
#  https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/auto_examples/object_detection_camera.html#sphx-glr-auto-examples-object-detection-camera-py

SAMPLE_NAME = 'fisheye.jpg'
LIMIT_GROWTH = True
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', SAMPLE_NAME)
LABEL_ID_OFFSET = 1

CAMERA_TOPIC = ''
#CAMERA_TOPIC = '/t265/fisheye1/image_raw/compressed'

CAMERA_BASE = ''
#CAMERA_BASE = '/t265/fisheye1'

USE_GPU = False
DO_FACES = True
PARAMETERS = (
    ('use_gpu', USE_GPU),
    ('max_detections', 30),
    ('min_threshold', .10),
    ('input_topic', CAMERA_TOPIC),
    ('model_level', 0)
)

if DO_FACES:
    from bdbd_docker.libpy.face_detect import extract_faces
# keyed by topic base
image_rectifiers = {}

def retrieve_parms():
    parms = dict()
    for parm in PARAMETERS:
        parms[parm[0]] = rospy.get_param('/object_detect/' + parm[0], parm[1])
    return parms

service_queue = Queue()

def on_service_call(req):
    responseQueue = Queue()
    service_queue.put((req, responseQueue))
    response = responseQueue.get()
    return response

def getCameraTopicBase(imageTopic):
    baseTopic = None
    offset = imageTopic.find('/image_')
    if offset >= 0:
        baseTopic = imageTopic[:offset]
    return baseTopic

def main():
    rospy.init_node('object_detect')
    name = rospy.get_name()
    rospy.loginfo(name + ' starting')
    
    # retrieve parameters
    parms = retrieve_parms()
    print('parms' + fstr(parms))

    od = ObjectDetector(parms['use_gpu'], parms['model_level'])

    objectDetectService = None
    firstCall = True
    while not rospy.is_shutdown():
        try:
            image_msg = None
            image_np = None
            response_queue = None
            imageRectify = None
            imageTopic = None
            min_threshold = parms['min_threshold']
            max_detections = parms['max_detections']
            if firstCall:
                firstCall = False
                rospy.loginfo('Processing sample image to prime the pump')
                image_np = cv2.imread(SAMPLE_PATH)
            elif not service_queue.empty():
                service_request = service_queue.get()
                service_msg = service_request[0]
                response_queue = service_request[1]
                if service_msg.image_topic:
                    imageTopic = service_msg.image_topic
                else:
                    image_msg = service_msg.image # we assume this is a rectified image
                max_detections = service_msg.max_detections
                min_threshold = service_msg.min_threshold
            if image_np is None and image_msg is None:
                if imageTopic is None and parms['input_topic']:
                    imageTopic = parms['input_topic']
                if imageTopic:
                    # get rectifier appropriate for the image topic
                    cameraTopicBase = getCameraTopicBase(imageTopic)
                    if cameraTopicBase in image_rectifiers:
                        print('Found image rect for base')
                        imageRectify = image_rectifiers[cameraTopicBase]
                    else:
                        print('looking for image rect for base')
                        try:
                            imageRectify = ImageRectify(cameraTopicBase, do_publish=False)
                        except rospy.ROSException as exception:
                            rospy.logwarn('Cannot rectify, proceeding without: ({})'.format(exception))
                            imageRectify = None
                    if imageRectify and imageRectify.info_msg.distortion_model != 'equidistant':
                        imageRectify = None
                    image_rectifiers[cameraTopicBase] = imageRectify
                    image_msg = dr.wait_for_message(imageTopic, CompressedImage, timeout=10.0)
                    if imageRectify:
                        image_np = imageRectify.get(image_msg)
            if image_np is None and image_msg is None:
                rospy.sleep(.02)
            else:
                print('asking for detections')
                if image_np is None and image_msg is not None:
                    image_np = od.cv_bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                if image_np is not None:
                    # gray to color if needed
                    if len(image_np.shape) == 2 or image_np.shape[2] != 3:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                    (boxes, class_ids, class_names, scores) = od.detection_np(image_np, min_threshold, max_detections, image_msg)
                print(fstr({'num_detected': len(boxes)}))

                if DO_FACES:
                    MIN_SIZE = 30
                    MIN_SCORE = 0.90
                    faces, results = extract_faces(image_np, min_size=(MIN_SIZE, MIN_SIZE), min_confidence=MIN_SCORE)
                    rospy.loginfo('found {} faces'.format(len(faces)))
                    for result in results:
                        x1, y1, width, height = result['box']
                        if width < MIN_SIZE or height < MIN_SIZE or result['confidence'] < MIN_SCORE:
                            continue
                        # add this to the detections at the beginning
                        boxes.insert(0, tuple(map(float, (y1, x1, y1 + height, x1 + width))))
                        class_ids.insert(0, 101)
                        class_names.insert(0,'face')
                        scores.insert(0, result['confidence'])
                        #print(result)

                rospy.loginfo('Found {} objects'.format(len(boxes)))
                for i in range(len(boxes)):
                    print(fstr({'score': scores[i], 'id': class_ids[i], 'name': class_names[i], 'box': boxes[i]}))

                if response_queue:
                    response = ObjectDetectResponse()
                    # match the response header to the request header for synchronization
                    response.header = service_msg.header
                    for i in range(len(boxes)):
                        (pymin, pxmin, pymax, pxmax) = map(int, boxes[i])
                        response.xmin.append(pxmin)
                        response.ymin.append(pymin)
                        response.xmax.append(pxmax)
                        response.ymax.append(pymax)
                    response.scores = scores
                    response.class_names = class_names
                    response.class_ids = class_ids
                    if image_msg:
                        response.image = image_msg
                    else:
                        response.image = od.cv_bridge.cv2_to_compressed_imgmsg(image_np)
                    response_queue.put(response)
                # we've delayed subscriptions until we are ready for action
                if not objectDetectService:
                    rospy.loginfo('Finished processing sample image, enabling subscriptions')
                    objectDetectService = rospy.Service('/bdbd/objectDetect', ObjectDetect, on_service_call)

        except (SystemExit, KeyboardInterrupt):
            break
        except:
            rospy.logwarn('object_detect failed: {}'.format(traceback.format_exc()))
            break

if __name__ == '__main__':
    main()
