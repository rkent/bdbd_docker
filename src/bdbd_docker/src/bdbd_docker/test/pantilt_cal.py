# calibration of pantilt transformations using an Apriltag image seen by both a calibrated t265 and the pantilt.

import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import rospy
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
from bdbd_common.msg import PanTilt
import random
import time
import apriltag
import numpy as np
centers = []
ir_centers = []

cvBridge = CvBridge()
detector = apriltag.Detector()
pan = 90
tilt = 45

rospy.init_node('camera_align')
pantilt_pub = rospy.Publisher('/bdbd/pantilt', PanTilt, queue_size=1)
while not rospy.is_shutdown():
    inputs = list(map(int, input('Enter pan tilt (negative to exit):').split()))
    if inputs[0] < 0:
        print('All done')
        break
    (pan, tilt) = inputs
    print(pan, tilt)
    pantilt_pub.publish(pan, tilt)

MAX_DPAN = 170
MIN_DPAN = 10
MAX_DTILT = 80
MIN_DTILT = 10
SETTLE_TIME = 1.0
TRIALS = 20
piTopic = '/bdbd/pantilt_camera/image_raw/compressed'
irTopic = '/t265/fisheye1/image_rect/compressed'

qrPub = rospy.Publisher('/camera_align/pi/image_color/compressed', CompressedImage, queue_size=1)

while not rospy.is_shutdown():
    for trial in range(TRIALS):
        rpan = random.randint(MIN_DPAN, MAX_DPAN)
        rtilt = random.randint(MIN_DTILT, MAX_DTILT)
        pantilt_pub.publish(rpan, rtilt)
        rospy.sleep(SETTLE_TIME)
        pantilt_pub.publish(pan, tilt)
        rospy.sleep(SETTLE_TIME)
        
        pi_msg = messageSingle(piTopic, CompressedImage)
        imageC = cvBridge.compressed_imgmsg_to_cv2(pi_msg, desired_encoding='bgr8')
        image = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)
        results = detector.detect(image)

        ir_msg = messageSingle(irTopic, CompressedImage)
        ir_image = cvBridge.compressed_imgmsg_to_cv2(ir_msg, desired_encoding='passthrough')
        ir_results = detector.detect(ir_image)
        print('pi_center', results and results[0].center, 'ir center', ir_results and ir_results[0].center)
        if ir_results:
            ir_centers.append(ir_results[0].center)

        if results:
            result = results[0]
            centers.append(result.center)
            print('center', result.center)
            corners = result.corners
            color = (127,255, 127)
            thick = 4
            #print(type(corners0), corners0, type(corners2), corners2)
            for i in range(4):
                p0 = tuple(corners[i].astype(int))
                p1 = tuple(corners[(i+1) % 4].astype(int))
                imageC = cv2.line(imageC, p0, p1, color, thick)
            #imageC = cv2.rectangle(imageC, corners0, corners2, (127,255, 127), 4)
            image_msg = cvBridge.cv2_to_compressed_imgmsg(imageC)
            qrPub.publish(image_msg)

        rospy.sleep(1)
    break

means = np.mean(centers, axis=0)
stds = np.std(centers, axis=0)
print('pi means, stds', means, stds)
print('ir mean, std', np.mean(ir_centers, axis=0), np.std(ir_centers, axis=0))

pantilt_pub.publish(90, 45)
'''
decodedText, points, _ = qrCodeDetector.detectAndDecode(image)
 
if points is not None:
 
    nrOfPoints = len(points)
    print('points', points)
 
    for i in range(nrOfPoints):
        nextPointIndex = (i+1) % nrOfPoints
        cv2.line(image, tuple(points[i][0]), tuple(points[nextPointIndex][0]), (255,0,0), 5)
 
    print(decodedText)    
 
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     
 
else:
    print("QR code not detected")
'''