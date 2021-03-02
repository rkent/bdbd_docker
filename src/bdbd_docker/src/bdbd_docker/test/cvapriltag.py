import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import rospy
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
import time
import apriltag

cvBridge = CvBridge()
detector = apriltag.Detector()

rospy.init_node('camera_align')

piTopic = '/bdbd/pantilt_camera/image_rect_color/compressed'

qrPub = rospy.Publisher('/camera_align/pi/image_color/compressed', CompressedImage, queue_size=1)

while not rospy.is_shutdown():
    image_msg = messageSingle(piTopic, CompressedImage)
    imageC = cvBridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    image = cv2.cvtColor(imageC, cv2.COLOR_BGR2GRAY)
    start = time.time()
    results = detector.detect(image)
    print('elapsed time {}'.format(time.time() - start))
    print('result\n', results)

    if results:
        result = results[0]
        print(type(result), dir(result))
        corners = result.corners
        #corners0 = tuple(corners[0].astype(int))
        #corners2 = tuple(corners[2].astype(int))
        #corners0 = (642, 320)
        #corners2 = (800, 468)
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