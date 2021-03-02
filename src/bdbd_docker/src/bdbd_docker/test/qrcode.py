import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import rospy
from bdbd_common.utils import fstr, sstr
from bdbd_common.messageSingle import messageSingle
import time

cvBridge = CvBridge()
qrCodeDetector = cv2.QRCodeDetector()

rospy.init_node('camera_align')

piTopic = '/bdbd/pantilt_camera/image_rect_color/compressed'

qrPub = rospy.Publisher('/camera_align/pi/image_color/compressed', CompressedImage, queue_size=1)

while not rospy.is_shutdown():
    image_msg = messageSingle(piTopic, CompressedImage)
    image = cvBridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    start = time.time()
    found, points = qrCodeDetector.detect(image)
    print('elapsed time {}'.format(time.time() - start))
    if found:
        image = cv2.rectangle(image, tuple(points[0, 0]), tuple(points[0, 2]), (127,255, 127), 4)
        image_msg = cvBridge.cv2_to_compressed_imgmsg(image)
        qrPub.publish(image_msg)
        print(points)
    else:
        print('qr code not found')
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