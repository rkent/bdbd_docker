# convert point in one camera to a point in another.

class Pt2pt():
    def __init__(self, pcm1, pcm2):
        self.pcm1 = pcm1
        self.pcm2 = pcm2

CAM1 = '/bdbd/pantilt_camera/image_raw/compressed'
CAM2 = '/sr305/color/image_raw/compressed'

# projected angles of image point p given pinhole camera model pcm 
import math
RADIANS_TO_DEGREES = 180. / math.pi
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

if __name__ == '__main__':
    from image_geometry import PinholeCameraModel
    import rospy
    import apriltag
    from cv_bridge import CvBridge
    import cv2
    from sensor_msgs.msg import CompressedImage, CameraInfo
    import time
    import inspect
    from bdbd_common.srv import SetPanTilt, SetPanTiltRequest

    detector = apriltag.Detector()
    cvBridge = CvBridge()

    rospy.init_node('pt2pt')
    CAM1_BASE = '/bdbd/pantilt_camera'
    CAM2_BASE = '/sr305/color'
    pcm1 = PinholeCameraModel()
    pcm2 = PinholeCameraModel()
    pcm1.fromCameraInfo(rospy.wait_for_message(CAM1_BASE + '/camera_info', CameraInfo))
    pcm2.fromCameraInfo(rospy.wait_for_message(CAM2_BASE + '/camera_info', CameraInfo))
    pt2pt = Pt2pt(pcm1, pcm2)

    image1_msg = rospy.wait_for_message(CAM1, CompressedImage)
    image1_np = cvBridge.compressed_imgmsg_to_cv2(image1_msg, desired_encoding='bgr8')
    image1 = cv2.cvtColor(image1_np, cv2.COLOR_BGR2GRAY)
    start = time.time()
    results = detector.detect(image1)
    print('elapsed time {}'.format(time.time() - start))
    if not results:
        print('no result')
        exit(0)
    center = results[0].center
    ray = pcm1.projectPixelTo3dRay(center)
    angles = dot_angles(pcm1, center)
    point2 = pcm2.project3dToPixel(ray)
    print('ray', ray, 'point2', point2, 'angles', angles)

    # compare angle methods
    a1 = RADIANS_TO_DEGREES * math.atan2(ray[0], ray[2])
    a2 = RADIANS_TO_DEGREES * math.atan2(ray[1], ray[2])
    print(a1, a2)

    pantilt_srv = rospy.ServiceProxy('/bdbd/set_pan_tilt', SetPanTilt)
    pan = 90 - a1
    tilt = 45 + a2
    pantilt_srv(SetPanTiltRequest(pan, tilt, False))

    '''
    for result in results:
        print(result.center)
        for member in inspect.getmembers(result):
            print('model member {}'.format(member[0]))
    '''

    rospy.loginfo('done')
