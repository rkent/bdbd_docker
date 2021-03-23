# Here I locate an apriltag with one camera, and use that to position the
# pan/tilt to center on that. Then I calculate the positioning error in the
# pan/tilt camera.

# apriltag_detect must be running as well as basic BDBD nodes

import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, SetPanTilt, SetPanTiltRequest
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
from image_geometry import PinholeCameraModel
import cv_bridge
import math
import asyncio

cvBridge = cv_bridge.CvBridge()
RADIANS_TO_DEGREES = 180. / math.pi

PANTILT_CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
LOCATE_CAMERA = '/t265/fisheye1/image_raw/compressed'

def handle_exception(loop, context):
    # context["message"] will always be there; but context["exception"] may not
    msg = context.get("exception", context["message"])
    rospy.logerr(f"Caught exception: {msg}")
    rospy.loginfo("Shutting down...")
    asyncio.create_task(shutdown(loop))

async def shutdown():
    # add any shutdown tasks here
    print('shutting down')

async def async_main():
    await asyncio.sleep(2)
    raise Exception('dummy exception')
    rospy.spin()

def main():
    rospy.init_node('centerpt')
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)

    try:
        task = asyncio.create_task(async_main())
        rospy.on_shutdown(lambda : task.cancel())
        loop.run_forever()
    except Exception as exc:
        print('got async exception', exc)

if __name__ == '__main__':
    main()
    