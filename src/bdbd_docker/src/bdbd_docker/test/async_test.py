#use assync/await with callbacks to two cameras
# demo and test of bdbd_common.libpy.rosasync
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, SetPanTilt, SetPanTiltRequest
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
import cv_bridge
import asyncio
from queue import Queue
import random
from bdbd_docker.libpy.rosasync import runner, asyncMessageSingle

cvBridge = cv_bridge.CvBridge()

PANTILT_CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
SR305_CAMERA = '/sr305/color/image_raw/compressed'

async def async_main():
    rospy.init_node('async_test')

    def shutdown(reason):
        # add any shutdown tasks here
        rospy.loginfo('ros shutting down because {}'.format(reason))

    rospy.core.add_preshutdown_hook(shutdown)

    try:
        while not rospy.is_shutdown():
            # running in parallel
            images = await asyncio.gather(
                asyncMessageSingle(PANTILT_CAMERA, CompressedImage, timeout=.2),
                asyncMessageSingle(SR305_CAMERA, CompressedImage)
            )
            for image in images:
                print('gathered format', image.format)
    
    except rospy.ROSInterruptException as exc:
        rospy.loginfo('async_main got exception type {}: {}'.format(type(exc), exc))
    except Exception as exc:
        rospy.logerr('async_main got exception type {}: {}'.format(type(exc), exc))
    finally:
        print('async_main exiting')

def main():
    try:
        runner(async_main)
    except Exception as exc:
        print('main got exception', exc)
    finally:
        print('main exiting')

if __name__ == '__main__':
    main()
