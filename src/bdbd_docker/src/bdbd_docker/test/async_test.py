#use assync/await with callbacks to two cameras
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, SetPanTilt, SetPanTiltRequest
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
import cv_bridge
import asyncio
from queue import Queue

cvBridge = cv_bridge.CvBridge()

PANTILT_CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
SR305_CAMERA = '/sr305/color/image_raw/compressed'

# old way
def messageSingle(topic, type):
    responseQueue = Queue()
    sub = rospy.Subscriber(topic, type, lambda msg:responseQueue.put(msg))
    result = responseQueue.get()
    sub.unregister()
    return result

# new way
async def asyncMessageSingle(topic, type, timeout=None):
    return rospy.wait_for_message(topic, type, timeout)

async def async_main():
    try:
        while not rospy.is_shutdown():
            # old way
            '''
            pantilt_msg = messageSingle(PANTILT_CAMERA, CompressedImage)
            print('pantilt format', pantilt_msg.format)
            sr305_msg = messageSingle(SR305_CAMERA, CompressedImage)
            print('sr305 format', sr305_msg.format)

            # new way
            aspantilt_msg = await asyncMessageSingle(PANTILT_CAMERA, CompressedImage)
            print('async pantilt format', aspantilt_msg.format)
            assr305_msg = await asyncMessageSingle(SR305_CAMERA, CompressedImage)
            print('async sr305 format', assr305_msg.format)
            '''
            # running in parallel
            images = await asyncio.gather(
                asyncMessageSingle(PANTILT_CAMERA, CompressedImage),
                asyncMessageSingle(SR305_CAMERA, CompressedImage)
            )
            for image in images:
                print('gathered format', image.format)
            #break
    except Exception as exc:
        print('got exception', exc)
    print('exited')

async def runner():
    
    except Exception as exc:
        task.cancel()
        raise exc

def main():
    rospy.init_node('async_test')

    try:
        task = asyncio.create_task(async_main())
        rospy.on_shutdown(lambda : task.cancel())
        await task
    except Exception as exc:
        print('got async exception', exc)

if __name__ == '__main__':
    main()
