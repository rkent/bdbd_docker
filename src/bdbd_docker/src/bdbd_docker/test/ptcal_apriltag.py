# calibration of the pan/tilt camera relative to the sr305 using an apriltag target

'''
2021-03-20

What I learned from all of this is that the ros camera calibration does a decent job of
determining the focal length. I'll redo this to simply center the camera, in a renamed
file. 

'''
import rospy
from bdbd_common.srv import ObjectDetect, ObjectDetectRequest, SetPanTilt, SetPanTiltRequest
from bdbd_common.utils import fstr, sstr
from sensor_msgs.msg import CompressedImage, CameraInfo
import numpy as np
from image_geometry import PinholeCameraModel
import cv_bridge
import math
from statistics import mean

cvBridge = cv_bridge.CvBridge()
RADIANS_TO_DEGREES = 180. / math.pi

# projected angles of image point p given pinhole camera model pcm 
def dot_angles(pcm, p):
    xa = RADIANS_TO_DEGREES * math.atan2(p[0] - pcm.cx(), pcm.fx())
    ya = RADIANS_TO_DEGREES * math.atan2(p[1] - pcm.cy(), pcm.fy())
    return (xa, ya)

rospy.init_node('ptcal_apriltag')
PANTILT_CAMERA = '/bdbd/pantilt_camera/image_raw/compressed'
SR305_CAMERA = '/sr305/color/image_raw/compressed'
SET_PT = '/bdbd/set_pan_tilt'
OBJECT_COUNT = 12
od_srv = rospy.ServiceProxy('/bdbd/apriltagDetect', ObjectDetect)
pantilt_srv = rospy.ServiceProxy('/bdbd/set_pan_tilt', SetPanTilt)

# Pan/tilt request
pt_odr = ObjectDetectRequest()
pt_odr.image_topic = PANTILT_CAMERA

# SR305 request
sr305_odr = ObjectDetectRequest()
sr305_odr.image_topic = SR305_CAMERA

while not rospy.is_shutdown():

    # phase 0: location using sr305 camera
    response = od_srv(sr305_odr)
    if len(response.class_names):
        center = ( (response.xmin[0] + response.xmax[0]) / 2., (response.ymin[0] + response.ymax[0]) / 2.)
        print('sr305 center {}'.format(center))

    else:
        print('sr305 bad')
        break

    pcm_sr305 = PinholeCameraModel()
    msg_sr305 = rospy.wait_for_message('/sr305/color' + '/camera_info', CameraInfo)
    pcm_sr305.fromCameraInfo(msg_sr305)
    sr305_ray = pcm_sr305.projectPixelTo3dRay(center)
    angles = dot_angles(pcm_sr305, center)
    print(sstr('sr305_ray angles'))
    #exit(0)

    # phase 1: confirm target on pan/tilt
    pan_min = 10
    pan_max = 170
    tilt_min = 10
    tilt_max = 80

    pan = 90
    tilt = 45
    ptr = SetPanTiltRequest()
    ptr.pan = pan
    ptr.tilt = tilt
    pantilt_srv(SetPanTiltRequest(pan, tilt, True))

    response = od_srv(pt_odr)
    count = len(response.class_names)
    if count != 1:
        rospy.logerr('No or duplicate target seen in center of pan/tilt')
        break

    center = ( (response.xmin[0] + response.xmax[0]) / 2., (response.ymin[0] + response.ymax[0]) / 2.)
    print('id {} center {}'.format(response.class_names[0], center))

    # phase 2: determine limits
    # pan
    min_pan_bad = pan_min
    min_pan_good = pan
    max_pan_bad = pan_max
    max_pan_good = pan

    # min pan
    while True:
        new_pan = (min_pan_bad + min_pan_good) / 2
        if abs(pan - new_pan) < 1:
            break
        pan = new_pan

        print('trying {} {}'.format(pan, tilt))
        pantilt_srv(SetPanTiltRequest(pan, tilt, True))
        response = od_srv(pt_odr)
        if len(response.class_names):
            print('good')
            min_pan_good = pan
        else:
            print('bad')
            min_pan_bad = pan

    pan = 90
    # max_pan
    while True:        
        new_pan = (max_pan_bad + max_pan_good) / 2
        if abs(pan - new_pan) < 1:
            break
        pan = new_pan
        print('trying {} {}'.format(pan, tilt))
        pantilt_srv(SetPanTiltRequest(pan, tilt, True))
        response = od_srv(pt_odr)
        if len(response.class_names):
            print('good')
            max_pan_good = pan
        else:
            print('bad')
            max_pan_bad = pan
    
    print('pan limits: {} - {}'.format(min_pan_good, max_pan_good))

    # tilt
    pan = 90
    tilt = 45

    min_tilt_bad = tilt_min
    min_tilt_good = tilt
    max_tilt_bad = tilt_max
    max_tilt_good = tilt

    # min tilt
    while True:
        new_tilt = (min_tilt_bad + min_tilt_good) / 2
        if abs(tilt - new_tilt) < 1:
            break
        tilt = new_tilt

        print('trying {} {}'.format(pan, tilt))
        pantilt_srv(SetPanTiltRequest(pan, tilt, True))
        response = od_srv(pt_odr)
        if len(response.class_names):
            print('good')
            min_tilt_good = tilt
        else:
            print('bad')
            min_tilt_bad = tilt

    tilt = 45
    while True:        
        new_tilt = (max_tilt_bad + max_tilt_good) / 2
        if abs(tilt - new_tilt) < 1:
            break
        tilt = new_tilt
        print('trying {} {}'.format(pan, tilt))
        pantilt_srv(SetPanTiltRequest(pan, tilt, True))
        response = od_srv(pt_odr)
        if len(response.class_names):
            print('good')
            max_tilt_good = tilt
        else:
            print('bad')
            max_tilt_bad = tilt
    
    print('tilt limits: {} - {}'.format(min_tilt_good, max_tilt_good))

    # phase 3: test edges
    '''
    for pan in [min_pan_good, max_pan_good]:
        for tilt in [min_tilt_good, max_tilt_good]:
            pantilt_srv(SetPanTiltRequest(90, 45, True))
            # determine margin to ensure good in corners
            start_pan = pan
            while True:
                pantilt_srv(SetPanTiltRequest(pan, tilt, True))
                response = od_srv(pt_odr)
                if len(response.class_names):
                    print('{}, {} good'.format(pan, tilt))
                    break
                else:
                    print('{}, {} bad'.format(pan, tilt))
                    # move toward center
                    if pan > 90:
                        pan -= 1
                    else:
                        pan += 1
                    if tilt > 45:
                        tilt -= .5
                    else:
                        tilt += .5
            print('needed correction {}'.format(abs(start_pan - pan)))
    '''

    # previous edge tests recommended 3.0 as correction
    min_pan_good += 3
    max_pan_good -=3
    min_tilt_good += 2
    max_tilt_good -= 2

    print('Final range for test: ({} - {}), ({} - {})'.format(min_pan_good, max_pan_good, min_tilt_good, max_tilt_good))

    # phase 4: confirm edges and get positions
    edge_centers = {}
    for pan in [min_pan_good, max_pan_good]:
        for tilt in [min_tilt_good, max_tilt_good]:
            pantilt_srv(SetPanTiltRequest(90, 45, True))
            pantilt_srv(SetPanTiltRequest(pan, tilt, True))
            response = od_srv(pt_odr)
            if len(response.class_names):
                center = ( (response.xmin[0] + response.xmax[0]) / 2., (response.ymin[0] + response.ymax[0]) / 2.)
                print('{}, {} good center {}'.format(pan, tilt, center))
                edge_centers[(pan, tilt)] = center
            else:
                print('{}, {} bad'.format(pan, tilt))
                exit(0)
    print(sstr('edge_centers'))

    # phase 5: calibration

    # A: determine pan, tilt to center on target
    panc = (min_pan_good + max_pan_good) / 2
    tiltc = (min_tilt_good + max_tilt_good) / 2
    xcs = []
    ycs = []
    for pp in [panc/2, + 3*panc/2]:
        for tt in [tiltc/2, 3*tiltc/2]:
            pantilt_srv(SetPanTiltRequest(pp, tt, True))
            pantilt_srv(SetPanTiltRequest(panc, tiltc, True))
            response = od_srv(pt_odr)
            if len(response.class_names):
                center = ( (response.xmin[0] + response.xmax[0]) / 2., (response.ymin[0] + response.ymax[0]) / 2.)
                xcs.append(center[0])
                ycs.append(center[1])
                print('{:6.1f}, {:6.1f} center {}'.format(pan, tilt, center))
            else:
                print('{:6.1f}, {:6.1f} bad'.format(pan, tilt))
    xmc, ymc = mean(xcs), mean(ycs)
    print('mean center: {} {}'.format(xmc, ymc))

    # interpolate using edge values
    cnn = (edge_centers[(min_pan_good, min_tilt_good)])
    cnx = (edge_centers[(min_pan_good, max_tilt_good)])
    cxn = (edge_centers[(max_pan_good, min_tilt_good)])
    cxx = (edge_centers[(max_pan_good, max_tilt_good)])
    delta_x = (cxn[0] + cxx[0] - cnn[0] - cnx[0]) / 2
    delta_y = (cnx[1] + cnx[1] - cnn[1] - cxn[1]) / 2
    print(sstr('delta_x delta_y cnn cnx cxn cxx'))

    # interpolate pan, tilt to center
    panc -= (xmc - 640) * ((max_pan_good - min_pan_good) / delta_x)  
    tiltc -= (ymc - 360) * ((max_tilt_good - min_tilt_good) / delta_y)
    print(sstr('panc tiltc'))
    xcs = []
    ycs = []
    for pp in [panc/2, + 3*panc/2]:
        for tt in [tiltc/2, 3*tiltc/2]:
            pantilt_srv(SetPanTiltRequest(pp, tt, True))
            pantilt_srv(SetPanTiltRequest(panc, tiltc, True))
            response = od_srv(pt_odr)
            if len(response.class_names):
                center = ( (response.xmin[0] + response.xmax[0]) / 2., (response.ymin[0] + response.ymax[0]) / 2.)
                xcs.append(center[0])
                ycs.append(center[1])
                print('{:6.1f}, {:6.1f} center {}'.format(panc, tiltc, center))
            else:
                print('{:6.1f}, {:6.1f} bad'.format(panc, tiltc))
    xmc, ymc = mean(xcs), mean(ycs)
    print('mean center: {} {}'.format(xmc, ymc))
    panc_0 = panc + angles[0]
    tiltc_0 = tiltc - angles[1]
    print(sstr('panc_0 tiltc_0'))

    '''
    for pan in np.linspace(min_pan_good, max_pan_good, 6):
        for tilt in np.linspace(min_tilt_good, max_tilt_good, 6):
            pantilt_srv(SetPanTiltRequest(90, 45, True))
            pantilt_srv(SetPanTiltRequest(pan, tilt, True))
            response = od_srv(pt_odr)
            if len(response.class_names):
                center = ( (response.xmin[0] + response.xmax[0]) / 2., (response.ymin[0] + response.ymax[0]) / 2.)
            else:
                print('{:6.1f}, {:6.1f} bad'.format(pan, tilt))
            thetax = (pan - panc_0) / RADIANS_TO_DEGREES
            thetay = (tiltc_0 - tilt) / RADIANS_TO_DEGREES
            fx = (center[0] - 640) / math.tan(thetax)
            fy = (center[1] - 360) / math.tan(thetay)
            print(sstr('pan tilt fx fy center'))

    pantilt_srv(SetPanTiltRequest(panc, tiltc, True))
    '''

    break
