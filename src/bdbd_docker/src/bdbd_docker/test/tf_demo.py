# testing understanding and use of tf transforms
import rospy
import tf
from geometry_msgs.msg import PointStamped

if __name__ == '__main__':
    rospy.init_node('tf_demo')
    tfl = tf.TransformListener()
    rospy.sleep(1)

    a = PointStamped()
    while not rospy.is_shutdown():
        (trans, rot) = tfl.lookupTransform('t265_fisheye1_optical_frame', 't265_gyro_optical_frame', rospy.Time(0))
        print('trans, rot', trans, rot)
        print('matrix\n', tfl.fromTranslationRotation(trans, rot))
        break
    
    