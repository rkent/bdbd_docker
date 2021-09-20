import rospy
from std_msgs.msg import String

rospy.init_node('sub')

def on_pub(msg):
    print(msg)
    print(dir(msg))
    print('calling node:', msg._connection_header['callerid'])

rospy.Subscriber('/pub', String, on_pub)

rospy.spin()
