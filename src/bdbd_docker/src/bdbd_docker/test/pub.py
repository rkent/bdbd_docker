import rospy
from std_msgs.msg import String

rospy.init_node('pubber')

pub = rospy.Publisher('/pub', String, queue_size=10)

while not rospy.is_shutdown():
    pub.publish('the message')
    print('published')
    rospy.sleep(1)