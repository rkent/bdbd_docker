import rospy
import traceback
from std_srvs.srv import SetBool
try:
    from Queue import Queue
except:
    from queue import Queue

service_queue = Queue()

def main():
    rospy.init_node('svcsub')
    svc = rospy.ServiceProxy('/svctest', SetBool)

    while not rospy.is_shutdown():
        try:
            rospy.loginfo('sending service request')
            result = svc(True)
            print(result)
            print(result._connection_header['callerid'])

        except rospy.ServiceException:
            rospy.logwarn(traceback.format_exc())
        except:
            rospy.logerr('Exception while waiting for service, exiting')
            break
        rospy.sleep(1)

main()