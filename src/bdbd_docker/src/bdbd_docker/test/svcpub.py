import rospy
import traceback
from std_srvs.srv import SetBool, SetBoolResponse
try:
    from Queue import Queue
except:
    from queue import Queue

service_queue = Queue()

def on_service_call(req):
    responseQueue = Queue()
    service_queue.put((req, responseQueue))
    response = responseQueue.get()
    return response

def main():
    rospy.init_node('svcpub')
    rospy.Service('/svctest', SetBool, on_service_call)

    while not rospy.is_shutdown():
        try:
            rospy.loginfo('waiting for service request')
            service_msg, response_queue = service_queue.get()
            node = service_msg._connection_header['callerid']
            print('got service request: {} from {}'.format(service_msg, node))
            response = SetBoolResponse()
            response.message = 'response message'
            response.success = True
            response_queue.put(response)

        except rospy.ServiceException:
            rospy.logwarn('Service Exception {}'.format(traceback.format_exc()))
        except:
            rospy.logerr('Exception while waiting for service, exiting {}'.format(traceback.format_exc()))
            break

main()