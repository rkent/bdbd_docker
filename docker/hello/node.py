#!/usr/bin/env python

#  a standalone ROS node to demo a ROS service in a docker container.

import rospy
import traceback
from std_srvs.srv import SetBool, SetBoolResponse
try:
    from Queue import Queue
except:
    from queue import Queue

PERIOD = 0.01 # update time in seconds
class Hello():
    def __init__(self):
        rospy.init_node('hello_docker')
        self.service = rospy.Service('/bdbd_docker/hello', SetBool, self.on_service_call)
        self.queue = Queue()

    def on_service_call(self, req):
        responseQueue = Queue()
        self.queue.put([req, responseQueue])
        response = responseQueue.get()
        return(response)

    def run(self):
        while not rospy.is_shutdown():
            try:
                while not self.queue.empty():
                    req, responseQueue = self.queue.get()
                    rospy.loginfo('Got hello with data {}'.format(req.data))
                    responseQueue.put([True, 'Hi from Hello'])
            except:
                rospy.logerr(traceback.format_exc())
            rospy.sleep(PERIOD)

def main():
    hello = Hello()
    hello.run()

if __name__ == '__main__':
    main()
