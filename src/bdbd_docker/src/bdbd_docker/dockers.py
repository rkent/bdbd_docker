#!/usr/bin/env python
#
# ROS service to manage docker containers that implement ROS functionality.
#

'''
Docker naming convention

Let NAME be the container identifier, for example 'hello'

* The container is started by running /opt/bdbd_docker/docker/NAME/run.sh
* The container name and hostname is NAME
* Containers are auto-remove, ie docker run --rm
* Containers are stopped using 'docker stop NAME'

'''
import rospy
import subprocess
import traceback
from bdbd_common.srv import DockerCommand
from queue import Queue
import docker
#import platform

ROS_IP = '192.168.0.59'
ROS_MASTER_URI =' http://nano.dryrain.org:11311/'

class DockerManagement():
    def __init__(self):
        rospy.init_node('dockers')
        self.client = {}
        self.client = docker.from_env()
        self.containers = set()
        #hostname = platform.node().split('.')[0]
        self.dockersService = rospy.Service('/bdbd_docker/containers', DockerCommand, self.handle_containers)
        rate = rospy.Rate(100)
        self.mainQueue = Queue()
        rospy.loginfo('Ready to process commands')
        while not rospy.is_shutdown():
            if not self.mainQueue.empty():
                req_type, name, command, responseQueue = self.mainQueue.get()
                response = None

                try:
                    if req_type == 'containers':
                        response = self.process_containers(name, command)

                    else:
                        rospy.logerr('Invalid dockerManagement req_type {}'.format(req_type))
                        response = 'error'

                except KeyboardInterrupt:
                    break
                except:
                    rospy.logerr(traceback.format_exc())
                finally:
                    if responseQueue:
                        responseQueue.put(response)
                        responseQueue = None
            else:
                rate.sleep()

        # shutdown
        names = list(self.containers)
        for name in names:
            self.process_containers(name, 'stop')

    def handle_containers(self, req):
        rospy.loginfo('Got docker launch request: name:{} command:{}'.format(req.name, req.command))
        responseQueue = Queue()
        self.mainQueue.put(['containers', req.name, req.command, responseQueue])
        response = responseQueue.get()
        rospy.loginfo('Docker launch response: {} {} {}'.format(response, req.name, req.command))
        return response

    def process_containers(self, name, command):
        rospy.loginfo('process container name:{} command:{}'.format(name, command))
        result = 'unexpected'

        container = None
        try:
            container = self.client.containers.get(name)
        except docker.errors.NotFound:
            pass

        if command == 'start':
            if container and container.status == 'running':
                rospy.logwarn('Container {} already active'.format(name))
                result = 'active'
            else:
                try:
                    if container:
                        container.remove()
                    subprocess.run(['/opt/bdbd_docker/docker/' + name + '/run.sh'], check=True)
                    result = 'started'
                    self.containers.add(name)
                except:
                    rospy.logerr(traceback.format_exc())
                    result = 'error'
        elif command == 'stop':
            if container and container.status == 'running':
                container.stop()
                result = 'stopped'
            elif container:
                result = 'inactive'
            else:
                result = 'notfound'
        elif command == 'kill':
            if container and container.status == 'running':
                container.kill()
                result = 'killed'
            elif container:
                result = 'inactive'
            else:
                result = 'notfound'
        else:
            rospy.logerr('Unexpected command: {}'.format(command))
        return result

def main():
    DockerManagement()

if __name__ == '__main__':
    main()
