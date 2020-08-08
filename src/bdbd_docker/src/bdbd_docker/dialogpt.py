#!/usr/bin/env python

#  ROS node to run Microsoft DialoGPT conversational model using hugging face transformers

import rospy
import traceback
import signal
import time
from bdbd_common.srv import Dialog
try:
    from Queue import Queue
except:
    from queue import Queue
import time

from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

PERIOD = 0.01 # update time in seconds
CONTEXT_COUNT = 8 # maximum number of context items

class DialoGPT():
    def __init__(self):
        rospy.init_node('dialogpt', disable_signals=True)
        name = rospy.get_name()
        rospy.loginfo(name + ' starting')
        self.queue = Queue()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium").cuda()
        self.idses = []
        rospy.loginfo(name + ' done with init')
        rospy.Service('/bdbd/dialog', Dialog, self.on_service_call)

        # https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.active = True     

    def exit_gracefully(self, signum, frame):
        rospy.loginfo('node {} shutting down 2'.format(rospy.get_name()))
        rospy.signal_shutdown('Got shutdown')
        print('Initiated shutdown')
        self.active = False

    def on_service_call(self, req):
        responseQueue = Queue()
        self.queue.put([req, responseQueue])
        response = responseQueue.get()
        return(response)

    def run(self):
        while self.active and not rospy.is_shutdown():
            try:
                while not self.queue.empty():
                    start = time.time()
                    req, responseQueue = self.queue.get()
                    rospy.loginfo('Speaker text: {}'.format(req.text))
                    if req.options:
                        rospy.loginfo('Dialog options: {}'.format(req.options))
                    
                    # encode the new user input, add the eos_token and return a tensor in Pytorch
                    new_user_input_ids = self.tokenizer.encode(req.text + self.tokenizer.eos_token, return_tensors='pt').cuda()
                    self.idses.append(new_user_input_ids)

                    # assemble the chat history, limiting context size
                    while len(self.idses) > CONTEXT_COUNT:
                        self.idses.pop(0)
                    context = self.idses[0]
                    for i in range(1, len(self.idses)):
                        context = torch.cat([context, self.idses[i]], dim=-1)

                    print('context.shape:{}'.format(context.shape))
                    # generated a response while limiting the total chat history to 1000 tokens, 
                    new_ids = self.model.generate(
                        context, 
                        max_length=1000, 
                        pad_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2
                    )
                    print('new_ids.shape:{}'.format(new_ids.shape))
                    response_ids = new_ids[:, context.shape[-1]:]
                    self.idses.append(response_ids)
                    response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                    rospy.loginfo('time: {:6.3f} DialoGPT: {}'.format(time.time() - start, response))
                    responseQueue.put(response)
            except:
                rospy.logerr(traceback.format_exc())
            rospy.sleep(PERIOD)
        print("Ready to shutdown")

def main():
    dialogpt = DialoGPT()
    dialogpt.run()

if __name__ == '__main__':
    main()
