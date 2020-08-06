#!/usr/bin/env python

#  Testing of Microsoft DialoGPT conversational model using hugging face transformers

import traceback

from transformers import AutoTokenizer, TFAutoModelWithLMHead
import tensorflow as tf
import torch
import time

PERIOD = 0.01 # update time in seconds
class DialoGPT():
    def __init__(self):
        name = 'dialogpt test'
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = TFAutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")
        print(name + ' done with init')

        # Testing
        #texts = ['Do you like movies?', 'What is your favorite movie?', 'My favorite is Star Wars.']
        #texts = ['Are you a robot?', 'But are you a robot?', "I'm sure you are a robot.", 'Are you a boy or girl?']
        #for text in texts:
        text = "I'm feeling kind of bored"
        print('Seed: {}'.format(text))
        context = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors='pt')
        count = 8
        idses = [context]
        while True:
            start = time.time()
            # encode the new user input, add the eos_token and return a tensor in Pytorch

            bot_input_ids = context
            # generate a response while limiting the total chat history to 1000 tokens, 
            new_ids = self.model.generate(
                bot_input_ids, 
                max_length=500, 
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.4
            )
            response_ids = new_ids[:, bot_input_ids.shape[-1]:]
            #print('new_ids.shape: {} response_ids.shape {}'.format(new_ids.shape, response_ids.shape))
            response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # keep count previous statements
            idses.append(response_ids)
            while len(idses) > count:
                idses.pop(0)
            context = idses[0]
            for i in range(1, len(idses)):
                context = torch.cat([context, idses[i]], dim=-1)
            #print('context.shape {}'.format(context.shape))
            print('({:6.4f}) DialoGPT: {}'.format(time.time()-start, response))
        exit()

def main():
    dialogpt = DialoGPT()

if __name__ == '__main__':
    main()
