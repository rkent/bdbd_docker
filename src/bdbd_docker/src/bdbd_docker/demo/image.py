# demo of object classification using pre-trained tensorflow models
# adapted from "Hands-On Machine Learnong with Scikit-Learn, Keras, and Tensorflow" pp479

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from sklearn.datasets import load_sample_image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions, ResNet50V2
from PIL import Image
import time

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
'''
tf.config.set_visible_devices([], 'GPU')

'''
dog = np.asarray(Image.open('dog.png'))
dog = tf.image.resize(dog, [224,224])
storkPIL = Image.open('stork.jpg')
#storkPIL.show()
stork = np.asarray(storkPIL)
#print('stork: {}'.format(stork))
stork = tf.image.resize(stork, [224,224])
print('stork is a {}'.format(type(stork)))
'''

imageList = []
for path in ['dog.png', 'stork.jpg']:
    img = keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    imageList.append(img_arr)
    #print('img_arr is a {}'.format(img_arr))

model = ResNet50V2(weights="imagenet")

for image in imageList:
    start = time.time()
    final_image = preprocess_input(image)
    #print('final_images: {}'.format(final_images))

    y = model.predict(final_image)
    print(decode_predictions(y))
    print('elapsed time: {:6.3f}'.format(time.time() - start))
