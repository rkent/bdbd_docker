# adapted from https://www.tutorialspoint.com/keras/keras_applications.htm
import tensorflow.keras as keras 
import numpy as np 

from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet 

#Load the VGG model 
vgg_model = vgg16.VGG16(weights = 'imagenet') 

#Load the Inception_V3 model 
inception_model = inception_v3.InceptionV3(weights = 'imagenet') 

#Load the ResNet50 model 
resnet_model = resnet50.ResNet50(weights = 'imagenet') 

#Load the MobileNet model
mobilenet_model = mobilenet.MobileNet(weights = 'imagenet')

import PIL 
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 

filename = '/home/kent/Pictures/banana.jpg' 
## load an image in PIL format 
#original = load_img(filename, target_size = (224, 224)) 
original = load_img(filename)
print('PIL image size', original.size)
#PIL image size (224, 224) 
plt.imshow(original) 
#<matplotlib.image.AxesImage object at 0x1304756d8> 
plt.show()

#convert the PIL image to a numpy array 
numpy_image = img_to_array(original) 

plt.imshow(np.uint8(numpy_image)) 
#<matplotlib.image.AxesImage object at 0x130475ac8> 

print('numpy array size',numpy_image.shape) 
#numpy array size (224, 224, 3) 

# Convert the image / images into batch format 
image_batch = np.expand_dims(numpy_image, axis = 0) 

print('image batch size', image_batch.shape) 
#image batch size (1, 224, 224, 3)


#prepare the image for the resnet50 model 
processed_image = resnet50.preprocess_input(image_batch.copy()) 

# create resnet model 

# get the predicted probabilities for each class 
predictions = resnet_model.predict(processed_image) 

# convert the probabilities to class labels 
label = decode_predictions(predictions)

print(label)
