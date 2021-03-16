# extract regions from an image, and classify objects in those regions

# adapted from https://www.tutorialspoint.com/keras/keras_applications.htm
import numpy as np 
import cv2
import time
import os
from numba import cuda
import rospy
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # stop grabbing all memory
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU

def load_model():
    global keras, decode_predictions, Xception, preprocess_input, model
    import tensorflow as tf
    (free, total) = cuda.current_context().get_memory_info()
    print('free', free, 'total', total)
    if free / 1.e6 < 5000:
        rospy.logwarn('disabling gpu due to inadequate memory')
        tf.config.set_visible_devices([], 'GPU')
    from tensorflow.keras.applications.xception import Xception, decode_predictions, preprocess_input
    model = Xception(weights = 'imagenet')

REPORT_COUNT = 9 # number of objects to report
#XSIZE = 224 # size of classifier image
#YSIZE = 224
XSIZE = 299
YSIZE = 299

def resize_with_pad(im, desired_size=(XSIZE, YSIZE), color=[0, 0, 0]):
    # adapted from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = max(*desired_size)/max(*old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size[1] - new_size[1]
    delta_h = desired_size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

class ObjectClassifier():

    def __init__(self):
        load_model()

    def classify(self, image_np, class_names, scores, xmins, xmaxs, ymins, ymaxs):
        #print('image_np array shape', image_np.shape)
        # gray to color if needed
        if len(image_np.shape) == 2 or image_np.shape[2] != 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            #print('image_np array re-shaped to', image_np.shape)
        # find something that is not a book

        xmin = 0
        ymin = 0
        xmax, ymax, _ = image_np.shape
        object_images = []
        object_scores = []
        object_names = []
        class_names = class_names
        #if len(class_names) == 0:
        #    continue
        for index in range(len(class_names)):
            if len(object_images) >= REPORT_COUNT:
                break
            if class_names[index] == 'book':
                continue
            xmin = xmins[index]
            ymin = ymins[index]
            xmax = xmaxs[index]
            ymax = ymaxs[index]

            image_slice = image_np[ymin:ymax, xmin:xmax, :]

            image_224 = resize_with_pad(image_slice, (XSIZE, YSIZE))

            object_images.append(image_224)
            # these are the previous values from the object detector
            object_scores.append(scores[index])
            object_names.append(class_names[index])

        # apply the tensorforlw classification model to the batch

        # Convert the image / images into batch format
        image_batch = np.array([*object_images])
        start = time.time()
        processed_image = preprocess_input(image_batch.copy())
        # get the predicted probabilities for each class 
        predictions = model.predict(processed_image) 

        # convert the probabilities to class labels 
        labels = decode_predictions(predictions)
        print('classification time {:6.3f}'.format(time.time() - start))
        for i in range(len(labels)):
            print(labels[i][0])
        #id, name, score = labels[0][0]
        #print('imagenet identified {} with score {}'.format(name, score))
        return (labels, object_scores, object_names, object_images)

    def annotate_image(self, labels, object_scores, object_names, object_images):
        count = min(9, len(object_images))
        xpositions = (1, 2, 2, 2, 3, 3, 3, 3, 3)[count - 1]
        ypositions = (1, 1, 2, 2, 2, 2, 3, 3, 3)[count - 1]
        combined_image = np.zeros((YSIZE * ypositions, XSIZE * xpositions, 3), dtype=np.uint8)
        #print('combined_image shape', combined_image.shape)
        for i in range(count):
            xpos = i % xpositions
            ypos = i // xpositions
            color1 = (128, 255, 255)
            score1 = object_scores[i]
            name1 = object_names[i]
            text1 = "{:5.2f} {}".format(score1, name1)
            color2 = (255, 128, 255)
            score2 = labels[i][0][2]
            name2 = labels[i][0][1]
            text2 = "{:5.2f} {}".format(score2, name2)
            display_image = cv2.putText(object_images[i], text1, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, color1, 2, cv2.LINE_AA)
            display_image = cv2.putText(display_image, text2, (5, XSIZE-20), cv2.FONT_HERSHEY_SIMPLEX, .8, color2, 2, cv2.LINE_AA)
            combined_image[YSIZE * ypos:YSIZE * (ypos + 1), XSIZE * xpos: XSIZE * (xpos + 1), :] = object_images[i]
        return combined_image

    def clear(self):
        # clear memory
        cuda.close()
        print('CUDA memory release: GPU0')
