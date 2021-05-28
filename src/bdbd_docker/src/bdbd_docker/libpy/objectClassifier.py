# extract regions from an image, and classify objects in those regions

# adapted from https://www.tutorialspoint.com/keras/keras_applications.htm
import numpy as np 
import cv2
import time
import os
from numba import cuda
import rospy
import math
from bdbd_common.utils import fstr, sstr
import traceback

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # stop grabbing all memory
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # disable GPU

def load_model():
    global keras, decode_predictions, Xception, preprocess_input, model, XSIZE, YSIZE
    import tensorflow as tf
    free = None
    try:
        (free, total) = cuda.current_context().get_memory_info()
    except Exception as exception:
        rospy.logerr('cuda error, disabling: {}'.format(exception))
    if free is None or free / 1.e6 < 5000:
        rospy.logwarn('disabling gpu due to inadequate memory or error')
        tf.config.set_visible_devices([], 'GPU')
    #from tensorflow.keras.applications.xception import Xception, decode_predictions, preprocess_input
    #model = Xception(weights = 'imagenet')
    from tensorflow.keras.applications.efficientnet import decode_predictions, preprocess_input
    from tensorflow.keras.applications.efficientnet import EfficientNetB1 as model_class
    model = model_class(weights = 'imagenet')
    (_, XSIZE, YSIZE, _) = model.input_shape
    
def resize_with_pad(im, desired_size=None, color=[0, 0, 0]):
    if desired_size is None:
        desired_size=(XSIZE, YSIZE)
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

def best_label(labels, reject_names=['home_theater', 'nematode']):
    print('checking {} labels'.format(len(labels)))
    # pick the highest match, and refine object
    best_score = 0.0
    best_i = -1
    best_name = ''
    for i, label in enumerate(labels):
        name = label[0][1]
        if name in reject_names:
            continue
        score = label[0][2]
        if score > best_score:
            best_name = name
            best_score = score
            best_i = i
    return (best_i, best_score, best_name)

def sort_labels(labels, reject_names=['home_theater', 'nematode']):
    # extend to include position
    sorted_labels = []
    for (i, label) in enumerate(labels):
        sorted_labels.append({'name': label[0][1], 'score': label[0][2], 'index': i})
    sorted_labels = list(filter(lambda label: not label['name'] in reject_names, sorted_labels))
    sorted_labels.sort(reverse=True, key=lambda label: label['score'])
    return sorted_labels

class ObjectClassifier():

    def __init__(self):
        load_model()

    def classify(self, image_np, class_names, scores, xmins, xmaxs, ymins, ymaxs,
            do_pad=True, aspect_limit=1.5, report_count=10):
        # gray to color if needed
        if len(image_np.shape) == 2 or image_np.shape[2] != 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        xmin = 0
        ymin = 0
        xmax, ymax, _ = image_np.shape
        object_images = []
        object_scores = []
        object_names = []
        for index in range(len(xmins)):
            if len(object_images) >= report_count:
                break
            # too many books are seen which are not interesting
            try:
                if class_names and class_names[index] == 'book':
                    continue
            except:
                pass
            xmin = xmins[index]
            ymin = ymins[index]
            xmax = xmaxs[index]
            ymax = ymaxs[index]
            (MAXY, MAXX, _) = image_np.shape
            # limit aspect ratio of slice
            xd = xmax - xmin
            yd = ymax - ymin
            xc = (xmax + xmin) / 2
            yc = (ymax + ymin) / 2
            if xd / yd > aspect_limit:
                yd = xd / aspect_limit
            elif yd / xd > aspect_limit:
                xd = yd / aspect_limit
            xmin = max(0, int(xc - xd / 2))
            xmax = min(MAXX-1, int(xmin + xd))
            ymin = max(0, int(yc - yd / 2))
            ymax = min(MAXY-1, int(ymin + yd))

            image_slice = image_np[ymin:ymax, xmin:xmax, :]

            if do_pad:
                image_224 = resize_with_pad(image_slice, (XSIZE, YSIZE))
            else:
                image_224= cv2.resize(image_slice, (XSIZE, YSIZE), interpolation=cv2.INTER_AREA)

            object_images.append(image_224)
            # these are the previous values from the object detector
            if isinstance(scores, list) or isinstance(scores, tuple):
                score = scores[index]
            else:
                if scores:
                    score = scores
                else:
                    score = 0.0
            if isinstance(class_names, list) or isinstance(class_names, tuple):
                name = class_names[index]
            else:
                if class_names:
                    name = class_names
                else:
                    name = 'None'
            object_scores.append(score)
            object_names.append(name)

        # apply the tensorflow classification model to the batch

        # Convert the image / images into batch format
        image_batch = np.array([*object_images])
        start = time.time()
        processed_image = preprocess_input(image_batch.copy())
        # get the predicted probabilities for each class 
        predictions = model.predict(processed_image) 

        # convert the probabilities to class labels 
        labels = decode_predictions(predictions)
        print('classification time {:6.3f} seconds'.format(time.time() - start))
        #for i in range(len(labels)):
        #    print(labels[i][0])
        #id, name, score = labels[0][0]
        #print('imagenet identified {} with score {}'.format(name, score))
        return (labels, object_scores, object_names, object_images)

    def annotate_image(self, labels, object_scores, object_names, object_images, min_score=0.25, do_sort=True):
        # sort images by best of either score
        bestscores = []
        for i in range(len(object_scores)):
            bestscores.append({'index': i, 'score': max(object_scores[i], labels[i][0][2])})
        if do_sort:
            bestscores.sort(reverse=True, key=lambda bestscores: bestscores['score'])
        count = 0
        for bestscore in bestscores:
            if bestscore['score'] < min_score or count == 9:
                break
            count += 1
        #from bdbd_common.utils import fstr
        #print(fstr(bestscores))
        #count = min(9, len(bestscores))
        xpositions = (1, 2, 2, 2, 3, 3, 3, 3, 3)[count - 1]
        ypositions = (1, 1, 2, 2, 2, 2, 3, 3, 3)[count - 1]
        combined_image = np.zeros((YSIZE * ypositions, XSIZE * xpositions, 3), dtype=np.uint8)
        #print('combined_image shape', combined_image.shape)
        for j in range(count):
            i = bestscores[j]['index']
            xpos = j % xpositions
            ypos = j // xpositions
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
        try:
            cuda.close()
            print('CUDA memory release: GPU0')
        except:
            print('Failed to clear cuda')
        rospy.logwarn(traceback.format_stack())

    # refine an image with multiple calls to classifier
    def refine_class(self, image_np, xd, yd, xc, yc,
            delta_factor=0.2, aspect_limit=1.5, do_pad=True,
            first_score=0.0, first_name='Unknown'):
        (MAXY, MAXX, _) = image_np.shape
        dxd = delta_factor * xd
        dyd = delta_factor * yd
        # limit aspect ratio of most severe areas
        if xd - dxd < (yd + dyd) / aspect_limit:
            xd = dxd + (yd + dyd) / aspect_limit
        elif yd - dyd < (xd + dxd) / aspect_limit:
            yd = dyd + (xd + dxd) / aspect_limit

        # revise positions to account for edges
        minx = max(0, xc - xd / 2 - dxd)
        miny = max(0, yc - yd / 2 - dyd)
        maxx = min(MAXX - 1, xc + xd / 2 + dxd)
        maxy = min(MAXY - 1, yc + yd / 2 + dyd)
        xc = (minx + maxx) / 2
        yc = (miny + maxy) / 2

        # scale dxd, dyd to account for limits
        dxd = (maxx - minx) / (2 + 1. / delta_factor)
        dyd = (maxy - miny) / (2 + 1. / delta_factor)

        # prepare regions
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        for imin in range(0,3):
            for imax in range(0,3):
                for jmin in range(0,3):
                    for jmax in range(0,3):
                        xmins.append(int(minx + imin * dxd))
                        xmaxs.append(int(maxx - imax * dxd))
                        ymins.append(int(miny + jmin * dyd))
                        ymaxs.append(int(maxy - jmax * dyd))

        (labels, object_scores, object_names, object_images) = \
            self.classify(
                image_np, first_name, first_score,
                xmins, xmaxs, ymins, ymaxs,
                aspect_limit=aspect_limit, do_pad=do_pad, report_count=81
            )
        combined_image = self.annotate_image(labels, object_scores, object_names, object_images, min_score = 0.0, do_sort=True)
        (best_i, best_score, best_name) = best_label(labels)
        xc = (xmins[best_i] + xmaxs[best_i]) / 2
        yc = (ymins[best_i] + ymaxs[best_i]) / 2
        xd = (xmaxs[best_i] - xmins[best_i])
        yd = (ymaxs[best_i] - ymins[best_i])
        return (best_score, best_name, xd, yd, xc, yc, combined_image)

if __name__ == '__main__':
    # test
    load_model()
    print(model.input_shape)