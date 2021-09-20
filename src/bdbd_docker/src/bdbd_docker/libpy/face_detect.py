# Face detection in an image, preliminary to face recognition
# See https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
#!sudo pip3 install mtcnn

# function for face detection with mtcnn
from mtcnn.mtcnn import MTCNN
import cv2

# pixels is a numpy array, RGB color space. Results will show RGB
def extract_faces(pixels, required_size=(160, 160), min_size=(100, 100), min_confidence=0.95):
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    faces = []

    for result in results:
        # extract the bounding box from the face
        x1, y1, width, height = result['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        if width < min_size[0] or height < min_size[1] or result['confidence'] < min_confidence:
            continue
        image = cv2.resize(face, required_size)
        faces.append(image)
    return faces, results

# test and demo
if __name__ == '__main__':
    # load image from file
    filename = '../data/person.jpg'
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    faces, results = extract_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            min_size=(50, 50), min_confidence=0.95)
    for result in results:
        print(result)
    for face in faces:
        print(face.shape)
        cv2.imshow('matches', face)
        cv2.waitKey(0) 
