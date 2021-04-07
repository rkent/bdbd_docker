# image using feature detectors
import cv2
import numpy as numpy

img = cv2.imread('../data/pantilt.jpg')
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.SIFT_create(nfeatures=50)
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray, kp, img)
cv2.imshow('SIFT', img)
  
cv2.waitKey(0) 
#cv2.imwrite('../data/sr305_sift.jpg',img)
