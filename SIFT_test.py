import cv2 as cv
import numpy as np
  
img1 = cv.imread('E:/Images/book_test.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('E:/Images/book_test2.jpg', cv.IMREAD_GRAYSCALE)

img1 = cv.resize(img1,(0,0),fx=0.8,fy=0.8, interpolation=cv.INTER_CUBIC)
img2 = cv.resize(img2,(0,0),fx=0.3,fy=0.3, interpolation=cv.INTER_CUBIC)

sift = cv.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

img1 = cv.drawKeypoints(img1, keypoints_1, None)
img2 = cv.drawKeypoints(img2, keypoints_2, None)

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:100], None, flags=2)

# cv.imshow('Original image 1', img1)
# cv.imshow('Original image 2', img2)
cv.imshow('Matching result', matching_result)

cv.waitKey(0)
cv.destroyAllWindows()
