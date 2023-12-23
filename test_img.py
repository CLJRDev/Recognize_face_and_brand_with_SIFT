import cv2 as cv
import os
import time


def extractDescriptors(images):
  desList = []
  sift = cv.SIFT_create() 
  for img in images:
    kp, des = sift.detectAndCompute(img, None)
    desList.append(des)
  return desList

def findID(img, desList, thres=30):
  sift = cv.SIFT_create()
  kp2, des2 = sift.detectAndCompute(img, None)
  bf = cv.BFMatcher()
  matchList = []
  finalVal = -1
  try:
    for des in desList:
      matches = bf.knnMatch(des, des2, k=2)
      good = []
      for m,n in matches:
        if m.distance < 0.75 * n.distance:
          good.append([m])
      matchList.append(len(good))
  except:
    pass
  print(matchList)
  if len(matchList) != 0:
    if max(matchList) > thres:
      finalVal = matchList.index(max(matchList))
  return finalVal

path = 'logo'
images = []
class_name = []

my_list = os.listdir(path)
print(my_list)
print('Total classed detected:', len(my_list))


for folder in os.listdir(path):
  for file in os.listdir(os.path.join(path,folder)):
    img = cv.imread(os.path.join(path, folder, file),cv.COLOR_BGR2GRAY)
    images.append(img)
    class_name.append(folder)
print(class_name)

  
desList = extractDescriptors(images)
print(len(desList))

test_path= r'logo_resize'
for file in os.listdir(test_path):
  img_test = cv.imread(os.path.join(test_path, file))
  original_img = img_test.copy()
  img_test = cv.cvtColor(img_test, cv.COLOR_BGR2RGB)
  id = findID(img_test, desList, 60)
  if id != -1:
    cv.putText(original_img, class_name[id],(50,50),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
  cv.imshow('Output',original_img)
  new_image_path = r'ket_qua/' + class_name[id] + '.jpg'  
  cv.imwrite(new_image_path, original_img)
  # k = cv.waitKey(0)
  # if k == ord('q'):
  #   continue

  
cv.destroyAllWindows()
