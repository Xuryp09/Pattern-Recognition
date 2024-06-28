import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    detectData_name = []
    detectData = []
    with open('data/detect/detectData.txt') as f:
        for line in f:
            s = line.split(' ')
            if(len(s) != 2):
                str_to_int = []
                for str in s:
                    str_to_int.append(int(str))
                detectData[len(detectData_name) - 1].append(str_to_int)
            else:
                detectData_name.append(s[0])
                detectData.append([])

    for i in range(len(detectData_name)):
        image = cv2.imread("data/detect/" + detectData_name[i])
        image_gray = cv2.imread("data/detect/" + detectData_name[i], cv2.IMREAD_GRAYSCALE)
        for j in range(len(detectData[i])):
            x, y, w, h = detectData[i][j]
            face_image = cv2.resize(image_gray[y : y + h, x : x + w], (19, 19), interpolation=cv2.INTER_CUBIC)
            if clf.classify(face_image) == 1:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            else:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        cv2.imwrite("result/test/test_" + detectData_name[i], image)
