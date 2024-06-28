import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    dataset = []
    for image in os.listdir(str(dataPath) + '/face/'):
        img = cv2.imread(str(dataPath) + '/face/' + image, cv2.IMREAD_GRAYSCALE)
        dataset.append((img, 1))

    for image in os.listdir(str(dataPath) + '/non-face/'):
        img = cv2.imread(str(dataPath) + '/non-face/' + image, cv2.IMREAD_GRAYSCALE)
        dataset.append((img, 0))

    return dataset