import os
from cv2 import cv2
import numpy as np
from PIL import Image
import sqlite3

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'


def getImageWithId(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        Ids.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return np.array(Ids), faces


Ids, faces = getImageWithId(path)
recognizer.train(faces, Ids)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
