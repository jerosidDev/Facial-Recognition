
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import pathlib
import json


def addFaceData():

    currentProjectDir = os.getcwd()

    imagePath = input("Enter the full path to the image:")

    img = cv.imread(imagePath,cv.IMREAD_UNCHANGED)
    try:
        faces = showAllFaces(img,currentProjectDir)
    except cv.error:
        print(f"incorrect path to image: {imagePath}")
        return

    imgNamesStr = input("Which known faces were in the picture? Enter the names separated with commas:")
    imgNames = [n.strip() for n in imgNamesStr.split(",")]
    imgNames.extend(["Unknown","Not a face"])

    initialImage = cv.imread(imagePath,cv.IMREAD_UNCHANGED);

    faceItems = []
    for face in faces:
        faceName = displayFaceSelection(initialImage,face,imgNames)
        x,y,w,h = face
        faceItems.append({ "name": faceName , "frame" : {"x":str(x),"y":str(y),"w":str(w),"h":str(h)} })

    newFaceData = { "filePath": imagePath, "faceItems" : faceItems}
    saveFaceData(newFaceData)


def getFaceCascade(projectDir):
    opencv_data_dir = f"{projectDir}\opencv_data"
    os.chdir(opencv_data_dir)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    os.chdir(projectDir)
    return face_cascade

def showAllFaces(img,currentProjectDir):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_cascade = getFaceCascade(currentProjectDir)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv.namedWindow('img', cv.WINDOW_GUI_EXPANDED)
    cv.imshow('img', img)
    cv.waitKey(5000)
    cv.destroyAllWindows()

    return faces

def displayFaceSelection(img,face,imgNames):

    x,y,w,h = face
    faceImg = img[y:y+h,x:x+w]

    fig = plt.figure(figsize=(36,24))

    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    radioButton = RadioButtons(rax, imgNames)
    radioButton.on_clicked(applyNameToFace)

    fig.add_subplot(1, 3, 2)
    plt.imshow(faceImg)
    plt.show()

    return radioButton.value_selected

def applyNameToFace(label):
    plt.close()
    print(label)

def saveFaceData(newFaceData):

    pathlib.Path('./face_data').mkdir(parents=True, exist_ok=True)  

    dataFilePath = './face_data/face_data_training.json'
    if not(os.path.exists(dataFilePath)):
        allData = np.array([]);
    else:
        with open(dataFilePath, mode='rt', encoding='utf-8') as dataFile:
            allData = json.load(dataFile)

    allFilePath = [ faceData["filePath"] for faceData in allData ]
    newFilePath = newFaceData["filePath"]
    if newFilePath in allFilePath:
        i = allFilePath.index(newFilePath)
        del allData[i]

    allData = np.append(allData,newFaceData)

    with open(dataFilePath, mode='wt', encoding='utf-8') as dataFile:
        json.dump( allData.tolist(), dataFile)

