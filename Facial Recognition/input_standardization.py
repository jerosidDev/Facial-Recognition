import skimage
from skimage import data
import skimage.transform as stransfor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import torchvision
import torch


#greyFace = skimage.color.rgb2gray(face_resized)

## todo: rescale the image to have pixel values between 0 and 1
## todo: substract the mean and divide by the standard deviation => normalization


def generateStandardInputs():

    dataFilePath = './face_data/face_data_training.json'
    with open(dataFilePath, mode='rt', encoding='utf-8') as dataFile:
        allData = json.load(dataFile)

    allFilePath = [ faceData["filePath"] for faceData in allData ]

    inputSquareSize = 100
    allImages = []
    plt.figure()
    for faceData in allData:
        filePath = faceData["filePath"]
        fullImage = mpimg.imread(filePath)
        for faceItem in faceData["faceItems"]:
            faceImage = rescaleImage(fullImage,faceItem,inputSquareSize)
            allImages.append([faceItem["name"],faceImage])

    faceList = [ faceIm[1] for faceIm in allImages]
    npGrid = np.array(faceList)
    # format: Batch, Channel, Height, Width
    npGrid = np.transpose(npGrid,(0,3,1,2))

    gridTensor = torch.Tensor(npGrid);

    gridImg = torchvision.utils.make_grid(gridTensor)
    gridImg = np.transpose(gridImg,(1,2,0))
    plt.figure()
    plt.imshow(gridImg, cmap="gray")
    plt.show()


def rescaleImage(fullImage,faceItem,squareSize):
    x = int(faceItem["frame"]["x"])
    y = int(faceItem["frame"]["y"])
    w = int(faceItem["frame"]["w"])
    h = int(faceItem["frame"]["h"])

    initialSquareSize = min(w,h)
    xCenter = x + w/2
    yCenter = y + h/2
    xMin, xMax, yMin, yMax = (xCenter - initialSquareSize/2, xCenter + initialSquareSize/2, yCenter - initialSquareSize/2, yCenter + initialSquareSize/2)
    initialSquare = fullImage[int(yMin):int(yMax),int(xMin):int(xMax)]

    newSquare = stransfor.resize(initialSquare,(squareSize,squareSize))

    #plt.imshow(newSquare)
    #plt.show()
    return newSquare