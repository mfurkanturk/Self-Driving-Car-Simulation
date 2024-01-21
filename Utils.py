import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.utils import shuffle
from imgaug import augmenters as ima
import random
import cv2

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from keras.optimizers import Adam

def getName(path):
    return path.split("\\")[-1]


def importData(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    dataframe = pd.read_csv(os.path.join(path, "driving_log.csv"), names=columns)
    # print(dataframe.head())
    # print(getName(dataframe["Center"][0]))
    dataframe["Center"] = dataframe["Center"].apply(getName)
    # print(dataframe.head())
    print("Total images imported: ", dataframe.shape[0])
    return dataframe


def dataPartition(data, display=True):
    samples_per_bin = 1250
    nBins = 31
    hist, bins = np.histogram(data["Steering"], bins=nBins)
    center = (bins[:-1] + bins[1:]) * 0.5
    if display:
        print(center)
        plt.bar(center, hist, width=0.08)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()

    removeIndexList = []

    for n in range(nBins):
        binDataList = []

        for i in range(len(data["Steering"])):
            if data["Steering"][i] >= bins[n] and data["Steering"][i] <= bins[n + 1]:
                binDataList.append(i)

        binDataList = shuffle(binDataList)
        binDataList = binDataList[samples_per_bin:]
        removeIndexList.extend(binDataList)

    print("Removed Images: ", len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print("Remaining images: ", len(data))
    if display:
        hist, _ = np.histogram(data["Steering"], bins=nBins)
        plt.bar(center, hist, width=0.08)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()
    return data


def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagesPath.append(os.path.join(path, "IMG", indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)

    return imagesPath, steering


def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)

    # pan
    if np.random.rand() < 0.5:
        pan = ima.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    # zoom
    if np.random.rand() < 0.5:
        zoom = ima.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    # brightness
    if np.random.rand() < 0.5:
        brightness = ima.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)

    # flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


def prprocessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


# imgP = prprocessing(mpimg.imread("test.jpg"))

# plt.imshow(imgP)
# plt.show()


def batchGen(imagesPath, steeringList, batchsize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        for i in range(batchsize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = prprocessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation="elu"))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))

    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))

    model.compile(Adam(learning_rate=1e-3), loss="mean_squared_error")
    return model
