import cv2
import os
import numpy as np

posdir = "./violence/train_data/Pos/1/"
negdir = "./violence/train_data/Neg/0/"

def data_input():
    posfiles = os.listdir(posdir)
    negfiles = os.listdir(negdir)
    x_data = np.zeros([50, 227, 227, 3])
    y_data = np.zeros([50])
    for i in range(25):
        r = np.random.rand()
        pp = int(r * (posfiles.__len__() - 1))
        n_p = int(r * (negfiles.__len__() - 1))
        img = cv2.imread(posdir + posfiles[pp], cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.resize(img, (227, 227))
        x_data[2 * i,] = img
        y_data[2 * i] = 1
        img = cv2.imread(negdir + negfiles[n_p], cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.resize(img, (227, 227))
        x_data[2 * i + 1,] = img
        y_data[2 * i + 1] = 0
    return x_data, y_data

def alldata_input():
    posfiles = os.listdir(posdir)
    negfiles = os.listdir(negdir)
    filenum = posfiles.__len__()+negfiles.__len__()
    x_data = np.zeros([filenum, 227, 227, 3])
    y_data = np.zeros([filenum])
    for i in range(posfiles.__len__()):
        img = cv2.imread(posdir + posfiles[i], cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.resize(img, (227, 227))
        x_data[i,] = img
        y_data[i] = 1
    for i in range(negfiles.__len__()):
        img = cv2.imread(negdir + negfiles[i], cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.resize(img, (227, 227))
        x_data[posfiles.__len__()+i,] = img
        y_data[posfiles.__len__()+i] = 1
    return x_data, y_data