import os

import pandas as pd
import efficientnet.path as path

import torchvision.transforms as transforms
from PIL import Image
import os
import PIL
import glob

# the function paris the image route with the label
def pair(folder: str, excel: str, sheet: str, col1: str, col2: str):
    """
    :param folder:
    :param excel: the name of the excel file
    :param sheet: the sheet inside the file
    :param col: the two columns will be extracted
    :return:
    """
    img = []  # the list contains the path to each image
    label = []  # the list contains the label corresponding to each img

    cols = col1 + "," + col2
    dataRead = pd.read_excel(excel, sheet_name=sheet, usecols=cols, engine='openpyxl')
    name = dataRead.columns.values  # the list with two elements: the title of the table

    n = dataRead[[name[0]]].values.tolist()
    column1 = []
    for e in n:
        column1.append(e[0])

    c = dataRead[[name[1]]].values.tolist()
    column2 = []
    for e in c:
        column2.append(e[0])  # the data type is int

    # carotid_data = carotid_data.to_dict()
    carotid_data = dataRead.values.tolist()  # a 2d list with pairs of images and label

    dic = {}
    for n, c in zip(column1, column2):
        dic[n] = c  #

    # print(len(dic))
    imgs = os.listdir(folder)
    for i in imgs:
        img.append(folder + "\\{}".format(i))
        label.append(dic[i])
        # print(i, dic[i])
        # label.append(dic2[i])

    return img, label, dic


# the function acts like join in SQL
def merge(excel1, sheet1, col1):
    dataRead1 = pd.read_excel(excel1, sheet_name=sheet1, usecols=col1, engine='openpyxl')
    name = dataRead1.columns.values
    a = dataRead1[name[0]].values.tolist()

    b = []
    d = {}
    for e in a:
        f = e
        e = e.translate({ord(i): None for i in 'abcdefghijklemnopqrstuvwxyz_-.ABCDEFGHIJKLMNOPQRSTUVWXYZ'})
        if len(e) >= 7:
            e = e[:7]
        b.append([e, f])
        d[e] = f
    # print(b)

    cols = "A, B"
    dataRead2 = pd.read_excel(path.CAROTIDDATA, sheet_name="Sheet1", usecols=cols, engine='openpyxl')
    name2 = dataRead2.columns.values  # the list with two elements: the title of the table

    n = dataRead2[[name2[0]]].values.tolist()
    column1 = []
    for e in n:
        column1.append(e[0])

    c = dataRead2[[name2[1]]].values.tolist()
    column2 = []
    for e in c:
        column2.append(e[0])  # the data type is int

    dic = {}
    for n, c in zip(column1, column2):
        dic[n] = c

    # print(dic)
    result = {}
    # print(len(d), len(dic))
    for e in b:
        # print(e[0])
        e0 = e[0]
        e1 = e[1]
        result[e1] = dic[int(e0)]
    # print(result)
    return result


def pair2(folder: str, excel: str, sheet: str, col1: str, col2: str):
    """
    :param folder:
    :param excel: the name of the excel file
    :param sheet: the sheet inside the file
    :param col: the two columns will be extracted
    :return:
    """
    img = []  # the list contains the path to each image
    label = []  # the list contains the label corresponding to each img

    cols = col1 + "," + col2
    dataRead = pd.read_excel(excel, sheet_name=sheet, usecols=cols, engine='openpyxl')
    name = dataRead.columns.values  # the list with two elements: the title of the table

    n = dataRead[[name[0]]].values.tolist()
    column1 = []
    for e in n:
        column1.append(e[0])

    c = dataRead[[name[1]]].values.tolist()
    column2 = []
    for e in c:
        column2.append(e[0])  # the data type is int

    # carotid_data = carotid_data.to_dict()
    carotid_data = dataRead.values.tolist()  # a 2d list with pairs of images and label

    dic = {}
    for n, c in zip(column1, column2):
        dic[n] = c  #

    dic2 = merge(path.TRAIN_VAL_TEST, sheet, "A")

    # print(len(dic))
    imgs = os.listdir(folder)
    for i in imgs:
        img.append(folder + "\\{}".format(i))
        # label.append(dic[i])
        label.append(dic2[i])

    return img, label, dic2


leftTrainBU = pair(path.LEFTTRAIN, path.TRAIN_VAL_TEST, "Train Data - Left Bulb", "A", "B")
leftValBU = pair(path.LEFTVAL, path.TRAIN_VAL_TEST, "Val Data - Left Bulb", "A", "B")
leftTestBU = pair(path.LEFTTEST, path.TRAIN_VAL_TEST, "Test Data - Left Bulb", "A", "B")
rightTrainBU = pair(path.RIGHTRAIN, path.TRAIN_VAL_TEST, "Train Data - Right Bulb", "A", "B")
rightValBU = pair(path.RIGHTVAL, path.TRAIN_VAL_TEST, "Val Data - Right Bulb", "A", "B")
rightTestBU = pair(path.RIGHTTEST, path.TRAIN_VAL_TEST, "Test Data - Right Bulb", "A", "B")

leftTrainAA = pair2(path.LEFTTRAIN, path.TRAIN_VAL_TEST, "Train Data - Left Bulb", "A", "B")
leftValAA = pair2(path.LEFTVAL, path.TRAIN_VAL_TEST, "Val Data - Left Bulb", "A", "B")
leftTestAA = pair2(path.LEFTTEST, path.TRAIN_VAL_TEST, "Test Data - Left Bulb", "A", "B")
rightTrainAA = pair2(path.RIGHTRAIN, path.TRAIN_VAL_TEST, "Train Data - Right Bulb", "A", "B")
rightValAA = pair2(path.RIGHTVAL, path.TRAIN_VAL_TEST, "Val Data - Right Bulb", "A", "B")
rightTestAA = pair2(path.RIGHTTEST, path.TRAIN_VAL_TEST, "Test Data - Right Bulb", "A", "B")

# maskLeftTrain = pair(path.MASKLEFTTRAIN, path.CD, "Sheet1", "A", "B")

import numpy as np
def re():
    # (851, 564)
    # (850, 550)
    img = os.listdir(path.RIGHTTEST)
    img2 = os.listdir(path.LEFTTEST)
    image2 = Image.open(path.DATA + "\\leftTest\\" + img2[0])
    image = Image.open(path.DATA + "\\rightTest\\" + img[0])
    # print(np.squeeze(image))
    # print(np.squeeze(image2))
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)
    img_tensor2 = transform(image2)
    # print(img_tensor)
    # print(img_tensor2)


    for i in img:
        image = Image.open(path.DATA + "\\rightTest\\" + i)
        # image.reshape(image, [3,3,3,24])

        # resize = image.resize((850, 550))
        # print(np.squeeze(image))
        # print(resize.size)
        # resize.save(path.DATA + "\\rightVal\\" + i)

if __name__ == "__main__":
    # ['Train Data - Right Bulb', 'Val Data - Right Bulb', 'Test Data - Right Bulb',
    # 'Train Data - Left Bulb', 'Val Data - Left Bulb', 'Test Data - Left Bulb']
    # print(pair(path.LEFTTEST, path.TRAIN_VAL_TEST, "Test Data - Left Bulb", "A", "B"))
    # print(merge(path.TRAIN_VAL_TEST, "Train Data - Left Bulb", "A"))
    # leftTestBU
    re()
    # print(leftValAA)