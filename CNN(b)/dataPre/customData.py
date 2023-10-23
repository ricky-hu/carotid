import os

import pandas as pd
import efficientnet.path as path


### Images are stored in one folder:
### Two conditions could occur:
### 1. the folder has images and nothing else -> the folder will be seperated into three folders: [train, val, test]
###
### 2. the folder has folders inside, each each folder is one class
### Name of the data should be in an excel
### Data should be in one sheet
###

class CustomData:
    def __init__(self, img: str,
                 excel: str,
                 sheet: str,
                 col: str,
                 random: str = True,
                 classified: bool = False):
        """
        :param img: the folder contains all the image, the img folder has two scenarios:
                     1. the folder contains all the images
                     2. the folder contains the folder of images classified by the classes
        :param excel: the excel contains the data of image
        :param sheet: the designated sheet
        :param col: the primary key of the excel, should be the name of the image
        :param classified: if the images are separated in different folders depends on the classes
        """
        self.img = img
        self.excel = excel
        self.sheet = sheet
        self.col = col
        self.images = os.listdir(img)  # the names of images contained in the folder
        self.random = random
        self.classes = []
        self.classified = classified

        self.train, self.val, self.test = self.split(0.75, 0.15, 0.1)

    def setClass(self, label: str):
        """
        Read one column from the column in the same excel and same sheet as the primary key
        :param label: label should be in the same
        :return:
        """
        # set the class attribute first
        if self.classified is True:  # the images are already in the corresponding folders
            self.classes = os.listdir(self.img)
        else:
            classes = set(self.readOneColumn(self.excel, self.sheet, label))
            for i in classes:
                self.classes.append(i)
            # put images into corresponding folders
            for c in classes:
                os.makedirs()

        # print(self.classes)

    # maybe these two methods can be merged into one
    def proportioalSplit(self, train: float, val: float, test: float):
        # count the number from each classes and split the data proportional
        return "train address", "val address", "test address"

    def split(self, train: float, val: float, test: float):
        if self.random is True:
            pass
        else:
            pass
        # produce three folders
        # should also give excel file
        return "train address", "val address", "test address"

    @staticmethod
    def readOneColumn(table: str, sh: str, col: str):
        """
        :param sh: the sheet name
        :param table: the excel file
        :param col: the column will be read
        :return:
        """
        dataRead = pd.read_excel(table, sheet_name=sh, usecols=col, engine='openpyxl')
        title = dataRead.columns.values[-1]
        temp = dataRead[[title]].values.tolist()  # the data from excel as 2d list
        data = [e[0] for e in temp]

        return data

    @staticmethod
    def checkMatch(folder: str, table, sh, col):
        """
        Check if the file in folders match the names in the excel
        :param folder:
        :param table: what type of folder is this: [img, train, val, test]
        :return:
        """
        # images = os.listdir(self.img)
        images = os.listdir(folder)
        name = CustomData.readOneColumn(table, sh, col)
        if len(images) != len(name):
            return "Number of images is different"

        for n in name:
            if n not in images:
                assert "{} not in ".format(n)
                return False

        for i in images:
            if i not in name:
                assert "{} not in ".format(i)
                return False

        return True

    # the problem of the function is that only paired data can be paired
    def pair(self, folder: str, table: str, sh: str, labelCol: str):
        """
        The function matches the image in the folder to image and its label in excel
        :param folder:
        :param labelCol:
        :return:
        """

        self.checkMatch(folder, table, sh, labelCol)
        col1 = self.readOneColumn(self.excel, self.sheet, self.col)
        label = self.readOneColumn(self.excel, self.sheet, labelCol)

        img = [self.img + e for e in col1]  # the path of the image
        dic = {}
        for e, l in zip(img, label):
            dic[e] = l

        return img, label, dic

    # def imgToLabel(self, labelCol: str):
    #     images = os.listdir(self.img)
    #     label = self.pair(labelCol)[1]
    #     imgPath = [self.img + "\\{}".format(i) for i in images]
    #
    #     label2 = []
    #     temp = self.pair(labelCol)[2]
    #     print(temp)
    #     for i in images:
    #         label2.append(temp[i])
    #
    #
    #     return imgPath, label, label2


leftTrainBU = CustomData(path.LEFTTRAIN, path.TRAIN_VAL_TEST, "Train Data - Left Bulb", "A")
leftValBU = CustomData(path.LEFTVAL, path.TRAIN_VAL_TEST, "Vsl Data - Left Bulb", "A")
leftTestBU = CustomData(path.LEFTTEST, path.TRAIN_VAL_TEST, "Test Data - Left Bulb", "A")
rightTrainBU = CustomData(path.RIGHTTEST, path.TRAIN_VAL_TEST, "Train Data - Right Bulb", "A")
rightValBU = CustomData(path.RIGHTVAL, path.TRAIN_VAL_TEST, "Vsl Data - Right Bulb", "A")
rightTestBU = CustomData(path.RIGHTTEST, path.TRAIN_VAL_TEST, "Test Data - Right Bulb", "A")


if __name__ == "__main__":

    leftTestBU = CustomData(path.LEFTTEST, path.TRAIN_VAL_TEST, "Test Data - Left Bulb", "A")
    print(leftTestBU.readOneColumn(path.TRAIN_VAL_TEST, "Test Data - Left Bulb", "B"))
    # print(leftTestBU.classes)
    # print(custom.readOneColumn("Test Data - Left Bulb", "A"))
    # print(leftTestBU.pair(path.LEFTTEST, "B"))
