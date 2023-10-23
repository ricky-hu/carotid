import os
from shutil import copy, rmtree
import random
import pandas
import openpyxl


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # remove the old path if it appears
        rmtree(file_path)
    os.makedirs(file_path)


def main(folder: str):
    # separate data into 80% and 20%
    split_rate = 0.2

    # point to all the data from one side
    cwd = os.getcwd()  # cwd is the directory this file in
    data_root = os.path.join(cwd, "\\", folder)
    origin_image_path = os.path.join(data_root, "origin")
    assert os.path.exists(origin_image_path), "path '{}' does not exist.".format(origin_image_path)

    # outcome_class = [cla for cla in os.listdir(origin_image_path)
    #                  if os.path.isdir(os.path.join(origin_image_path, cla))]
    # print(outcome_class)

    outcome_class = ['0', '1']
    # the train set
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in outcome_class:
        # folders in train folder for clssses
        mk_file(os.path.join(train_root, cla))

    # the validation set
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in outcome_class:
        # folders for classes in validation folder
        mk_file(os.path.join(val_root, cla))

    for cla in outcome_class:
        cla_path = os.path.join(origin_image_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_rate))
        # assign images into corresponding menu
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main("dataset")
