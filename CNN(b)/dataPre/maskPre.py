import cv2, os
import pandas as pd
import shutil
import tensorflow as tf

# This file works for mask data preprocessing
base_path = "./images_mask/right_images/"
new_path = "./images_mask/right"

leftTrain = "./images_mask/leftTrain"
leftVal = "./images_mask/leftVal"
leftTest = "./images_mask/leftTest"

rightTrain = "./images_mask/rightTrain"
rightVal = "./images_mask/rightVal"
rightTest = "./images_mask/rightTest"

combineTrain = "./images_mask/combineTrain"
combineVal = "./images_mask/combineVal"
combineTest = "./images_mask/combineTest"


# unsegLeftTrain = "./unsegmented/leftTrain"
# for infile in os.listdir(base_path):
#     print("file : " + infile)
#
#     read = cv2.imread(base_path + infile)
#     outfile = infile.split('.')[0] + '.jpg'
#     cv2.imwrite(new_path+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])

# for infile in os.listdir(new_path):
#     old_file = os.path.join(new_path, infile)
#     new_file = os.path.join(new_path, infile[8:])
#     os.rename(old_file, new_file)


def splitType(origin, sheet, des):
    file = os.listdir(origin)
    name = [f[:7] for f in file]  # type is str
    # print(file)

    dataRead = pd.read_excel("MASK-train_val_test_split.xlsx", sheet_name=sheet, usecols="A", engine='openpyxl')
    split = dataRead.values
    # print(split)
    s = [f[0] for f in split]  # the file supposed to in the folder
    print(len(s))

    for e in file:
        # print(type(e[:7]))
        if int(e[:7]) in s:
            print(e)
            shutil.move(origin + '/' + e, des + '/' + e)


def match(folder):
    file = os.listdir(folder)
    dataRead = pd.read_excel("CD.xlsx", sheet_name="Sheet1", usecols="A, B", engine='openpyxl')
    cols = dataRead.columns.values

    n = dataRead[[cols[0]]].values.tolist()
    n2 = dataRead[[cols[1]]].values.tolist()

    labelDic = {}
    for e, e1 in zip(n, n2):
        # print(type(e1[0]))
        labelDic[e[0]] = e1[0]

    # print(type(labelDic[2]))
    dicLable = {}

    path = []
    label = []
    for f in file:
        path.append(folder + '/' + f)
        # print(folder + '/' + f)
        label.append(labelDic[int(f[:7])])
        # print(labelDic[int(f[:7])])
        dicLable[f] = labelDic[int(f[:7])]

    return path, label, dicLable

def generate_ds(train: tuple,
                val: tuple,
                train_im_height: int = None,
                train_im_width: int = None,
                val_im_height: int = None,
                val_im_width: int = None,
                batch_size: int = 8,
                cache_data: bool = False):
    """

    :param train: the tuple contains the address list and label list for the training set
    :param val: the tuple contains the address list and label list for the validation set
    :param train_im_height:
    :param train_im_width:
    :param val_im_height:
    :param val_im_width:
    :param batch_size:
    :param cache_data:
    :return:
    """
    print(train_im_height)
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    train_img_path, train_img_label, c = train
    val_img_path, val_img_label, d = val
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def process_train_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, train_im_height, train_im_width)
        # print("resize", image)
        image = tf.image.random_flip_left_right(image)
        image = (image / 255. - 0.5) / 0.5
        # print("final", image)
        return image, label

    def process_val_info(img_path, label):
        image = tf.io.read_file(img_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, val_im_height, val_im_width)
        image = (image / 255. - 0.5) / 0.5
        return image, label

    # Configure dataset for performance
    def configure_for_performance(ds,
                                  shuffle_size: int,
                                  shuffle: bool = False,
                                  cache: bool = False):
        if cache:
            ds = ds.cache()  # 读取数据后缓存至内存
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # 打乱数据顺序
        ds = ds.batch(batch_size)                      # 指定batch size
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # 在训练的同时提前准备下一个step的数据
        return ds

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_path),
                                                   tf.constant(train_img_label)))
    total_train = len(train_img_path)

    # Use Dataset.map to create a dataset of image, label pairs
    train_ds = train_ds.map(process_train_info, num_parallel_calls=AUTOTUNE)
    train_ds = configure_for_performance(train_ds, total_train, shuffle=True, cache=cache_data)

    val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(val_img_path),
                                                 tf.constant(val_img_label)))
    total_val = len(val_img_path)
    # Use Dataset.map to create a dataset of image, label pairs
    val_ds = val_ds.map(process_val_info, num_parallel_calls=AUTOTUNE)
    val_ds = configure_for_performance(val_ds, total_val, cache=False)

    return train_ds, val_ds


LEFTTRAIN = match(leftTrain)
LEFTVAL = match(leftVal)
LEFTTEST = match(leftTest)
#
RIGHTTRAIN = match(rightTrain)
RIGHTVAL = match(rightVal)
RIGHTTEST = match(rightTest)
# print(LEFTTEST)

COMTRAIN = match(combineTrain)
COMVAL = match(combineVal)
COMTEST = match(combineTest)


if __name__ == "__main__":
    # splitType(unsegLeft, "Test - Left Bulb", unsegLeftTest)
    print(type(COMTRAIN))
    print(generate_ds(COMTRAIN, COMVAL, 300, 300, 384, 384))
