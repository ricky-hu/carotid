import cv2, os
import pandas as pd
import shutil
import tensorflow as tf

# This file works for mask data preprocessing
base_path = "./images_mask/all/"
new_path = "./images_mask/comTest2"

leftTrain = "./images_mask/leftTrain"
leftVal = "./images_mask/leftVal"
leftTest = "./images_mask/leftTest"

rightTrain = "./images_mask/rightTrain"
rightVal = "./images_mask/rightVal"
rightTest = "./images_mask/rightTest"

combineTrain = "./images_mask/combineTrain"
combineVal = "./images_mask/combineVal"
combineTest = "./images_mask/combineTest"

combineTrain2 = "./images_mask/comTrain2"
combineVal2 = "./images_mask/comVal2"
combineTest2 = "./images_mask/comTest2"

random1 = "./images_mask/random1"
random2 = "./images_mask/random2"
random3 = "./images_mask/random3"
random4 = "./images_mask/random4"

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

# move the segmented images into teh corresponding folder for the further training
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
    """
    The function reads the name of the image from the folder and match the image into the label in table CD
    :param folder: the folder images are read from, ex: folder should be leftTrain when i
    :return:
    """
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
    :param train_im_height: the height of train image
    :param train_im_width:the width of train image
    :param val_im_height:the height of validation image
    :param val_im_width:the width of validation image
    :param batch_size: the batch size
    :param cache_data: whether cache the data
    :return:
    """
    # print(train_im_height)
    assert train_im_height is not None
    assert train_im_width is not None
    if val_im_width is None:
        val_im_width = train_im_width
    if val_im_height is None:
        val_im_height = train_im_height

    # read the train and validation path and label
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
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_size)  # shuffle the sequence
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)         # prepare the data needed next when training on the current one
        return ds

    train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(train_img_path),
                                                   tf.constant(train_img_label)))
    total_train = len(train_img_path)

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

COMTRAIN2 = match(combineTrain2)
COMVAL2 = match(combineVal2)
COMTEST2 = match(combineTest2)

RAN1 = match(random1)
RAN2 = match(random2)
RAN3 = match(random3)
RAN4 = match(random4)
if __name__ == "__main__":
    # splitType(unsegLeft, "Test - Left Bulb", unsegLeftTest)
    # print(type(COMTRAIN))
    # print(generate_ds(COMTRAIN, COMVAL, 300, 300, 384, 384))
    splitType(base_path, "Test - all images", new_path)