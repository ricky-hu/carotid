import os
import sys
import math
import datetime

import tensorflow as tf
from tqdm import tqdm

from model import efficientnetv2_s as create_model
from maskPre import generate_ds
import maskPre as mp


def main(epoch: str):
    cwd = os.getcwd()
    train = mp.COMTRAIN2
    val = mp.COMVAL2

    if not os.path.exists("save_seg_com2_60"):
        os.makedirs("save_seg_com2_60")

    batch_size = 4
    epochs = 1
    num_classes = 2
    freeze_layers = True
    initial_lr = 0.01

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(train,
                                   val,
                                   train_im_height=300,
                                   train_im_width=300,
                                   val_im_height=384,
                                   val_im_width=384,
                                   batch_size=batch_size)

    # create model
    model = create_model(num_classes=num_classes)
    model.build((1, 300, 384, 3))

    # load weights
    pre_weights_path = cwd + '/weights/efficientnetv2-s.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # freeze bottom layers
    if freeze_layers:
        unfreeze_layers = "head"
        for layer in model.layers:
            if unfreeze_layers not in layer.name:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            loss = loss_object(train_labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds, file=sys.stdout)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = cwd + "/save_seg_com2_60/efficientnetv2.ckpt"
            model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main('60')
    # print(os.getcwd())
