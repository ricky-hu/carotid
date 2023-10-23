import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
import seaborn as sns

from model import efficientnetv2_s as create_model
import path
import maskPre as mp


import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm


# the pred is the dictionary contains the predictions, the fact column contains the true result
def confusionMatrix(pred: dict, fact: dict):

    tp, fp, fn, tn = 0, 0, 0, 0
    cm = [[0, 0],
          [0, 0]]

    a = []
    b = []
    for key in pred:
        a.append(pred[key])
        result = fact[key]
        b.append(result)
        print(key, pred[key], fact[key])
        if pred[key] == 1 and result == 1:
            tp += 1
        elif pred[key] == 1 and result == 0:
            fp += 1
        elif pred[key] == 0 and result == 0:
            tn += 1
        elif pred[key] == 0 and result == 1:
            fn += 1

    cm[0][0], cm[0][1], cm[1][0], cm[1][1] = tn, fp, fn, tp

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)


    f1 = 2 * ((ppv * tpr) / (ppv + tpr))
    acc = (tp + tn) / (tp + tn + fp + fn)

    print("Recall/Sensitivity/True positive rate: {:.3f}".format(tpr))
    print("Specificity/selectivity/True negative rate: {:.3f}".format(tnr))
    print("Precision/Positive predictive value: {:.3f}".format(ppv))
    print("Negative predictive value: {:.3f}".format(npv))
    print("F1 score: {:.3f}".format(f1))
    print("Accuracy: {:.3f}".format(acc))

    print(cm)

    ax = sns.heatmap(cm, annot=True, cmap='Blues')

    ax.set_title('CM\n')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    roc(a, b)


def predict(img_path: str):
    correct = mp.RAN4[2]
    # test is the folder contains the test images
    num_classes = 2

    im_height = im_width = 384

    model = create_model(num_classes=num_classes)

    weights_path = 'save_seg_right45/efficientnetv2.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    prediction = {}

    for img in os.listdir(img_path):
        name = img
        assert os.path.exists(img_path + "\\{}".format(img)), "file: '{}' dose not exist.".format(img)
        img = Image.open(img_path + "\\{}".format(img))
        # resize image
        img = img.resize((im_width, im_height))
        plt.imshow(img)

        # read image
        img = np.array(img).astype(np.float32)

        # preprocess
        img = (img / 255. - 0.5) / 0.5

        # Add the image to a batch where it's the only member.
        img = (np.expand_dims(img, 0))

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        result = np.squeeze(model.predict(img))
        result = tf.keras.layers.Softmax()(result)
        predict_class = np.argmax(result)

        # print(name, correct[name])
        # print(name)
        prediction[name] = predict_class

        print(name, correct[name])
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                     result[predict_class])
        plt.title(print_res)
        # print("class: {}   prob: {:.3}".format(class_indict[str(i)],
        #                                           result[i].numpy()))

        # print(predict_class)
        for i in range(len(result)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      result[i].numpy()))

        print(" ")

    confusionMatrix(prediction, correct)

    return prediction


def roc(l1, l2):
    fpr, tpr, threshold = metrics.roc_curve(l2, l1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    predict('./images_mask/random4')
    # roc()
