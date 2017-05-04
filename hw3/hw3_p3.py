import os
import sys
import numpy as np
import itertools
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
import matplotlib.pyplot as plt

figure_title = "Confusion Matrix.png"

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_val_data(train_path):
    #load data
    with open(train_path) as trainFile:
      trainList = trainFile.read().splitlines()
      train_arr = np.array([line.split(",") for line in trainList])
      x_arr = train_arr[1:,1]
      y_arr = train_arr[1:,0]
      x_arr = np.array([str(line).split() for line in x_arr])
      y_arr = np.array([str(line).split() for line in y_arr])

      x_train_data = x_arr.reshape(x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)
      y_train_data = y_arr.astype(np.int)#(28709,1)

    #rescale
    x_train_data /= 255

    # convert class vectors to binary class matrices (one hot vectors)
    #original, idx = np.unique(y_train_data, return_inverse = True)
    #y_train_data = np_utils.to_categorical(y_train_data, 7)

    ratio = 0.9
    xSize = int(x_train_data.shape[0]*ratio)
    ySize = int(y_train_data.shape[0]*ratio)
    x_train_data = x_train_data[xSize:]
    y_train_data = y_train_data[ySize:]

    return x_train_data, y_train_data

def main(train_path, model_path):
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    dev_feats, te_labels = read_val_data(train_path)
    predictions = emotion_classifier.predict_classes(dev_feats)
    conf_mat = confusion_matrix(te_labels,predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
    plt.savefig(figure_title)
    plt.show()

if __name__ == '__main__':
    train_path = sys.argv[1]
    model_path = sys.argv[2]
    main(train_path, model_path)