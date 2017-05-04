import numpy as np
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import Callback
from keras import regularizers
import os
from keras.utils.vis_utils import plot_model
import tensorflow as tf

np.set_printoptions(suppress=True)

def dump_history(store_path,logs):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))

validation = 0.2 #0.2

first_layer = 64
second_layer = 128
third_layer = 250
fourth_layer = 500

first_dropout = 0.2#0.2
second_dropout = 0.3#0.3
third_dropout = 0.3#0.3
fourth_dropout = 0.6#0.6

batch_size = 64
nb_epoch = 1800

#load data
with open(sys.argv[1]) as trainFile:
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
original, idx = np.unique(y_train_data, return_inverse = True)
y_train_data = np_utils.to_categorical(y_train_data, 7)

ratio = 0.9
xSize = int(x_train_data.shape[0]*ratio)
ySize = int(y_train_data.shape[0]*ratio)
x_train_data = x_train_data[:xSize]
y_train_data = y_train_data[:ySize]

model = Sequential()
model.add(Conv2D(first_layer,(3,3) ,padding="valid", input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(first_dropout))

model.add(Conv2D(second_layer,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(second_dropout))

model.add(Conv2D(third_layer,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(third_dropout))

model.add(Conv2D(fourth_layer,(3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(fourth_dropout))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu', kernel_initializer='normal'))

model.add(Dense(7, activation='softmax', kernel_initializer='normal'))

model.summary()

nadam = Nadam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=nadam, metrics=['accuracy'])

#...............................................................................
base_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(base_dir,'exp')
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

dir_cnt = 0
log_path = "epoch_{}".format(str(nb_epoch))
log_path += '_'
store_path = os.path.join(exp_dir,log_path+str(dir_cnt))
while dir_cnt < 30:
    if not os.path.isdir(store_path):
        os.mkdir(store_path)
        break
    else:
        dir_cnt += 1
        store_path = os.path.join(exp_dir,log_path+str(dir_cnt))

history_data = History()
#...............................................................................

history = model.fit(x_train_data, y_train_data, batch_size=batch_size, epochs=nb_epoch, verbose=1, shuffle=True, validation_split=validation, callbacks=[history_data])

dump_history(store_path,history_data)
model.save(os.path.join(store_path,'model.h5'))
#plot_model(model,to_file=os.path.join(store_path,'model.png'))

'''
with open(sys.argv[2]) as testFile:
  testList = testFile.read().splitlines()
  test_arr = np.array([line.split(",") for line in testList])
  test_x_arr = test_arr[1:,1]
  test_x_arr = np.array([str(line).split() for line in test_x_arr])

  x_test_data = test_x_arr.reshape(test_x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)

#rescale
x_test_data /= 255

test_y = model.predict_classes(x_test_data, batch_size=batch_size)

test_output = np.zeros((len(test_y)+1, 2), dtype='|S5')
test_output[0,0] = "id"
test_output[0,1] = "label"

for i in range (test_output.shape[0]-1):
    test_output[i+1,0] = str(i)
    test_output[i+1,1] = str(test_y[i])

np.savetxt(sys.argv[3], test_output, delimiter=",", fmt = "%s")
'''