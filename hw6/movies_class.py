import os
import numpy as np
import pandas as pd
from keras.callbacks import Callback
import keras.backend as K

class data_process:
    def get_path(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        exp_dir = os.path.join(base_dir,'exp')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return base_dir, exp_dir

    def read_data(self, path, option):
        data = []
        genres = []
        with open(path) as file:
            file = file.read().splitlines()
            if ((option=='train') or (option=='test')):
                data = np.array([line.split(",") for line in file])
            else:
                data = np.array([line.split("::") for line in file])
            if option=='movies':
                genres = np.array([line.split("|") for line in data[:,-1]])
                data = data[:,:2]
            
        return data, genres

    def split_data(self, X1, X2, Y, split_ratio):
        indices = np.arange(len(X1))
        np.random.shuffle(indices) 
            
        X1_data = X1[indices]
        X2_data = X2[indices]
        Y_data = Y[indices]
            
        num_validation_sample = int(split_ratio * len(X1))
            
        X1_train = X1_data[num_validation_sample:]
        X2_train = X2_data[num_validation_sample:]
        Y_train = Y_data[num_validation_sample:]

        X1_val = X1_data[:num_validation_sample]
        X2_val = X2_data[:num_validation_sample]
        Y_val = Y_data[:num_validation_sample]
        return X1_train, X2_train, Y_train, X1_val, X2_val, Y_val

    def root_mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def history_data(self, exp_dir, epoch):
        dir_cnt = 0
        log_path = "epoch_{}".format(str(epoch))
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

        return store_path, history_data

    def dump_history(self, store_path,logs):
        with open(os.path.join(store_path,'train_loss'),'a') as f:
            for loss in logs.tr_loss:
                f.write('{}\n'.format(loss))
        with open(os.path.join(store_path,'train_accuracy'),'a') as f:
            for acc in logs.tr_accuracy:
                f.write('{}\n'.format(acc))
        with open(os.path.join(store_path,'valid_loss'),'a') as f:
            for loss in logs.val_loss:
                f.write('{}\n'.format(loss))
        with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
            for acc in logs.val_accuracy:
                f.write('{}\n'.format(acc))

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_loss=[]
        self.val_loss=[]
        self.tr_accuracy=[]
        self.val_accuracy=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.tr_accuracy.append(logs.get('root_mean_squared_error'))
        self.val_accuracy.append(logs.get('val_root_mean_squared_error'))