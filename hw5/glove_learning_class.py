import os
import re
import nltk
from nltk.corpus import stopwords
import cPickle as pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
import keras.backend as K

class data_process:
    def __init__(self, thresh):
        self.thresh = thresh

    def get_path(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        exp_dir = os.path.join(base_dir,'exp')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return base_dir, exp_dir

    def read_data(self, path, training):
        print ('Reading data from ',path)
        with open(path,'r') as f:
            tags = []
            articles = []
            tags_list = []
            
            f.readline()
            for line in f:
                if training :
                    start = line.find('\"')
                    end = line.find('\"',start+1)
                    tag = line[start+1:end].split(' ')
                    article = line[end+2:]
                    
                    for t in tag :
                        if t not in tags_list:
                            tags_list.append(t)
                    tags.append(tag)
                else:
                    start = line.find(',')
                    article = line[start+1:]
                articles.append(article)
                
            if training :
                assert len(tags_list) == 38,(len(tags_list))
                assert len(tags) == len(articles)
        return (tags,articles,tags_list)

    def to_multi_categorical(self, tags, tags_list): 
        tags_num = len(tags)
        tags_class = len(tags_list)
        Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
        for i in range(tags_num):
            for tag in tags[i] :
                Y_data[i][tags_list.index(tag)]=1
            assert np.sum(Y_data) > 0
        return Y_data

    def split_data(self, X, Y, split_ratio):
        indices = np.arange(X.shape[0])  
        np.random.shuffle(indices) 
        
        X_data = X[indices]
        Y_data = Y[indices]
        
        num_validation_sample = int(split_ratio * X_data.shape[0] )
        
        X_train = X_data[num_validation_sample:]
        Y_train = Y_data[num_validation_sample:]

        X_val = X_data[:num_validation_sample]
        Y_val = Y_data[:num_validation_sample]
        return (X_train,Y_train),(X_val,Y_val)

    def get_embedding_dict(self, path):
        embedding_dict = {}
        with open(path, 'r') as f:
            for line in f:
                values = line.split(" ")
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embedding_dict[word] = coefs

        embedding_dim = re.findall(r'\d+', path)
        
        return embedding_dict, int(embedding_dim[1])

    def get_embedding_matrix(self, word_index, embedding_dict, num_words, embedding_dim):
        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, i in word_index.items():
            if i < num_words:
                embedding_vector = embedding_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def filter_Tokenizer_word(self, text):
        process_tokenizer = Tokenizer()
        process_tokenizer.fit_on_texts(text)
        process_word_index = process_tokenizer.word_index
        nltk.download('stopwords')
        stop = stopwords.words('english')
        filter_all_corpus = [word for word in process_word_index if not word in stop]
        
        use_tags = set(['JJ', 'NN', 'ADJ', 'VBD', 'VBG', 'VBP', 'VB', 'NNP'])
        nltk.download(['averaged_perceptron_tagger', 'maxent_treebank_pos_tagger', 'punkt'])

        filter_text = []
        for (i, label) in enumerate(filter_all_corpus):
            pos = nltk.pos_tag([label])
            if(pos[0][1] in use_tags):
                filter_text.append(label)
        filter_all_corpus = filter_text
        
        return filter_all_corpus

    def f1score(self, y_true, y_pred):
        #pred_norm = K.l2_normalize(y_pred, axis=1)
        #y_pred = y_pred / pred_norm[:,None]
        y_pred = K.cast(K.greater(y_pred,self.thresh),dtype='float32')
        tp = K.sum(y_true * y_pred,axis=-1)
    
        precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
        recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
        f1 = K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

        return f1
        
    def store_pickle(self, pack_item, path):
        with open(path, 'wb') as handle:
            pickle.dump(pack_item, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, path):
        with open(path, 'rb') as handle:
            unpack_item = pickle.load(handle)
        return unpack_item

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
            for acc in logs.tr_f1score:
                f.write('{}\n'.format(acc))
        with open(os.path.join(store_path,'valid_loss'),'a') as f:
            for loss in logs.val_loss:
                f.write('{}\n'.format(loss))
        with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
            for acc in logs.val_f1score:
                f.write('{}\n'.format(acc))

class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_loss=[]
        self.val_loss=[]
        self.tr_f1score=[]
        self.val_f1score=[]

    def on_epoch_end(self,epoch,logs={}):
        self.tr_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.tr_f1score.append(logs.get('f1score'))
        self.val_f1score.append(logs.get('val_f1score'))