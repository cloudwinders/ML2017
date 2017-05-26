import os
import sys
import collections
import numpy as np
from numpy.linalg import norm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import merge, Input
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Nadam, RMSprop
from keras.utils.vis_utils import plot_model
from keras import metrics
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.models import load_model
import keras.backend as K
from glove_learning_class import data_process

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import Normalizer,normalize
from sklearn.decomposition import PCA, TruncatedSVD

maxlen_pad_sequences = None
epoch = 100
batch_size = 64
validation_split = 0.2
gap_probability = 0.45


if __name__ == "__main__":
    data_process = data_process(gap_probability)
    base_dir, exp_dir = data_process.get_path()
    store_path, history_data = data_process.history_data(exp_dir, epoch)

    (Y_data,X_data,tag_list) = data_process.read_data(sys.argv[1],True)
    (_, X_test,_) = data_process.read_data(sys.argv[2],False)

    train_tag = data_process.to_multi_categorical(Y_data,tag_list)

    all_corpus = X_data + X_test
    print('Find %d articles.' %len(all_corpus))

    '''
    #For Question 3
    tag_sum_analyze = np.sum(train_tag, axis=0)
    print ("\n\n")
    for i in range(len(tag_list)):
        print(str(tag_list[i])+":"+str(int(tag_sum_analyze[i])))
    print("\n\n")
    '''
    '''
    #For Question 5: bag of words
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=2, max_df  = 0.7)
    all_corpus = vectorizer.fit_transform(all_corpus)
    #####normalize
    normalizer = Normalizer(norm='l2',copy=True)
    all_corpus = normalize(all_corpus, norm='l2')
    #svd = TruncatedSVD(n_components = 2000)
    #all_corpus = svd.fit_transform(all_corpus)
    all_corpus = all_corpus.toarray()

    X_data = all_corpus[:len(X_data)]
    X_test = all_corpus[len(X_data):]

    (X_train,Y_train),(X_val,Y_val) = data_process.split_data(X_data,train_tag,validation_split)

    model = Sequential()
    model.add(Dense(1024,input_shape=X_train.shape[1:],activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()
    '''
    
    all_corpus = data_process.filter_Tokenizer_word(all_corpus)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)

    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1

    tokenizer_file = os.path.join(store_path, "tokenizer.pickle")
    data_process.store_pickle(tokenizer, tokenizer_file)

    label_file = os.path.join(store_path, "label.pickle")
    data_process.store_pickle(tag_list, label_file)

    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    train_sequences = pad_sequences(train_sequences, maxlen=maxlen_pad_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences, maxlen=max_article_length)

    (X_train,Y_train),(X_val,Y_val) = data_process.split_data(train_sequences,train_tag,validation_split)

    dictionary_pretrained, embedding_dim = data_process.get_embedding_dict(sys.argv[3])
    embedding_layer_input = data_process.get_embedding_matrix(word_index, dictionary_pretrained, vocab_size, embedding_dim)

    '''
    ##JUST FOR FUTURE REFERENCE
    input_size = Input(shape=(max_article_length,), dtype='int32')
    embedding_layer = Embedding((vocab_size), embedding_dim, weights=[embedding_layer_input], input_length=max_article_length, trainable=False)(input_size)
    #print embedding_layer.shape
    conv = Conv1D(1024, 3, padding='same', activation='relu')(embedding_layer)
    conv2 = Conv1D(1024, 1, padding='same', activation='relu')(embedding_layer)
    rnn_layer1 = GRU(512)(conv)
    rnn_layer2 = GRU(512)(conv)
    rnn_layer3 = GRU(512)(conv2)
    rnn_layer4 = GRU(512)(conv2)
    combine_layer = merge([rnn_layer1,rnn_layer2,rnn_layer3,rnn_layer4], mode='concat')
    #combine_layer2 = keras.layers.concatenate([rnn_layer1,rnn_layer2,rnn_layer3,rnn_layer4])
    model = Dense(2048, activation='relu')(combine_layer)
    model = Dropout(0.2)(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(64, activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(len(tag_list), activation='sigmoid')(model)
    model = Model(input_size, model)
    ##JUST FOR FUTURE REFERENCE
    '''
    
    #.......................Model
    model = Sequential()
    model.add(Embedding((vocab_size), embedding_dim, weights=[embedding_layer_input], input_length=max_article_length, trainable=False))
    #model.add(GRU(512, return_sequences=True))#, recurrent_dropout=0.2))
    model.add(GRU(256, dropout=0.4))
    model.add(Dense(1536, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(tag_list), activation='softmax'))
    
    model.summary()

    nadam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.002)
    rmsprop = RMSprop(lr=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=[data_process.f1score])
    earlyStopping = EarlyStopping(monitor='val_f1score', patience=20, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=os.path.join(store_path,'best_model.h5'), verbose=1, save_best_only=True, monitor='val_f1score', mode='max')
    csvLogger = CSVLogger(os.path.join(store_path,'training_bag_of_words.log'))
    
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epoch, batch_size=batch_size, callbacks=[earlyStopping, checkpoint, history_data, csvLogger])

    data_process.dump_history(store_path,history_data)
    model.save(os.path.join(store_path,'model.h5'))
    plot_model(model,to_file=os.path.join(store_path,'model.png'))
    #print ("Sentence text_sequences length: %d" %max_article_length)

    #.......................predict test X

    model = load_model(os.path.join(store_path, 'best_model.h5'), custom_objects={'f1score': data_process.f1score})
    test_Y = model.predict(test_sequences) #X_test

    #normalize probability
    #linfnorm = norm(test_Y, axis=1, ord=np.inf)
    #test_Y = test_Y.astype(np.float) / linfnorm[:, None]

    test_Y[test_Y>=gap_probability] = 1
    test_Y[test_Y<gap_probability] = 0

    output_file = []
    output_file.append('"id","tags"')

    for i,labels in enumerate(test_Y):
        labels = [tag_list[x] for x,value in enumerate(labels) if value==1 ]
        temp = '"'+str(i)+'"' + ',' + '"' + str(" ".join(labels)) + '"'
        output_file.append(temp)

    with open(sys.argv[4],'w') as f:
        for data in output_file:
            f.write('{}\n'.format(data))