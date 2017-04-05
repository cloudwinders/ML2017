import sys
import numpy as np
from math import log, floor
import time 

lamda = 0.0000001#0.00005
train_ratio = 0.8
col_idx = np.arange(31,38) #delete mariage status
col_idx = np.append(col_idx, np.array([15,16,17,18,19,20,21]))
norm_index = [0, 1, 3, 4, 5]

start_time = time.time()

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def load_data(delete_feature):
    X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
    X_train = np.delete(X_train, delete_feature, 1)
    X_train = np.delete(X_train, 0, 0)
    Y_train = np.genfromtxt(sys.argv[4], delimiter=',')
    
    X_test = np.genfromtxt(sys.argv[5], delimiter = ",")
    X_test = np.delete(X_test, delete_feature, 1)
    X_test = np.delete(X_test, 0, 0)

    return X_train, Y_train, X_test

def feature_normalize(X_train, X_test, index):
    # feature normalization with all X
    X_all = np.concatenate((X_train, X_test))
    mu = np.mean(X_all, axis=0)
    sigma = np.std(X_all, axis=0)
    
    # only apply normalization on continuos attribute
    mean_vec = np.zeros(X_all.shape[1])
    std_vec = np.ones(X_all.shape[1])
    mean_vec[index] = mu[index]
    std_vec[index] = sigma[index]

    X_all_normed = (X_all - mean_vec) / std_vec

    # split train, test again
    X_train_normed = X_all_normed[0:X_train.shape[0]]
    X_test_normed = X_all_normed[X_train.shape[0]:]

    return X_train_normed, X_test_normed

def split_data(X_train, Y_train, ratio):
    xSize = int(X_train.shape[0]*ratio)
    ySize = int(Y_train.shape[0]*ratio)

    #xA and yA is training set, xB and yB is validation set
    xA = X_train[:xSize]
    xB = X_train[xSize:]
    yA = Y_train[:ySize]
    yB = Y_train[ySize:]

    return xA,yA,xB,yB

def shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def train(X_train_normed, Y_train, X_valid, Y_valid, lamda):
    # parameter initiallize
    w = np.ones((X_train_normed.shape[1], ))
    w2 = np.ones((X_train_normed.shape[1], ))
    b = np.zeros((1,))

    prev_w_gra = np.zeros((X_train_normed.shape[1], ))
    prev_w2_gra = np.zeros((X_train_normed.shape[1], ))
    prev_b_gra= np.zeros((1, ))

    l_rate = 0.03
    epoch_num = 400
    batch_size = 50#128
    train_data_size = X_train_normed.shape[0]
    batch_num = int(floor(train_data_size / batch_size))
    display_num = 50
    # train with batch
    for epoch in range(epoch_num):
        # random shuffle
        X_train_normed, Y_train = shuffle(X_train_normed, Y_train)
        epoch_loss = 0.0
        for idx in range(batch_num):
            X = X_train_normed[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]
            
            z = b + np.dot(X, np.transpose(w)) + np.dot(X**2, np.transpose(w2))
            y = sigmoid(z)
            m = len(y)
            
            cross_entropy = -(1.0/m) * (np.dot(Y, np.log(y)) + np.dot((1 - Y), np.log(1 - y))) + (lamda/(2.0*m))*(w.T.dot(w) + w2.T.dot(w2))
            epoch_loss += cross_entropy
            
            w_grad = (1.0/m) * np.sum(-1 * X * (Y - y).reshape((batch_size,1)), axis=0) + (lamda/m)*(w.T.dot(w))
            w2_grad = (1.0/m) * np.sum(-1 * X**2 * (Y - y).reshape((batch_size,1)), axis=0) + (lamda/m)*(w2.T.dot(w2))
            b_grad = np.sum(-1 * (Y - y))

            prev_w_gra += w_grad**2
            prev_w2_gra += w2_grad**2
            prev_b_gra += b_grad**2

            ada_w = np.sqrt(prev_w_gra)
            ada_w2 = np.sqrt(prev_w2_gra)
            ada_b = np.sqrt(prev_b_gra)
            ada_w[ada_w == 0] = 1
            ada_w2[ada_w2 == 0] = 1
            ada_b = np.sqrt(prev_b_gra)
            ada_b[ada_b == 0] = 1

            w = w - l_rate * w_grad/ada_w
            w2 = w2 - l_rate * w2_grad/ada_w2
            b = b - l_rate * b_grad/ada_b

        if (epoch+1) % display_num == 0:
            z2 = b + np.dot(X_valid, np.transpose(w)) + np.dot(X_valid**2, np.transpose(w2))
            y2 = sigmoid(z2)
            y2 = np.around(y2)
            y = np.around(y)
            train_accuracy = np.sum(y == Y_train[idx*batch_size:(idx+1)*batch_size])
            valid_accuracy = np.sum(y2 == Y_valid)
            #print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / train_data_size)))
            print ('train accuracy in epoch%d: %f' % (epoch+1, (float(train_accuracy) / y.shape[0])))
            print ('valid accuracy in epoch%d: %f' % (epoch+1, (float(valid_accuracy) / y2.shape[0])))

    return w, w2, b

def predict(w, w2, b, X_test_normed):
    # output prediction to 'prediction.csv'
    z = b + np.dot(X_test_normed, np.transpose(w)) + np.dot(X_test_normed**2, np.transpose(w2))
    y = sigmoid(z)
    y_ = np.around(y)
    with open(sys.argv[6], 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))
    return

if __name__ == '__main__':
    X_train, Y_train, X_test = load_data(col_idx)
    X_train_normed, X_test_normed = feature_normalize(X_train, X_test, norm_index)
    #X_train_normed, Y_train = shuffle(X_train_normed, Y_train)
    X_train_normed, Y_train, X_valid, Y_valid = split_data(X_train_normed, Y_train, train_ratio)
    w, w2, b = train(X_train_normed, Y_train, X_valid, Y_valid, lamda)
    
    predict(w, w2, b, X_test_normed)

    print("--- %s seconds ---" % (time.time() - start_time))