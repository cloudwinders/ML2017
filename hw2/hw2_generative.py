import sys
import numpy as np
import math

col_idx = np.arange(31,38) #delete mariage status
col_idx = np.append(col_idx, np.array([15,16,17,18,19,20,21]))

def load_data(delete_feature):
    X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
    X_train = np.delete(X_train, delete_feature, 1)
    X_train = np.delete(X_train, 0, 0)
    Y_train = np.genfromtxt(sys.argv[4], delimiter=',')
    
    X_test = np.genfromtxt(sys.argv[5], delimiter = ",")
    X_test = np.delete(X_test, delete_feature, 1)
    X_test = np.delete(X_test, 0, 0)

    return X_train, Y_train, X_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def predict(X_test, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    return y
def train(X_train, Y_train):
    # gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0
    dim = X_train.shape[1]

    mu1 = np.zeros((dim,))
    mu2 = np.zeros((dim,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    return (mu1, mu2, shared_sigma, cnt1, cnt2)

def sample_submission(Y,path):
    with open(path,'w') as fp:
        fp.write('id,label\n')
        for i in range(len(Y)):
            fp.write('{:d},{:d}\n'.format(i+1,int(Y[i])))
if __name__ == '__main__':
    (X_train,Y_train,X_test) = load_data(col_idx)
    #X_train, X_test = normalize(X_train, X_test)
    mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train)
    y = predict(X_train, mu1, mu2, shared_sigma, N1, N2)
    y_ = np.around(y)
    result = (Y_train == y_)
    print 'Train acc = %f' % (float(result.sum()) / result.shape[0])
    y = predict(X_test, mu1, mu2, shared_sigma, N1, N2)
    y_ = np.around(y)

    sample_submission(y_,sys.argv[6])