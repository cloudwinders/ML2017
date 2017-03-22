import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab as plt
import math
import time

#"/home/MachineLearning/HW1/data/train.csv"
iterations = 100000
alpha = 0.2
ratio = 0.8
sample_hours = 8
row_idx = np.array([9])

start_time = time.time()

def feature_normalize(X):
    #Normalize data
    mean_r = []
    std_r = []

    X_norm = X
    n_c = X.shape[1]

    for i in range(n_c):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])
        mean_r.append(m)
        std_r.append(s)

        X_norm[:, i] = (X_norm[:, i] - m) / s

    mean_r = np.asarray(mean_r)
    std_r = np.asarray(std_r)

    return X_norm, mean_r, std_r

def compute_cost(X, y, theta):
    #Number of training samples
    m = y.size

    predictions = X.dot(theta)
    sqErrors = (predictions - y)
    J = math.sqrt((1.0 / m) * sqErrors.T.dot(sqErrors))

    return J

def gradient_descent(X, y, theta, alpha, num_iters, ratio):
    xSize = int(X.shape[0]*ratio)
    ySize = int(y.shape[0]*ratio)

    #xA and yA is training set, xB and yB is validation set
    xA = X[:xSize]
    xB = X[xSize:]
    yA = y[:ySize]
    yB = y[ySize:]

    m = yA.size
    J_history_train = np.zeros(shape=(num_iters, 1))
    J_history_val = np.zeros(shape=(num_iters, 1))

    b = theta[0]
    w = theta[1:]

    b_lr = 1.0
    w_lr = np.ones(shape=(theta.size-1, 1))
    
    for i in range (1,num_iters+1):
        predictions = xA.dot(theta)
        theta_size = theta.size

        b_grad = -2*(yA - predictions)
        b_grad_sum = b_grad.sum()
        w_grad = (-2*(yA - predictions)*xA[:,1:]).T
        w_grad_sum = np.sum(w_grad,1)
        w_grad_sum = w_grad_sum.reshape((w_grad_sum.shape[0],1))

        #Adagrad
        b_lr += b_grad_sum**2
        w_lr += w_grad_sum**2

        b -= alpha/np.sqrt(b_lr) * b_grad_sum
        w -= alpha/np.sqrt(w_lr) * w_grad_sum

        theta[0] = b
        theta[1:] = w

        J_history_train[i-1, 0] = compute_cost(xA, yA, theta)
        J_history_val[i-1, 0] = compute_cost(xB, yB, theta)

        if i%100 == 0 or i == 1:
            print("Iteration time: " + str(i))
            print("train_error: \t\t" + str(J_history_train[i-1,0]))
            print("validation_error: \t" + str(J_history_val[i-1,0]))

    return theta, J_history_train, J_history_val

#........................................Function End........................................

train_data = np.genfromtxt(sys.argv[1], dtype = "S", skip_header=True, delimiter = ",")
test_data = np.genfromtxt(sys.argv[2], dtype = "S" ,delimiter = ",")

train_data[train_data == "NR"] = 0
test_data[test_data == "NR"] = 0
train_data = train_data[:, 3:]
train_data = train_data.astype(np.float)
test_data = test_data[:, 2:]
test_data = test_data.astype(np.float)

where_is_pm25 = int(np.where(row_idx == 9)[0])

train_process = train_data[:18,:]

for i in range (1,12*20):
    train_process = np.append(train_process, train_data[18*i:18+18*i,:], 1)

#.........................................Fill in the missing data for train.csv.........................................

where_is_negative_pm25 = np.where(train_process == -1)
negative_pm25_rows = where_is_negative_pm25[0]
negative_pm25_cols = where_is_negative_pm25[1]
for i in range (len(negative_pm25_rows)):
    num = 1

    row = negative_pm25_rows[i]
    col = negative_pm25_cols[i]
    while (train_process[row, col + num] == -1):
        num += 1

    avg = 0.0
    if (num%2) == 0:
        avg = (train_process[row, col - 1] - train_process[row, col + num])/(num+1.)
    else:
        avg = (train_process[row, col - 1] + train_process[row, col + num])/(num+2.)
    for k in range(num):
        train_process[row, col + k] = avg*k + train_process[row, col - 1]

#.........................................Pick the selected features in train.csv.........................................

train_process = train_process[row_idx[:, None], :]
train_process = train_process.reshape(len(row_idx), train_process.shape[2])

train_x = np.ones(shape=((24*20*12 - sample_hours*12), (sample_hours*row_idx.size+1)))
train_y = np.zeros(shape=((24*20*12 - sample_hours*12), 1))

for i in range (12):
    for j in range (24*20-sample_hours):
        col_sample = np.arange(0+j+(i*24*20),sample_hours+j+(i*24*20))
        train_sample = train_process[:, col_sample]
        train_x[j+i*(24*20-sample_hours),1:(sample_hours*row_idx.size+1)] = train_sample.flatten()
        train_y[j+i*(24*20-sample_hours),0] = train_process[where_is_pm25, sample_hours+j+(i*24*20)]

train_combine = np.hstack([train_x, train_y])
np.random.shuffle(train_combine)

train_x = train_combine[:,:-1]
train_y = train_combine[:,-1]
train_y = train_y.reshape((train_combine.shape[0],1))

#.........................................Fill the missing data in test_X.csv.........................................

test_negative_pm25 = np.where(test_data == -1)
test_negative_row = test_negative_pm25[0]
test_negative_col = test_negative_pm25[1]

for i in range (len(test_negative_row)):
    num = 1

    row = test_negative_row[i]
    col = test_negative_col[i]
    if col == 0:
        while (test_data[row, col + num] == -1):
            num += 1
        diff = test_data[row, col + num + 1] - test_data[row, col + num]
        for j in range (num):
            test_data[row, col + (num-j-1)] = test_data[row, col + (num-j)] - diff
    elif col == test_data.shape[1]:
        while (test_data[row, col - num] == -1):
            num += 1
        diff = test_data[row, col - num - 1] - test_data[row, col - num]
        for j in range (num):
            test_data[row, col - (num-j-1)] = test_data[row, col - (num-j)] - diff
    else:
        while(test_data[row, col + num] == -1):
            num += 1

        avg = 0.0
        if (num%2) == 0:
            avg = (test_data[row, col - 1] - test_data[row, col + num])/(num+1.)
        else:
            avg = (test_data[row, col - 1] + test_data[row, col + num])/(num+2.)
        for k in range(num):
            test_data[row, col + k] = avg*k + test_data[row, col - 1]


#.........................................Pick the selected feature in test_X.csv.........................................

col_test = np.arange(sample_hours) + (9-sample_hours)
test_x_part = test_data[row_idx[:, None], col_test]
test_x = test_x_part.flatten()

for i in range (1, 12*20):
    test_x_part = test_data[row_idx[:, None]+(18*i), col_test]
    test_x = np.vstack([test_x, test_x_part.flatten()])

#scale features and set them to zero mean
train_input = train_x[:,1:]
x_norm, mean_r, std_r = feature_normalize(train_input)
train_x[:,1:] = x_norm

mean_r = np.asarray(mean_r)
mean_r = mean_r.reshape((1, mean_r.shape[0]))
std_r = np.asarray(std_r)
std_r = std_r.reshape((1, std_r.shape[0]))

theta = np.genfromtxt("model_best.csv", dtype = "S" ,delimiter = ",")
theta = theta.astype(np.float)

#normalize data
test_input_norm = (test_x - mean_r) / std_r
test_input = np.ones(shape=(test_x.shape[0], (test_x.shape[1]+1)))
test_input[:, 1:(test_x.shape[1]+1)] = test_input_norm
test_y = test_input.dot(theta)
test_y = test_y.reshape(len(test_y),1)


test_output = np.zeros((test_y.shape[0]+1, test_y.shape[1]+1), dtype='|S10')
test_output[0,0] = "id"
test_output[0,1] = "value"

for i in range (test_output.shape[0]-1):
    test_output[i+1,0] = "id_" + str(i)
    test_output[i+1,1] = str(test_y[i,0])

print("Iterations: " + str(iterations))
print("alpha: " + str(alpha))
print("Ratio: " + str(ratio))
print("Sample hours: " + str(sample_hours))
print row_idx

print("--- %s seconds ---" % (time.time() - start_time))

np.savetxt(sys.argv[3], test_output, delimiter=",", fmt = "%s")
