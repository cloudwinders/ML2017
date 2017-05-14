import os
import numpy as np
import csv
import sys
import glob
from scipy import misc
from sklearn.svm import LinearSVR as SVR
from hand_process_class import process

base_dir = os.path.dirname(os.path.realpath(__file__))

test = []

for image_path in glob.glob(sys.argv[1]+"/*.png"):
    image = misc.imread(image_path).flatten()
    test.append(image)

test = np.asarray(test)

size = []
temp = test.shape[1]/8
for i in range (8+1):
    size.append(temp*(i))

separate_test = []
for i in range (8):
    separate_test.append(test[:, size[i]:size[i+1]])

separate_test = np.asarray(separate_test)

# Train a linear SVR

npzfile = np.load('hand_train_data.npz')
X = npzfile['X']
y = npzfile['y']

svr = SVR(C=1)
svr.fit(X, y)

# predict 
test_X = []
process_data = process()

for i in range(8):
    data = separate_test[i]
    w = process_data.get_eigenvalues(data)
    test_X.append(w)

test_X = np.array(test_X)

pred_y = svr.predict(test_X)

ans = np.log(pred_y)

pred_dim = 0

for i in range(8):
    pred_dim += ans[i]


file = open('hand_predict.csv', 'w') 
f_write = csv.writer(file,delimiter =',')
f_write.writerow(['Setid','LogDim'])
f_write.writerow([str(0),str(pred_dim)])
file.close()

