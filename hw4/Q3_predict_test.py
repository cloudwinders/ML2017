import os
import sys
import time
import numpy as np
import csv
from sklearn.svm import LinearSVR as SVR
from Q3_data_process_class import process

base_dir = os.path.dirname(os.path.realpath(__file__))

st_time = time.time()

npzfile = np.load(os.path.join(base_dir, 'train_data.npz'))
X = npzfile['X']
y = npzfile['y']

# Train a linear SVR
svr = SVR(C=1.5)
svr.fit(X, y)

# predict 
test_data = np.load(sys.argv[1])
test_data_size = len(test_data.files)

test_X = []
process_data = process()

for i in range(test_data_size):
    data = test_data[test_data.files[i]]
    data = np.asarray(data)
    vs = process_data.get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)

pred_y = svr.predict(test_X)
ans = np.log(pred_y)

file = open(sys.argv[2], 'w') 
f_write = csv.writer(file,delimiter =',')
f_write.writerow(['Setid','LogDim'])
for i in range(test_data_size):
    f_write.writerow([str(i), str(ans[i])])
file.close()

ed_time = time.time()
duration = ed_time - st_time
print ("Duration: " + time.strftime('%H:%M:%S', time.gmtime(duration)))