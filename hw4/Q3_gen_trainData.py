import os
import time
import numpy as np
from Q3_data_process_class import process

st_time = time.time()

base_dir = os.path.dirname(os.path.realpath(__file__))

# generate some data for training
process_data = process()

X = []
y = []
for i in range(60):
    dim = i + 1  #dim >= 1
    for N in [10000, 20000, 50000, 80000, 100000]:
        layer_dims = [np.random.randint(60, 80), 100]
        data = process_data.gen_data(dim, layer_dims, N).astype('float32')
        eigenvlues = process_data.get_eigenvalues(data)
        X.append(eigenvlues)
        y.append(dim)

X = np.array(X)
y = np.array(y)

np.savez(os.path.join(base_dir, 'train_data.npz'), X=X, y=y)

ed_time = time.time()
duration = ed_time - st_time
print ("Duration: " + time.strftime('%H:%M:%S', time.gmtime(duration)))