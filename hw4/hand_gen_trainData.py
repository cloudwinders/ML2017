import numpy as np
from hand_process_class import process


# generate some data for training
process_data = process()
X = []
y = []
for i in range(10):
    dim = i + 1
    layer_dims = [np.random.randint(10, 1000),np.random.randint(1000, 10000), 30720]
    data = process_data.gen_data(dim, layer_dims, 481).astype('float32')
    eigenvlues = process_data.get_eigenvalues(data)
    X.append(eigenvlues)
    y.append(dim)

X = np.array(X)
y = np.array(y)

np.savez('hand_trainData.npz', X=X, y=y)
