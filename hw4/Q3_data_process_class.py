import numpy as np
from sklearn.neighbors import NearestNeighbors

class process:
    def elu(self,arr):
        return np.where(arr > 0, arr, np.exp(arr) - 1)

    def make_layer(self,in_size, out_size):
        w = np.random.normal(scale=0.5, size=(in_size, out_size))
        b = np.random.normal(scale=0.5, size=out_size)
        return (w, b)


    def forward(self,inpd, layers):
        out = inpd
        for layer in layers:
            w, b = layer
            out = self.elu(np.dot(out,w) + b)

        return out

    def gen_data(self,dim, layer_dims, N):
        layers = []
        data = np.random.normal(size=(N, dim))

        nd = dim
        for d in layer_dims:
            layers.append(self.make_layer(nd, d))
            nd = d

        w, b = self.make_layer(nd, nd)
        gen_data = self.forward(data, layers)
        gen_data = np.dot(gen_data,w) + b
        return gen_data

    def get_eigenvalues(self,data):
        SAMPLE = 50 # sample some points to estimate
        NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
        randidx = np.random.permutation(data.shape[0])[:SAMPLE]
        knbrs = NearestNeighbors(n_neighbors=NEIGHBOR, algorithm='ball_tree').fit(data)

        sing_vals = []
        for idx in randidx:
            dist, ind = knbrs.kneighbors(data[idx:idx+1])
            nbrs = data[ind[0,1:]]
            u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
            #normalize the value
            s = (s - np.amin(s)) / (np.amax(s) - np.amin(s))

            sing_vals.append(s)
            
        sing_vals = np.array(sing_vals).mean(axis=0)
        return sing_vals
