# -*- coding: utf-8 -*-
import os
import sys
import word2vec
import nltk
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from adjustText import adjust_text

base_dir = os.path.dirname(os.path.realpath(__file__))

#set word2vec variables
sizeVectors = 50
downSamplingHFWords = '1e-5'
iterations = 30
min_appear = 500

data_ratio = 1.0

def combine_data(data_path):
    filenames = [data_path+"/Book 1 - The Philosopher's Stone_djvu.txt",\
    data_path+"/Book 2 - The Chamber of Secrets_djvu.txt",\
    data_path+"/Book 3 - The Prisoner of Azkaban_djvu.txt",\
    data_path+"/Book 4 - The Goblet of Fire_djvu.txt",\
    data_path+"/Book 5 - The Order of the Phoenix_djvu.txt",\
    data_path+"/Book 6 - The Half Blood Prince_djvu.txt",\
    data_path+"/Book 7 - The Deathly Hallows_djvu.txt"]

    combine_data_path = base_dir+"/all.txt"
    with open(combine_data_path, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

    return combine_data_path

if __name__ == "__main__":
    data_combine_path = combine_data(sys.argv[1])
    word2vec.word2vec(data_combine_path, os.path.join(sys.argv[1], "Q1_data.bin"), size = sizeVectors, \
                sample = downSamplingHFWords, iter_= iterations, min_count = min_appear, verbose = True)
    print("")

    #load model
    model = word2vec.load(os.path.join(sys.argv[1], "Q1_data.bin"))

    print("original size: " + str(model.vectors.shape))

    vocabs = []
    vecs = []
    for word in model.vocab:
        vocabs.append(word)
        vecs.append(model[word])
    vecs_length = len(vecs)
    vecs = np.array(vecs)[:int((data_ratio*vecs_length))]
    vocabs = vocabs[:int((data_ratio*vecs_length))]    

    #reduce dimension to 2
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vecs)

    print("modified size: " + str(model.vectors.shape))

    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", ".", ":", ";", ",", "?", "!", u"’"]
    
    plt.figure()
    texts = []
    nltk.download('averaged_perceptron_tagger', 'maxent_treebank_pos_tagger', 'punkt')
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags and all(c not in label for c in puncts)):
            x, y = Y[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=1.5))

    plt.savefig('word2vec_visualisation.png', dpi=600)
