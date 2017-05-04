import os
import sys
from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.realpath(__file__))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
orig_see_dir = os.path.join(img_dir, 'orig')
if not os.path.exists(orig_see_dir):
    os.makedirs(orig_see_dir)

def read_data(data_path):
    with open(data_path) as trainFile:
            trainList = trainFile.read().splitlines()
            train_arr = np.array([line.split(",") for line in trainList])
            x_arr = train_arr[1:,1]
            y_arr = train_arr[1:,0]
            x_arr = np.array([str(line).split() for line in x_arr])
            y_arr = np.array([str(line).split() for line in y_arr])

            x_train_data = x_arr.reshape(x_arr.shape[0], 48, 48, 1).astype(np.float32)#(28709,48,48,1)
            y_train_data = y_arr.astype(np.int)#(28709,1)

    x_train_data /= 255

    ratio = 0.9
    xSize = int(x_train_data.shape[0]*ratio)
    ySize = int(y_train_data.shape[0]*ratio)
    x_val_data = x_train_data[xSize:]
    y_val_data = y_train_data[ySize:]

    return x_val_data, y_val_data

def main():
    model_path = os.path.join(base_dir, "model.h5")
    emotion_classifier = load_model(model_path)

    private_pixels, private_label = read_data(sys.argv[1])

    input_img = emotion_classifier.input
    img_ids = [5]#["image ids from which you want to make heatmaps"]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx].reshape(1,48,48,1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = None
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        heatmap = np.array(fn([private_pixels[idx].reshape(1,48,48,1), 1])).reshape(48,48)
        heatmap = np.absolute(heatmap)
        heatmap = (heatmap.astype(float) - np.amin(heatmap)) / (np.amax(heatmap) - np.amin(heatmap))

        thres = 0.5
        see = np.copy(private_pixels[idx].reshape(48, 48))
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        #plt.axis('off')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}_heat.png'.format(idx)), dpi=100)
        #fig.savefig(os.path.join(cmap_dir, '{}_heat_w.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        #plt.axis('off')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}_part.png'.format(idx)), dpi=100)
        #fig.savefig(os.path.join(partial_see_dir, '{}_part_w.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow((private_pixels[idx].reshape(48,48)), cmap='gray')
        plt.colorbar()
        #plt.axis('off')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(orig_see_dir, '{}_orig.png'.format(idx)), dpi=100)
        #fig.savefig(os.path.join(orig_see_dir, '{}_orig_W.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()