import os
import sys
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_dir, "model.h5")
filter_dir = os.path.join(base_dir, "filter")
if not os.path.exists(filter_dir):
    os.makedirs(filter_dir)
orig_see_dir = os.path.join(base_dir, "filter/origin")
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

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    for i in range(20):
        loss_value, grads_value = iter_func([input_image_data, 1])
        input_image_data += grads_value * num_step

    img = np.copy(input_image_data)
    img = deprocess_image(img)
    return img, loss_value

def main():    
    nb_filter = 32#128 #(draw_picture multiplies)
    draw_picture = 4

    num_step = 1
    pick_img = 25

    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input

    name_ls = ["dropout_1", "dropout_2", "dropout_3", "dropout_4"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    input_img_DATA, input_img_label = read_data(sys.argv[1])

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(nb_filter)]

        jump = 0
        for filter_idx in range(nb_filter):
            filter_imgs[filter_idx] = [[] for i in range (2)]
            input_img_data = np.copy(input_img_DATA[pick_img].reshape(1,48,48,1))

            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])

            ###
            "You need to implement it."
            filter_imgs[filter_idx][0], filter_imgs[filter_idx][1] = grad_ascent(num_step, input_img_data, iterate)
            ###
            print("Filter: " + str(jump))
            jump +=1

        plt.figure()
        plt.imshow((input_img_DATA[pick_img].reshape(48,48)), cmap='gray')
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(orig_see_dir, '{}.png'.format(pick_img)), dpi=100)

        fig = plt.figure(figsize=(28, 16))
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/draw_picture, draw_picture, i+1)
            ax.imshow(np.asarray(filter_imgs[i][0]).reshape(48,48), cmap='Greys')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('{:.3f}'.format(filter_imgs[i][1]))
            plt.tight_layout()

        fig.suptitle('Filters of layer {} (# {} Figures )'.format(name_ls[cnt], nb_filter))
        img_path = os.path.join(filter_dir, '{}-{}'.format("filter", name_ls[cnt]))
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'e{}'.format(nb_filter)))#it*RECORD_FREQ)))

if __name__ == "__main__":
    main()