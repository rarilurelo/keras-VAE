from keras.models import load_model
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets import fetch_mldata
from custom_batchnormalization import CustomBatchNormalization

def get_data():
    mnist = fetch_mldata('MNIST original')
    X_data, y_data = mnist.data, mnist.target.astype(np.int32)
    X_data = X_data/255.0
    y_data = np.eye(np.max(y_data)+1)[y_data]
    return X_data, y_data

if __name__ == '__main__':
    X_data, y_data = get_data()
    X_data, y_data = shuffle(X_data, y_data)

    encoder = load_model("./trained_model/encoder_normal.h5", custom_objects={'CustomBatchNormalization': CustomBatchNormalization})

    decoder = load_model("./trained_model/decoder_normal.h5", custom_objects={'CustomBatchNormalization': CustomBatchNormalization})

    targets = X_data[0:5]
    labels = y_data[0:5]

    latents = encoder.predict([targets, labels], batch_size=5)
    fig = plt.figure(figsize=(14, 14))
    for i, latent in enumerate(latents):
        ax = fig.add_subplot(5, 11, 11*i+1, xticks=[], yticks=[])
        ax.imshow(targets[i].reshape(28, 28), 'gray')
        for j, newlabel in enumerate(np.eye(10)):
            analogy_image = decoder.predict([newlabel.reshape(1, -1), latent.reshape(1, -1)], batch_size=1)
            ax = fig.add_subplot(5, 11, 11*i+j+2, xticks=[], yticks=[])
            ax.imshow(analogy_image.reshape(28, 28), 'gray')
    plt.savefig('./images/analogy_normal.png')

