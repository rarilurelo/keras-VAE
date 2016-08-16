from keras.models import load_model
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets import fetch_mldata
from custom_batchnormalization import CustomBatchNormalization


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('rotation', True, 'use rotate dataset?')
rotation = 'rot' if FLAGS.rotation else 'normal'

def get_data():
    if FLAGS.rotation:
        if os.path.exists('./data/rotation.npz'):
            data = np.load('./data/rotation.npz')
            X_data = data['x']
            y_data = data['y']
        else:
            X_data, y_data = get_rotation()
            np.savez('./data/rotation.npz', x=X_data, y=y_data)
    else:
        mnist = fetch_mldata('MNIST original')
        X_data, y_data = mnist.data, mnist.target.astype(np.int32)
        X_data = X_data/255.0
        y_data = np.eye(np.max(y_data)+1)[y_data]
    return X_data, y_data

if __name__ == '__main__':
    X_data, y_data = get_data()
    X_data, y_data = shuffle(X_data, y_data)

    encoder = load_model("./trained_model/encoder_{}.h5".format(rotation), custom_objects={'CustomBatchNormalization': CustomBatchNormalization})

    decoder = load_model("./trained_model/decoder_{}.h5".format(rotation))

    images = X_data[0:5]
    labels = y_data[0:5]

    latents = encoder.predict([images, labels], batch_size=5)
    reconstruct_images = decoder.predict([labels, latents], batch_size=5)

    fig = plt.figure(figsize=(14, 14))
    for i, image in enumerate(images):
        ax = fig.add_subplot(2, 5, i+1)
        ax.imshow(image.reshape(28, 28))
    for i, reconstruct_image in enumerate(reconstruct_images):
        ax = fig.add_subplot(2, 5, 6+i)
        ax.imshow(reconstruct_image.reshape(28, 28))
    plt.savefig("./images/reconstruct_image_{}.png".format(rotation))







