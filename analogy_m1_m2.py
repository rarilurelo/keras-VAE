from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from custom_batchnormalization import CustomBatchNormalization

custom_objects = {'CustomBatchNormalization': CustomBatchNormalization}

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    X_train = X_train/255.0
    X_test = X_test/255.0
    X_train[X_train > 0.5] = 1.0
    X_train[X_train <= 0.5] = 0.0
    X_test[X_test > 0.5] = 1.0
    X_test[X_test <= 0.5] = 0.0
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    encoder_m1 = load_model('./trained_model/encoder_m1.h5', custom_objects=custom_objects)
    decoder_m1 = load_model('./trained_model/decoder_m1.h5', custom_objects=custom_objects)
    encoder_m2 = load_model('./trained_model/encoder_m2.h5', custom_objects=custom_objects)
    decoder_m2 = load_model('./trained_model/decoder_m2.h5', custom_objects=custom_objects)

    X_targets = X_train[9:17]
    y_targets = y_train[9:17]
    z1 = encoder_m1.predict(X_targets, batch_size=8)
    z2 = encoder_m2.predict([z1, y_targets], batch_size=8)

    fig = plt.figure(figsize=(14, 14))
    for i, z in enumerate(z2):
        ax = fig.add_subplot(8, 11, 11*i+1, xticks=[], yticks=[])
        ax.imshow(X_targets[i].reshape(28, 28), 'gray')
        for j, y in enumerate(np.eye(10)):
            z1_reconstruct = decoder_m2.predict([y.reshape(1, -1), z2[i].reshape(1, -1)], batch_size=1)
            x_reconstruct = decoder_m1.predict(z1_reconstruct, batch_size=1)
            ax = fig.add_subplot(8, 11, 11*i+j+2, xticks=[], yticks=[])
            ax.imshow(x_reconstruct.reshape(28, 28), 'gray')
    plt.savefig('./images/analogy_m1_m2.png')




