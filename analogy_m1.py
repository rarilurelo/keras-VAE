from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from custom_batchnormalization import CustomBatchNormalization

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

    encoder = load_model('./trained_model/encoder_m1.h5', custom_objects={'CustomBatchNormalization': CustomBatchNormalization})
    decoder = load_model('./trained_model/decoder_m1.h5', custom_objects={'CustomBatchNormalization': CustomBatchNormalization})

    target1 = X_train[0:1]
    target2 = X_train[8:9]

    latent1 = encoder.predict(target1, batch_size=1)
    latent2 = encoder.predict(target2, batch_size=1)

    fig = plt.figure(figsize=(14, 14))
    for i, d in enumerate(np.linspace(0, 1, 10)):
        latent = latent1+d*(latent2-latent1)
        reconstruct_image = decoder.predict(latent, batch_size=1)
        ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
        ax.imshow(reconstruct_image.reshape(28, 28), 'gray')
    plt.savefig('./images/analogy_m1.png')



