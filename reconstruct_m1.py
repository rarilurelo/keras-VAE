from keras.models import load_model
from keras.datasets import mnist
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

    target = X_train[0:5]

    latent = encoder.predict(target, batch_size=5)
    reconstruct_image = decoder.predict(latent, batch_size=5)

    fig = plt.figure(figsize=(14, 14))
    for i, image in enumerate(images):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        ax.imshow(target.reshape(28, 28), 'gray')
    for i, reconstruct_image in enumerate(reconstruct_images):
        ax = fig.add_subplot(2, 5, 6+i, xticks=[], yticks=[])
        ax.imshow(reconstruct_image.reshape(28, 28), 'gray')
    plt.savefig('./images/reconstruct_m1.png')






