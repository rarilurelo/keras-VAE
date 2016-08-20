from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

custom_objects = {}

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
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    encoder_m1 = load_model('./trained_model/encoder_m1.h5', custom_objects=custom_objects)
    decoder_m1 = load_model('./trained_model/decoder_m1.h5', custom_objects=custom_objects)
    encoder_m2 = load_model('./trained_model/encoder_m2.h5', custom_objects=custom_objects)
    decoder_m2 = load_model('./trained_model/decoder_m2.h5', custom_objects=custom_objects)

    targets = X_train[0:5]
    labels = y_train[0:5]


    z1 = encoder_m1.predict(targets, batch_size=5)
    print z1
    z2 = encoder_m2.predict([z1, labels], batch_size=5)
    reconstruct_z1 = decoder_m2.predict([labels, z2], batch_size=5)
    print reconstruct_z1
    reconstruct_images = decoder_m1.predict(reconstruct_z1, batch_size=5)

    fig = plt.figure(figsize=(14, 14))
    for i, target in enumerate(targets):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        ax.imshow(target.reshape(28, 28), 'gray')
    for i, reconstruct_image in enumerate(reconstruct_images):
        ax = fig.add_subplot(2, 5, 6+i, xticks=[], yticks=[])
        ax.imshow(reconstruct_image.reshape(28, 28), 'gray')
    plt.savefig('./images/reconstruct_m1_m2.png')







