from __future__ import division
from keras.datasets import mnist
from keras.models import load_model
from keras.callbacks import EarlyStopping
from vae_m1 import VAEM1
from vae_m2 import VAEM2
import numpy as np
import os

nb_epoch = 30
custom_objects = {}


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    X_train = X_train/255.0
    X_test = X_test/255.0
    X_train[X_train > 0.5] = 1.0
    X_train[X_train <= 0.5] = 0.0
    X_test[X_test > 0.5] = 1.0
    X_test[X_test <= 0.5] = 0.0

    if os.path.exists('./trained_model/encoder_m1.h5') and os.path.exists('./trained_model/decoder_m1.h5'):
        encoder_m1 = load_model('./trained_model/encoder_m1.h5', custom_objects=custom_objects)
        decoder_m1 = load_model('./trained_model/decoder_m1.h5', custom_objects=custom_objects)
    else:
        vaem1 = VAEM1()
        training = vaem1.training_model()
        training.compile(optimizer='adam', loss=vaem1.cost)
        training.fit(X_train, X_train,
                     batch_size=100,
                     nb_epoch=nb_epoch,
                     shuffle=True,
                     validation_data=(X_test, X_test),
                     callbacks=[EarlyStopping(patience=3)]
                     )
        encoder_m1 = vaem1.encoder()
        encoder_m1.save('./trained_model/encoder_m1.h5')
        decoder_m1 = vaem1.decoder()
        decoder_m1.save('./trained_model/decoder_m1.h5')

    z1_train = encoder_m1.predict(X_train, batch_size=100)
    z1_test = encoder_m1.predict(X_test, batch_size=100)

    #####################
    # only labeled data #
    #####################
    vaem2 = VAEM2()
    label_training = vaem2.label_training_model()
    label_training.compile(optimizer='adam', loss=vaem2.label_cost)
    label_training.fit([z1_train, y_train],
                       z1_train,
                       batch_size=100,
                       nb_epoch=nb_epoch,
                       validation_data=([z1_test, y_test], z1_test),
                       callbacks=[EarlyStopping(patience=3)],
                       shuffle=True)

    ##############################
    # labeled and unlabeled data #
    ##############################
    #y_u_train = np.tile(np.eye(10), (len(z1_train), 1)).reshape(len(z1_train), 10, 10)
    #y_u_test = np.tile(np.eye(10), (len(z1_test), 1)).reshape(len(z1_test), 10, 10)
    #training = vaem2.training_model()
    #training.compile(optimizer='adam', loss=vaem2.cost)
    #training.fit([z1_train, y_train, z1_train, y_u_train],
    #             z1_train,
    #             nb_epoch=nb_epoch,
    #             validation_data=([z1_test, y_test, z1_test, y_u_test], z1_test)
    #             shuffle=True)

    encoder_m2 = vaem2.encoder()
    encoder_m2.save('./trained_model/encoder_m2.h5')
    decoder_m2 = vaem2.decoder()
    decoder_m2.save('./trained_model/decoder_m2.h5')
    classifier_m2 = vaem2.classifier()
    classifier_m2.save('./trained_model/classifier_m2.h5')
