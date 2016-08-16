# -*- coding: utf-8 -*-
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Input
from custom_batchnormalization import BatchNormalization
from keras import backend as K
from keras import objectives

class VAEM1M2(object):
    """
    VAE model M1+M2. This model can train only by supervised data.
    """
    def __init__(self, in_dim=784, hid_dim=300, z_dim=20, y_dim=10, batch_size=100, alpha=10):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.alpha = alpha

        self.x = Input((self.in_dim, ))
        self.y = Input((self.y_dim, ))

        # inference y
        self.inferencer = self.q_y_x()
        self.y_pred = self.inferencer(self.x)

        # sampling z
        self.encoder_mean, self.encoder_var = self.q_z_xy()
        self.mean = self.encoder_mean([self.x, self.y])
        self.var = self.encoder_var([self.x, self.y])
        def sampling(args):
            z_mean, z_var = args
            epsilon = K.random_normal(shape=(self.batch_size, self.z_dim))
            return z_mean+z_var*epsilon
        self.sampling_fn = Sequential([Merge([self.encoder_mean, self.encoder_var], mode=sampling, output_shape=lambda x: x[0])])
        self.z = self.sampling_fn([self.x, self.y])

        # reconstruct x
        self.decoder = self.p_x_yz()
        self.x_reconstruct = self.decoder([self.y, self.z])

    def q_y_x(self):
        q = Sequential()
        q.add(Dense(self.hid_dim, input_dim=self.in_dim))
        q.add(BatchNormalization())
        q.add(Activation('relu'))
        q.add(Dense(self.hid_dim))
        q.add(BatchNormalization())
        q.add(Activation('relu'))

        q.add(Dense(self.y_dim))
        q.add(Activation('softmax'))
        return q

    def q_z_xy(self):
        """
        sampling z given x and y
        """
        x_branch = Sequential()
        x_branch.add(Dense(self.hid_dim, input_dim=self.in_dim))
        x_branch.add(BatchNormalization())
        x_branch.add(Activation('relu'))

        y_branch = Sequential()
        y_branch.add(Dense(self.hid_dim, input_dim=self.y_dim))
        y_branch.add(BatchNormalization())
        y_branch.add(Activation('relu'))

        merged = Sequential([Merge([x_branch, y_branch], mode='concat')])
        merged.add(Dense(self.hid_dim))
        merged.add(BatchNormalization())
        merged.add(Activation('relu'))

        z_mean = Sequential([merged])
        z_mean.add(Dense(self.z_dim))
        z_mean.add(BatchNormalization())
        z_mean.add(Activation('relu'))

        z_var = Sequential([merged])
        z_var.add(Dense(self.z_dim))
        z_var.add(BatchNormalization())
        z_var.add(Activation('softmax'))


        return z_mean, z_var

    def p_x_yz(self):
        """
        reconstruct x give y and z
        """
        y_branch = Sequential()
        y_branch.add(Dense(self.hid_dim, input_dim=self.y_dim))
        y_branch.add(BatchNormalization())
        y_branch.add(Activation('relu'))

        z_branch = Sequential()
        z_branch.add(Dense(self.hid_dim, input_dim=self.z_dim))
        z_branch.add(BatchNormalization())
        z_branch.add(Activation('relu'))

        merged = Sequential([Merge([y_branch, z_branch], mode='concat')])
        merged.add(Dense(self.hid_dim))
        merged.add(BatchNormalization())
        merged.add(Activation('relu'))

        pi = Sequential([merged])
        pi.add(Dense(self.in_dim))
        pi.add(Activation('sigmoid'))

        return pi

    def loss_function(self, x, x_reconstruct):
        """
        This loss function only applies labeled data
        """
        logliklihood = objectives.binary_crossentropy(x, x_reconstruct)
        KL = -1/2*K.mean(K.sum(1+K.log(self.var)-self.mean**2-self.var, axis=1))
        Eq_y_x = objectives.categorical_crossentropy(self.y, self.y_pred)
        return KL-logliklihood+Eq_y_x*self.alpha







