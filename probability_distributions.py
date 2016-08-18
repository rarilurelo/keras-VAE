from __future__ import division
import keras.backend as K
from keras.models import Sequential

import numpy as np

class ProbabilityDistribution(object):
    """
    Abstruct class of ProbabilityDistribution.
    This class contains Neural Network Arcitecture of keras.
    """
    def sampling(self):
        raise NotImplementedError()

    def prob(self):
        raise NotImplementedError()

    def logliklihood(self):
        raise NotImplementedError()


class GaussianDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, hid_dim=200, z_dim=50, mean=0, log_var=np.e, model=None):
        """
        If you pass a model, it must return z_dim*2 shape
        """
        self.variable = variable
        self.givens = givens
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        if self.givens is not None:
            if model is None:
                in_dim = K.int_shape(K.concatenate(self.givens, axis=1))[1]
                self.model = Sequential()
                self.model.add(Dense(self.hid_dim, input_shape=in_dim, activation='relu'))
                self.model.add(Dense(self.hid_dim, activation='relu'))
                self.model.add(Dense(self.z_dim*2))
                self.params = self.model(K.concatenate(self.givens, axis=1))
                self.mean = self.params[:, :z_dim]
                self.log_var = self.params[:, z_dim:]
                self.var = K.exp(self.log_var)
            else:
                # expect CNN or other arcitecture including multi mordal
                self.model = model
                self.params = self.model(givens)
                self.mean = self.params[:, :z_dim]
                self.log_var = self.params[:, z_dim:]
                self.var = K.exp(self.log_var)
        else:
            self.model = model
            if isinstance(mean, float) or isinstance(mean, int):
                self.mean = K.ones_like(self.variable)*mean
            else:
                self.mean = mean
            if isinstance(log_var, float) or isinstance(log_var, int):
                self.log_var = K.ones_like(self.variable)*log_var
            else:
                self.log_var = log_var
            self.var = K.exp(self.log_var)
            self.params = K.concatenate([self.mean, self.log_var], axis=1)

    def sampling(self):
        epsilon = K.random_normal(K.shape(self.mean))
        return self.mean+self.var*epsilon

    def prob(self):
        return 1/K.sqrt(2*np.pi*self.var)*K.exp(-1/2*(self.variable-self.mean)**2/self.var)

    def logliklihood(self):
        """
        a mean logliklihood of minibatch
        """
        return K.mean(K.sum(-1/2*K.log(2*np.pi*self.var)-1/2*(self.variable-self.mean)**2/self.var, axis=1))


class BernulliDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, hid_dim=200, z_dim=50, pi=0.5, model=None):
        self.variable = variable
        self.givens = givens
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        if self.givens is not None:
            """
            compute params by givens
            """
            if model is None:
                in_dim = K.int_shape(K.concatenate(self.givens, axis=1))[1]
                self.model = Sequential()
                self.model.add(Dense(self.hid_dim, input_dim=in_dim, activation='relu'))
                self.model.add(Dense(self.hid_dim, activation='relu'))
                self.model.add(Dense(self.z_dim))
                self.params = self.model(K.concatenate(self.givens, axis=1))
                self.pi = self.params
            else:
                self.model = model
                self.params = self.model(self.givens)
                self.pi = self.params
        else:
            """
            this distribution is prior
            set params
            """
            self.model = model
            if isinstance(pi, float) or isinstance(pi, int):
                self.pi = K.ones_like(self.variable)*pi
            else:
                self.pi = pi
            self.params = self.pi

    def sampling(self):
        sample = K.random_uniform(shape=[1])
        return K.switch(sample <= self.pi, 1, 0)

    def prob(self):
        return self.pi**self.variable*(1-self.pi)**(1-self.variable)

    def logliklihood(self):
        return K.mean(K.sum(self.variable*K.log(K.clip(self.pi, K._EPSILON, 1-K._EPSILON))+(1-self.variable)*K.log(K.clip(1-self.pi, K._EPSILON, 1-K._EPSILON)), axis=1))


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, hid_dim=200, z_dim=10, model=None):
        self.variable = variable
        self.givens = givens
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        if self.givens is not None:
            if model is None:
                in_dim = K.int_shape(K.concatenate(self.givens, axis=1))[1]
                self.model = Sequential()
                self.model.add(Dense(self.hid_dim, input_dim=in_dim, activation='relu'))
                self.model.add(Dense(self.hid_dim, activation='relu'))
                self.model.add(Dense(self.z_dim), activation='softmax')
                self.params = self.model(K.concatenate(slef.givens, axis=1))
                self.pi = self.params
            else:
                self.model = model
                self.params = self.model(self.givens)
                self.pi = self.params
        else:
            self.model = model
            self.pi = K.ones_like(self.variable)*(1/self.z_dim)
            self.params = self.pi

    def sampling(self):
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            sample = tf.multinomial(self.pi, num_sumples=1)
            return tf.one_hot(sample, z_dim)
        else:
            # using theano backend
            raise NotImplementedError()

    def prob(self):
        return K.prod(self.pi**self.variable, axis=1)

    def logliklihood(self):
        return K.mean(K.sum(self.variable*K.log(K.clip(self.pi, K._EPSILON, 1-K._EPSILON)), axis=1))

