from __future__ import division
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Lambda, Merge

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
    def __init__(self, variable, givens=None, hid_dim=200, mean=0, var=1, mean_model=None, log_var_model=None):
        if not isinstance(givens, list):
            raise ValueError()
        self.variable = variable
        self.givens = givens
        self.hid_dim = hid_dim
        # the class only accept rank 2 variable(bacth, z_dim)
        # need a little bit change to apply conv and deconv
        self.z_dim = K.int_shape(self.variable)[1]
        def sample(args):
            mean, log_var = args
            epsilon = K.random_normal(K.shape(mean))
            return mean+K.exp(log_var)*epsilon
        self.draw = Lambda(sample)
        def _merge(args):
            if any([K.ndim(arg) > 2 for arg in args]):
                return args
            if len(args) == 1:
                return args[0]
            else:
                return Merge(args, mode='concat', concat_axis=1)
        self.merge = _merge
        if self.givens is not None:
            if mean_model is None and log_var_model is None:
                in_dim = K.int_shape(K.concatenate(self.givens, axis=1))[1]
                self.model = Sequential()
                self.model.add(Dense(self.hid_dim, input_dim=in_dim, activation='relu'))
                self.model.add(Dense(self.hid_dim, activation='relu'))
                self.model.add(Dense(self.z_dim*2))
                self.mean_model = Sequential(layers=[self.model])
                self.mean_model.add(Dense(self.z_dim))
                self.log_var_model = Sequential(layers=[self.model])
                self.log_var_model.add(Dense(self.z_dim))
                self.mean = self.mean_model(self.merge(self.givens))
                self.log_var = self.log_var_model(self.merge(self.givens))
                self.var = K.exp(self.log_var)
            else:
                # expect CNN or other arcitecture including multi mordal
                self.mean_model = mean_model
                self.log_var_model = log_var_model
                self.mean = self.mean_model(self.merge(self.givens))
                self.log_var = self.log_var_model(self.merge(self.givens))
                self.var = K.exp(self.log_var)
        else:
            self.mean_model = mean_model
            self.log_var_model = log_var_model
            if isinstance(mean, float) or isinstance(mean, int):
                self.mean = K.ones_like(self.variable)*mean
            else:
                self.mean = mean
            if isinstance(var, float) or isinstance(var, int):
                self.var = K.ones_like(self.variable)*var
            else:
                self.var = var
            self.log_var = K.log(self.var)

    def get_params(self, givens=None):
        if givens is None:
            return self.mean, self.var
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            mean = self.mean_model(self.merge(givens))
            log_var = self.log_var_model(self.merge(givens))
            var = K.exp(log_var)
            return mean, var
        return self.mean, self.var

    def sampling(self, givens=None):
        if givens is None:
            return self.draw([self.mean, self.log_var])
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            mean = self.mean_model(self.merge(givens))
            log_var = self.log_var_model(self.merge(givens))
            return self.draw([mean, log_var])
        return self.draw([self.mean, self.log_var])

    def prob(self, variable, givens=None):
        if givens is None:
            if variable is not self.variable:
                return 1/K.sqrt(2*np.pi*self.var)*K.exp(-1/2*(variable-self.mean)**2/self.var)
            else:
                return 1/K.sqrt(2*np.pi*self.var)*K.exp(-1/2*(self.variable-self.mean)**2/self.var)
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            mean = self.mean_model(self.merge(givens))
            log_var = self.log_var_model(self.merge(givens))
            var = K.exp(log_var)
            if variable is not self.variable:
                return 1/K.sqrt(2*np.pi*var)*K.exp(-1/2*(variable-mean)**2/var)
            else:
                return 1/K.sqrt(2*np.pi*var)*K.exp(-1/2*(self.variable-mean)**2/var)
        else:
            if variable is not self.variable:
                return 1/K.sqrt(2*np.pi*self.var)*K.exp(-1/2*(variable-self.mean)**2/self.var)
            else:
                return 1/K.sqrt(2*np.pi*self.var)*K.exp(-1/2*(self.variable-self.mean)**2/self.var)


    def _log_gausian(self, variable, mean, var):
        return -1/2*K.log(2*np.pi*var)-1/2*(variable-mean)**2/var

    def logliklihood(self, variable, givens=None):
        """
        a mean logliklihood of minibatch
        """
        if givens is None:
            if variable is not self.variable:
                return K.mean(K.sum(self._log_gausian(variable, self.mean, self.var), axis=1))
            else:
                return K.mean(K.sum(self._log_gausian(self.variable, self.mean, self.var), axis=1))
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            mean = self.mean_model(self.merge(givens))
            log_var = self.log_var_model(self.merge(givens))
            var = K.exp(log_var)
            if variable is not self.variable:
                return K.mean(K.sum(self._log_gausian(variable, mean, var), axis=1))
            else:
                return K.mean(K.sum(self._log_gausian(self.variable, mean, var), axis=1))
        else:
            if variable is not self.variable:
                return K.mean(K.sum(self._log_gausian(variable, self.mean, self.var), axis=1))
            else:
                return K.mean(K.sum(self._log_gausian(self.variable, self.mean, self.var), axis=1))


class BernoulliDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, hid_dim=200, pi=0.5, model=None):
        self.variable = variable
        self.givens = givens
        self.hid_dim = hid_dim
        self.z_dim = K.int_shape(self.variable)[1]
        def sample(args):
            pi = args
            return K.random_binomial(shape=K.shape(pi), p=pi)
        self.draw = Lambda(sample)
        def _merge(args):
            if any([K.ndim(arg) > 2 for arg in args]):
                return args
            if len(args) == 1:
                return args[0]
            else:
                return Merge(args, mode='concat', concat_axis=1)
        self.merge = _merge
        if self.givens is not None:
            """
            compute params by givens
            """
            if model is None:
                in_dim = K.int_shape(K.concatenate(self.givens, axis=1))[1]
                self.model = Sequential()
                self.model.add(Dense(self.hid_dim, input_dim=in_dim, activation='relu'))
                self.model.add(Dense(self.hid_dim, activation='relu'))
                self.model.add(Dense(self.z_dim, activation='sigmoid'))
                self.pi = self.model(self.merge(self.givens))
            else:
                self.model = model
                self.pi = self.model(self.givens)
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

    def get_params(self, givens=None):
        if givens is None:
            return self.pi
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            return pi
        return self.pi

    def sampling(self, givens=None):
        if givens is None:
            return self.draw(self.pi)
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            return self.draw(pi)
        return self.draw(self.pi)

    def _bernoulli(self, variable, pi):
        return pi**variable*(1-pi)**(1-variable)

    def prob(self, variable, givens=None):
        if givens is None:
            if variable is not self.variable:
                return self._bernoulli(variable, self.pi)
            else:
                return self._bernoulli(self.variable, self.pi)
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            if variable is not self.variable:
                return self._bernoulli(variable, pi)
            else:
                return self._bernoulli(self.variable, pi)
        else:
            if variable is not self.variable:
                return self._bernoulli(variable, self.pi)
            else:
                return self._bernoulli(self.variable, self.pi)

    def _help_logliklihood(self, variable, pi):
        return K.mean(K.sum(variable*K.log(K.clip(pi, K._epsilon, 1-K._epsilon))+(1-variable)*K.log(K.clip(1-pi, K._epsilon, 1-K._epsilon)), axis=1))

    def logliklihood(self, variable, givens=None):
        if givens is None:
            if variable is not self.variable:
                return self._help_logliklihood(variable, self.pi)
            else:
                return self._help_logliklihood(self.variable, self.pi)
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            if variable is not self.variable:
                return self._help_logliklihood(variable, pi)
            else:
                return self._help_logliklihood(self.variable, pi)
        else:
            if variable is not self.variable:
                return self._help_logliklihood(variable, self.pi)
            else:
                return self._help_logliklihood(self.variable, self.pi)


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, hid_dim=200, model=None):
        self.variable = variable
        self.givens = givens
        self.hid_dim = hid_dim
        self.z_dim = K.int_shape(self.variable)[1]
        def sample(args):
            pi = args
            if K._BACKEND == 'tensorflow':
                import tensorflow as tf
                draw = tf.multinomial(pi, num_samples=1)
                return tf.one_hot(draw, self.z_dim)
            else:
                # using theano backend
                raise NotImplementedError()
        self.draw = Lambda(sample)
        def _merge(args):
            if any([K.ndim(arg) > 2 for arg in args]):
                return args
            if len(args) == 1:
                return args[0]
            else:
                return Merge(args, mode='concat', concat_axis=1)
        self.merge = _merge
        if self.givens is not None:
            if model is None:
                in_dim = K.int_shape(K.concatenate(self.givens, axis=1))[1]
                self.model = Sequential()
                self.model.add(Dense(self.hid_dim, input_dim=in_dim, activation='relu'))
                self.model.add(Dense(self.hid_dim, activation='relu'))
                self.model.add(Dense(self.z_dim), activation='softmax')
                self.pi = self.model(self.merge(self.givens))
            else:
                self.model = model
                self.pi = self.model(self.givens)
        else:
            self.model = model
            self.pi = K.ones_like(self.variable)*(1/self.z_dim)

    def get_params(self, givens=None):
        if givens is None:
            return self.pi
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            return pi
        return self.pi

    def sampling(self, givens=None):
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            if givens is None:
                return self.draw(self.pi)
            if not all([given is _given for given, _given in zip(givens, self.givens)]):
                pi = self.model(self.merge(givens))
                return self.draw(pi)
            return self.draw(self.pi)
        else:
            # using theano backend
            raise NotImplementedError()

    def _help_prob(self, variable, pi):
        return K.prod(pi**variable, axis=1)

    def prob(self, variable, givens=None):
        if givens is None:
            if variable is not self.variable:
                return self._help_prob(variable, self.pi)
            else:
                return self._help_prob(self.variable, self.pi)
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            if variable is not self.variable:
                return self._help_prob(variable, pi)
            else:
                return self._help_prob(self.variable, pi)
        else:
            if variable is not self.variable:
                return self._help_prob(variable, self.pi)
            else:
                return self._help_prob(self.variable, self.pi)

    def _help_logliklihood(self, variable, pi):
        return K.mean(K.sum(variable*K.log(K.clip(pi, K._epsilon, 1-K._epsilon)), axis=1))


    def logliklihood(self, variable, givens=None):
        if givens is None:
            if variable is not self.variable:
                return self._help_logliklihood(variable, self.pi)
            else:
                return self._help_logliklihood(self.variable, self.pi)
        if not all([given is _given for given, _given in zip(givens, self.givens)]):
            pi = self.model(self.merge(givens))
            if variable is not self.variable:
                return self._help_logliklihood(variable, pi)
            else:
                return self._help_logliklihood(self.variable, pi)
        else:
            if variable is not self.variable:
                return self._help_logliklihood(variable, self.pi)
            else:
                return self._help_logliklihood(self.variable, self.pi)


