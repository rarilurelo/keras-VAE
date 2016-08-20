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
    def __init__(self, variable, givens=None, mean=0, var=1, mean_model=None, var_model=None):
        if not isinstance(givens, list):
            raise ValueError()
        self.variable = variable
        # the class only accept rank 2 variable(bacth, z_dim)
        # need a little bit change to apply conv and deconv
        self.variable_shape = K.int_shape(self.variable)
        def sample(args):
            mean, var = args
            epsilon = K.random_normal(K.shape(mean))
            return mean+var*epsilon
        self.draw = Lambda(sample)
        self.mean_model = mean_model
        self.var_model = var_model
        if givens is None:
            if isinstance(mean, float) or isinstance(mean, int):
                self.mean = K.ones_like(self.variable)*mean
            else:
                self.mean = mean
            if isinstance(var, float) or isinstance(var, int):
                self.var = K.ones_like(self.variable)*var
            else:
                self.var = var

    def get_params(self, givens=None):
        if givens is None:
            return self.mean, self.var
        mean = self.mean_model(givens)
        var = self.var_model(givens)
        return mean, var

    def sampling(self, givens=None):
        if givens is None:
            return self.draw([self.mean, self.var])
        mean = self.mean_model(givens)
        var = self.var_model(givens)
        return self.draw([mean, var])

    def prob(self, variable, givens=None):
        if givens is None:
            return 1/K.sqrt(2*np.pi*self.var)*K.exp(-1/2*(variable-self.mean)**2/self.var)
        mean = self.mean_model(givens)
        var = self.var_model(givens)
        return 1/K.sqrt(2*np.pi*var)*K.exp(-1/2*(variable-mean)**2/var)

    def _log_gausian(self, variable, mean, var):
        return -1/2*K.log(2*np.pi*var)-1/2*(variable-mean)**2/var

    def logliklihood(self, variable, givens=None):
        """
        a mean logliklihood of minibatch
        """
        if givens is None:
            return K.mean(K.sum(self._log_gausian(variable, self.mean, self.var), axis=1))
        mean = self.mean_model(givens)
        var = self.var_model(givens)
        return K.mean(K.sum(self._log_gausian(variable, mean, var), axis=1))


class BernoulliDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, pi=0.5, model=None):
        self.variable = variable
        self.variable_shape = K.int_shape(self.variable)
        def sample(args):
            pi = args
            return K.random_binomial(shape=K.shape(pi), p=pi)
        self.draw = Lambda(sample)
        self.model = model
        if givens is None:
            """
            this distribution is prior
            set params
            """
            if isinstance(pi, float) or isinstance(pi, int):
                self.pi = K.ones_like(self.variable)*pi
            else:
                self.pi = pi

    def get_params(self, givens=None):
        if givens is None:
            return self.pi
        pi = self.model(givens)
        return pi

    def sampling(self, givens=None):
        if givens is None:
            return self.draw(self.pi)
        pi = self.model(givens)
        return self.draw(pi)

    def _bernoulli(self, variable, pi):
        return pi**variable*(1-pi)**(1-variable)

    def prob(self, variable, givens=None):
        if givens is None:
            return self._bernoulli(variable, self.pi)
        pi = self.model(givens)
        return self._bernoulli(variable, pi)

    def _help_logliklihood(self, variable, pi):
        return K.mean(K.sum(variable*K.log(K.clip(pi, K._epsilon, 1-K._epsilon))+(1-variable)*K.log(K.clip(1-pi, K._epsilon, 1-K._epsilon)), axis=1))
        #return K.mean(K.sum(variable*K.log(pi)+(1-variable)*K.log(1-pi), axis=1))


    def logliklihood(self, variable, givens=None):
        if givens is None:
            return self._help_logliklihood(variable, self.pi)
        pi = self.model(givens)
        return self._help_logliklihood(variable, pi)


class CategoricalDistribution(ProbabilityDistribution):
    def __init__(self, variable, givens=None, pi=None, model=None):
        self.variable = variable
        self.variable_shape = K.int_shape(self.variable)
        if not K.ndim(self.variable) == 2:
            raise ValueError()
        def sample(args):
            pi = args
            if K._BACKEND == 'tensorflow':
                import tensorflow as tf
                draw = tf.multinomial(pi, num_samples=1)
                return tf.one_hot(draw, self.variable_shape[1])
            else:
                # using theano backend
                raise NotImplementedError()
        self.draw = Lambda(sample)
        self.model = model
        if givens is None:
            if pi is not None:
                self.pi = pi
            else:
                self.pi = K.ones_like(self.variable)*(1/self.variable_shape[1])

    def get_params(self, givens=None):
        if givens is None:
            return self.pi
        pi = self.model(givens)
        return pi

    def sampling(self, givens=None):
        if givens is None:
            return self.draw(self.pi)
        pi = self.model(givens)
        return self.draw(pi)

    def _help_prob(self, variable, pi):
        return K.prod(pi**variable, axis=1)

    def prob(self, variable, givens=None):
        if givens is None:
            return self._help_prob(variable, self.pi)
        pi = self.model(givens)
        return self._help_prob(variable, pi)

    def _help_logliklihood(self, variable, pi):
        return K.mean(K.sum(variable*K.log(K.clip(pi, K._epsilon, 1-K._epsilon)), axis=1))

    def logliklihood(self, variable, givens=None):
        if givens is None:
            return self._help_logliklihood(variable, self.pi)
        pi = self.model(givens)
        return self._help_logliklihood(variable, pi)


