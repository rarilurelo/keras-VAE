from __future__ import division
from keras.layers import Input
from keras.models import Model
import keras.backend as K
from probability_distributions import GaussianDistribution, BernoulliDistribution


class VAEM1(object):
    def __init__(self, in_dim=784, hid_dim=300, z_dim=50):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.x = Input((self.in_dim, ))
        self.z = Input((self.z_dim, ))

        # givens determinate params of distribution by deep neural network.
        # hid_dim represent the hid_dim of deep neurl network
        self.p_x_z = BernoulliDistribution(self.x, givens=[self.z], hid_dim=self.hid_dim)
        self.q_z_x = GaussianDistribution(self.z, givens=[self.x], hid_dim=self.hid_dim)

        self.mean, self.var = self.q_z_x.get_params(givens=[self.x])
        self.sampling_z = self.q_z_x.sampling(givens=[self.x])
        self.reconstruct_x = self.p_x_z.sampling(givens=[self.sampling_z])

    def cost(self, inputs, output):
        self.KL = 1/2*K.mean(K.sum(1+K.log(self.var)-self.mean**2-self.var, axis=1))
        self.logliklihood = self.p_x_z.logliklihood(self.x, givens=[self.sampling_z])
        self.lower_bound = -self.KL+self.logliklihood
        self.lossfunc = -self.lower_bound
        return self.lossfunc

    def training_model(self):
        model = Model(input=self.x, output=self.reconstruct_x)
        return model

    def encoder(self):
        model = Model(input=self.x, output=self.mean)
        return model

    def decoder(self):
        decode = self.p_x_z.sampling(givens=[self.z])
        model = Model(input=self.z, output=decode)
        return model

