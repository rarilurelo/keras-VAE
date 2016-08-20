from __future__ import division
from keras.layers import Input, Dense, Activation, Merge
from keras.models import Model, Sequential
import keras.backend as K
from probability_distributions import GaussianDistribution, BernoulliDistribution, CategoricalDistribution
from custom_batchnormalization import CustomBatchNormalization

class VAEM2(object):
    def __init__(self, in_dim=50, cat_dim=10, hid_dim=300, z_dim=50, alpha=1):
        self.in_dim = in_dim
        self.cat_dim = cat_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.alpha = alpha
        self.x_l = Input((self.in_dim, ))
        self.x_u = Input((self.in_dim, ))
        self.y_l = Input((self.cat_dim, ))
        y_u0 = Input((self.cat_dim, ))
        y_u1 = Input((self.cat_dim, ))
        y_u2 = Input((self.cat_dim, ))
        y_u3 = Input((self.cat_dim, ))
        y_u4 = Input((self.cat_dim, ))
        y_u5 = Input((self.cat_dim, ))
        y_u6 = Input((self.cat_dim, ))
        y_u7 = Input((self.cat_dim, ))
        y_u8 = Input((self.cat_dim, ))
        y_u9 = Input((self.cat_dim, ))
        self.y_u = [y_u0, y_u1, y_u2, y_u3, y_u4, y_u5, y_u6, y_u7, y_u8, y_u9]
        self.z = Input((self.z_dim, ))

        ###############
        # q(z | x, y) #
        ###############
        x_branch = Sequential()
        x_branch.add(Dense(self.hid_dim, input_dim=self.in_dim))
        x_branch.add(CustomBatchNormalization())
        x_branch.add(Activation('softplus'))
        y_branch = Sequential()
        y_branch.add(Dense(self.hid_dim, input_dim=self.cat_dim))
        y_branch.add(CustomBatchNormalization())
        y_branch.add(Activation('softplus'))
        merged = Sequential([Merge([x_branch, y_branch], mode='concat')])
        merged.add(Dense(self.hid_dim))
        merged.add(CustomBatchNormalization())
        merged.add(Activation('softplus'))
        mean = Sequential([merged])
        mean.add(Dense(self.hid_dim))
        mean.add(CustomBatchNormalization())
        mean.add(Activation('softplus'))
        mean.add(Dense(self.z_dim))
        var = Sequential([merged])
        var.add(Dense(self.hid_dim))
        var.add(CustomBatchNormalization())
        var.add(Activation('softplus'))
        var.add(Dense(self.z_dim, activation='softplus'))
        self.q_z_xy = GaussianDistribution(self.z, givens=[self.x_l, self.y_l], mean_model=mean, var_model=var)

        ###############
        # p(x | y, z) #
        ###############
        y_branch = Sequential()
        y_branch.add(Dense(self.hid_dim, input_dim=self.cat_dim))
        y_branch.add(CustomBatchNormalization())
        y_branch.add(Activation('softplus'))
        z_branch = Sequential()
        z_branch.add(Dense(self.hid_dim, input_dim=self.z_dim))
        z_branch.add(CustomBatchNormalization())
        z_branch.add(Activation('softplus'))
        merged = Sequential([Merge([y_branch, z_branch], mode='concat')])
        merged.add(Dense(self.hid_dim))
        merged.add(CustomBatchNormalization())
        merged.add(Activation('softplus'))
        mean = Sequential([merged])
        mean.add(Dense(self.hid_dim))
        mean.add(CustomBatchNormalization())
        mean.add(Activation('softplus'))
        mean.add(Dense(self.in_dim))
        var = Sequential([merged])
        var.add(Dense(self.hid_dim))
        var.add(CustomBatchNormalization())
        var.add(Activation('softplus'))
        var.add(Dense(self.in_dim, activation='softplus'))
        self.p_x_yz = GaussianDistribution(self.x_l, givens=[self.y_l, self.z], mean_model=mean, var_model=var)

        ########
        # p(y) #
        ########
        self.p_y = CategoricalDistribution(self.y_l)

        ############
        # q(y | x) #
        ############
        inference = Sequential()
        inference.add(Dense(self.hid_dim, input_dim=self.in_dim))
        inference.add(CustomBatchNormalization())
        inference.add(Activation('softplus'))
        inference.add(Dense(self.hid_dim))
        inference.add(CustomBatchNormalization())
        inference.add(Activation('softplus'))
        inference.add(Dense(self.cat_dim, activation='softmax'))
        self.q_y_x = CategoricalDistribution(self.y_l, givens=[self.x_l], model=inference)

        ##########################
        # sample and reconstruct #
        ##########################
        self.sampling_z = self.q_z_xy.sampling(givens=[self.x_l, self.y_l])
        self.reconstruct_x_l = self.p_x_yz.sampling(givens=[self.y_l, self.sampling_z])

    def _KL(self, mean, var):
        return -1/2*K.mean(K.sum(1+K.log(var)-mean**2-var, axis=1))

    def label_cost(self, y_true, y_false):
        ###########
        # Labeled #
        ###########
        self.mean, self.var = self.q_z_xy.get_params(givens=[self.x_l, self.y_l])
        KL = self._KL(self.mean, self.var)
        logliklihood = -self.p_x_yz.logliklihood(self.x_l, givens=[self.y_l, self.sampling_z])-self.p_y.logliklihood(self.y_l)
        L = KL+logliklihood
        L = L+self.alpha*self.q_y_x.logliklihood(self.y_l, givens=[self.x_l])
        return L

    def cost(self, y_true, y_false):
        ###########
        # Labeled #
        ###########
        self.mean, self.var = self.q_z_xy.get_params(givens=[self.x_l, self.y_l])
        KL = self._KL(self.mean, self.var)
        logliklihood = -self.p_x_yz.logliklihood(self.x_l, givens=[self.y_l, self.sampling_z])-self.p_y.logliklihood(self.y_l)
        L = self.KL+self.logliklihood
        L = self.L+self.alpha*self.q_y_x.logliklihood(self.y_l, givens=[self.x_l])

        #############
        # UnLabeled #
        #############
        U = 0
        # marginalization
        for y in self.y_u:
            mean, var = self.q_z_xy.get_params(givens=[self.x_u, y])
            sampling_z = self.q_z_xy.sampling(givens=[self.x_u, y])
            U += self.q_y_x.prob(y, givens=[x])*(-self.p_x_yz.logliklihood(self.x_u, givens=[y, sampling_z])
                                                   -self.p_y.logliklihood(y)
                                                   +self._KL(mean, var)
                                                   +self.q_y_x.logliklihood(y, givens=[self.x_u])
                                                )
        return U+L

    def label_training_model(self):
        model = Model(input=[self.x_l, self.y_l], output=self.reconstruct_x_l)
        return model

    def training_model(self):
        model = Model(input=[self.x_l, self.y_l, self.x_u]+self.y_u, output=self.reconstruct_x_l)
        return model

    def encoder(self):
        model = Model(input=[self.x_l, self.y_l], output=self.mean)
        return model

    def decoder(self):
        decode = self.p_x_yz.sampling(givens=[self.y_l, self.z])
        model = Model(input=[self.y_l, self.z], output=decode)
        return model

    def classifier(self):
        inference = self.q_y_x.get_params(givens=[self.x_l])
        model = Model(input=self.x_l, output=inference)
        return model


