'''Variational autoencoder models and helpers'''

from __future__ import print_function
from __future__ import absolute_import

import keras
import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras import Model
from keras.layers import Input, Dense, Lambda, Conv1D, MaxPool1D, UpSampling1D, LSTM, RepeatVector
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

# Variational autoencoder sampling
def sampling(args):
    '''Reparameterization trick to calculate z'''
    mu, log_sigma = args
    batch_shape = mu.shape
    epsilon = K.random_normal(shape=batch_shape, mean=0., stddev=1.)
    return mu + (.5 * log_sigma) * epsilon

# Custom Callbacks
class CheckReconstruction(keras.callbacks.Callback):
    '''Qualitative check of name reconstruction'''
    def __init__(self,
                 type='letter',
                 n=5,
                 random=False):
        """

        :param type: 'letter' or 'shingle'
        :param n: number of name-reconstruction pairs to print
        :param random: display random n if True else display first n in batch
        """
        super().__init__()
        self.type = type
        self.n = n
        self.random = random

    def on_model_check(self, epoch, logs):
        pass

    def on_train_end(self, logs):
        pass

# Metrics
class KLDivergence():
    def __init__(self):
        pass

    def kldivergence(self):
        pass

# Helper
class BinaryEncoder():
    def __init__(self):
        pass

    def calculate_median_mu(self):
        pass

    def binary_encode(self):
        pass

# Encoders
def encoder_dense(batch_size,
                  orig_dim,
                  encode_dim,
                  latent_dim,
                  activation):
    '''Dense encoder with an arbitrary number of hidden layers'''
    if not isinstance(encode_dim, list):
        encode_dim = list(encode_dim)

    inp = Input(batch_shape=(batch_size, orig_dim))
    encode_layers = [inp]
    for i,units in enumerate(encode_dim):
        layer = Dense(units,
                      activation=activation,
                      name='enc_{}'.format(i))(encode_layers[-1])
        encode_layers.append(layer)
    return Model(inp, encode_layers[-1])


def encoder_conv(batch_size,
                 orig_dim,
                 activation):
    inp = Input(batch_shape=(batch_size, orig_dim))
    conv = Conv1D(activation=activation)(inp)
    pool = MaxPool1D()
    conv2 = Conv1D()
    encoded = MaxPool1D()
    return Model(inp, encoded)


def encoder_lstm(timesteps,
                 orig_dim,
                 latent_dim):
    inputs = Input(shape=(timesteps, orig_dim))
    return LSTM(latent_dim)(inputs)


def encoder_binary(encoder_path, train_data):
    pass


# Decoders
def decoder_dense(dec_input,
                  orig_dim,
                  decode_dim,
                  activation):
    '''Dense decoder with arbitrary number of hidden layers'''

    if not isinstance(decode_dim, list):
        decode_dim = list(decode_dim)
    decode_layers = [dec_input]
    for i, units in enumerate(decode_dim):
        layer = Dense(units,
                      activation=activation,
                      name='dec_{}'.format(i))(decode_layers[-1])
        decode_layers.append(layer)
    reconstruction = Dense(orig_dim, name='reconstruction')(decode_layers[-1])
    return Model(dec_input, reconstruction)


def decoder_conv(z):
    conv = Conv1D()(z)
    upsample = UpSampling1D()(conv)
    conv2 = Conv1D(upsample)
    upsample2 = UpSampling1D(conv2)
    return Model(conv, upsample2)


def decoder_lstm(timesteps, encoded, input_dim):
    decoded = RepeatVector(timesteps)(encoded)
    return LSTM(input_dim, return_sequences=True)(decoded)


# End to end models
class VAE():
    def __init__(self,
                 batch_size,
                 orig_dim,
                 latent_dim,
                 encode_dim,
                 decode_dim,
                 lr,
                 activation='relu'):
        self.batch_size = batch_size
        self.orig_dim = orig_dim
        self.latent_dim = latent_dim
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.lr = lr
        self.activation = activation

    def _build_model(self):
        encoder = encoder_dense(self.batch_size,
                                self.orig_dim,
                                self.encode_dim,
                                self.latent_dim,
                                self.activation)
        mu = Dense(self.latent_dim, name='mu')(encoder.layers[-1])
        log_sigma = Dense(self.latent_dim, name='log_sigma')(encoder.layers[-1])
        z = Lambda(sampling)([mu, log_sigma])
        decoder = decoder_dense(z,
                                self.orig_dim,
                                self.decode_dim,
                                self.activation)
        model = Model(encoder, decoder)
        print(model.summary())
        return model

    def train(self):
        K.clear_session()
        model = self._build_model()

        model.compile()

        earlystop = EarlyStopping()
        tensorboard = TensorBoard()
        check_reconstruction = CheckReconstruction()
        callbacks = [earlystop, tensorboard, check_reconstruction]

        model.fit()

        # Save full model
        model.save()

        # Save encoder

        # Save decoder


class ConvolutionalVAE(VAE):
    def __init__(self):
        super().__init__()

    def _build_model(self):
        encoder = encoder_conv()
        mu = Dense(self.latent_dim, name='mu')(encoder.layers[-1])
        log_sigma = Dense(self.latent_dim, name='log_sigma')(encoder.layers[-1])
        z = Lambda(sampling)([mu, log_sigma])
        decoder = decoder_conv(z)
        model = Model(encoder, decoder)
        print(model.summary())
        return model

    def train(self):
        K.clear_session()
        model = self._build_model()

        model.compile()

        earlystop = EarlyStopping()
        tensorboard = TensorBoard()
        check_reconstruction = CheckReconstruction()
        callbacks = [earlystop, tensorboard, check_reconstruction]

        model.fit()

        # Save full model
        model.save()

        # Save encoder

        # Save decoder


class LSTMVAE(VAE):
    def __init__(self):
        super().__init__()

    def _build_model(self):
        encoder = encoder_lstm()

        mu = Dense(self.latent_dim, name='mu')(encoder.layers[-1])
        log_sigma = Dense(self.latent_dim, name='log_sigma')(encoder.layers[-1])
        z = Lambda(sampling)([mu, log_sigma])

        decoder = decoder_conv(z)
        model = Model(encoder, decoder)
        print(model.summary())
        return model

    def train(self):
        K.clear_session()
        model = self._build_model()

        model.compile()

        earlystop = EarlyStopping()
        tensorboard = TensorBoard()
        check_reconstruction = CheckReconstruction()
        callbacks = [earlystop, tensorboard, check_reconstruction]

        model.fit()

        # Save full model
        model.save()

        # Save encoder

        # Save decoder