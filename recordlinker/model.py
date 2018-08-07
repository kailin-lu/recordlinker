'''Variational autoencoder models and helpers'''

from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import keras
import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras import Model
from keras.layers import Input, Dense, Lambda, Conv1D, MaxPool1D, UpSampling1D, LSTM, RepeatVector
from keras.optimizers import RMSprop, Adam
from keras.losses import categorical_crossentropy

# Variational autoencoder sampling
def sampling(args):
    '''Reparameterization trick to calculate z'''
    mu, log_sigma = args
    batch_shape = mu.shape
    epsilon = K.random_normal(shape=batch_shape, mean=0., stddev=1.)
    return mu + (.5 * log_sigma) * epsilon

# Custom callback
class CheckReconstruction(keras.callbacks.Callback):
    '''Qualitative check of name reconstruction'''
    def __init__(self,
                 type='letter',
                 n=5,
                 random=False,
                 validation=True):
        """

        :param type: 'letter' or 'shingle'
        :param n: number of name-reconstruction pairs to print
        :param random: display random n if True else display first n in batch
        """
        super().__init__()
        self.type = type
        self.n = n
        self.random = random
        self.validation = validation
        self.names = None
        self.names_to_reconstruct = None

    def on_train_begin(self, logs):
        if self.validation:
            self.names = self.model.validation_data
        else:
            self.names = self.model.training_data
        if self.random:
            # Pick n random rows from validation data
            rows = np.random.randint(0, self.names.shape[0], self.n)
            self.names_to_reconstruct = self.names[rows,:]
        else:
            # Take first n names from data
            self.names_to_reconstruct = self.names[:self.n, :]

    def on_epoch_end(self):
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
def encoder_dense(enc_input,
                  batch_size,
                  orig_dim,
                  encode_dim,
                  latent_dim,
                  activation='relu'):
    '''Dense encoder with an arbitrary number of hidden layers'''
    if isinstance(encode_dim, int):
        encode_dim = [encode_dim]

    assert isinstance(encode_dim, list), 'encode_dim must be int or list of ints'

    encode_layers = [enc_input]
    for i,units in enumerate(encode_dim):
        layer = Dense(units,
                      activation=activation,
                      name='enc_{}'.format(i))(encode_layers[-1])
        encode_layers.append(layer)
    return encode_layers


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

    if isinstance(decode_dim, int):
        decode_dim = [decode_dim]

    assert isinstance(decode_dim, list), 'decode_dim must be int or list of ints'
    decode_layers = [dec_input]
    for i, units in enumerate(decode_dim):
        layer = Dense(units,
                      activation=activation,
                      name='dec_{}'.format(i))(decode_layers[-1])
        decode_layers.append(layer)
    decode_layers.append(Dense(orig_dim,
                               activation='sigmoid',
                               name='reconstruction')(decode_layers[-1]))
    return decode_layers

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
        K.clear_session()
        enc_input = Input(batch_shape=(self.batch_size, self.orig_dim))
        encoder_layers = encoder_dense(enc_input,
                                       self.batch_size,
                                       self.orig_dim,
                                       self.encode_dim,
                                       self.latent_dim,
                                       self.activation)
        mu = Dense(self.latent_dim, name='mu')(encoder_layers[-1])
        log_sigma = Dense(self.latent_dim, name='log_sigma')(encoder_layers[-1])
        z = Lambda(sampling)([mu, log_sigma])
        decoder_layers = decoder_dense(z,
                                       self.orig_dim,
                                       self.decode_dim,
                                       self.activation)
        model = Model(enc_input, decoder_layers[-1])
        encoder = Model(enc_input, mu)

        dec_input = Input(batch_shape=(self.batch_size, self.latent_dim))
        decoder_model = decoder_dense(dec_input,
                                      self.orig_dim,
                                      self.decode_dim,
                                      self.activation)
        decoder = Model(dec_input, decoder_model[-1])

        print('Full Model:', model.summary())

        def _vae_loss(y_true, y_pred):
            loss = categorical_crossentropy(y_true, y_pred)
            kl_loss = - 0.5 * K.mean(
                1 + log_sigma - K.square(mu) - K.exp(log_sigma), axis=-1)
            return loss + kl_loss

        return model, encoder, decoder, _vae_loss

    def train(self,
              namesA,
              namesB,
              epochs,
              run_id,
              optimizer='adam',
              validation_split=.2,
              earlystop=True,
              tensorboard=True,
              reconstruct=True,
              reconstruct_type='letter',
              reconstruct_n=5,
              reconstruct_val=True):
        '''
        Train the dense autoencoder model

        :param namesA: column A of known matches
        :param namesB: column B of known corresponding matches
        :param epochs:
        :param run_id:
        :param optimizer: keras optimizer to compile model
        :param

        :return:
        '''
        model, encoder, decoder, vae_loss = self._build_model()

        if optimizer == 'adam':
            op = Adam(lr=self.lr)
        if optimizer == 'rmsprop':
            op = RMSprop(lr=self.lr)
        else:
            op = optimizer

        model.compile(optimizer=op,
                      loss=vae_loss,
                      metrics=['accuracy'])

        # Callbacks
        callbacks = []
        if earlystop:
            early_stop = EarlyStopping(patience=5,
                                       min_delta=.0001)
            callbacks.append(early_stop)
        if tensorboard:
            tensor_board = TensorBoard(log_dir='/tmp/' + run_id,
                                       histogram_freq=10,
                                       batch_size=self.batch_size)
            callbacks.append(tensor_board)
        if reconstruct:
            check_recon = CheckReconstruction(type='letter')
            callbacks.append(check_recon)


        model.fit(namesA, namesB,
                  shuffle=True,
                  epochs=epochs,
                  batch_size=self.batch_size,
                  validation_split=validation_split,
                  callbacks=callbacks)

        # Save full model

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
        return model, encoder, decoder

    def train(self):
        K.clear_session()
        model, encoder, decoder = self._build_model()

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
        return model, encoder, decoder

    def train(self,
              namesA,
              namesB,
              earlystop=True,
              tensorboard=True,
              check_recon=True):
        K.clear_session()
        model, encoder, decoder = self._build_model()

        model.compile()

        earlystop = EarlyStopping(monitor='val_acc', patience=5, baseline=.6)
        tensorboard = TensorBoard(log_dir='/tmp/lstm_vae')
        check_reconstruction = CheckReconstruction()
        callbacks = [earlystop, tensorboard, check_reconstruction]

        model.fit()

        # Save full model
        model.save()

        # Save encoder

        # Save decoder