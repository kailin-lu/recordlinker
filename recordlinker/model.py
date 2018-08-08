'''Variational autoencoder models and helpers'''

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import warnings

import keras
import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras import Model
from keras.layers import Input, Dense, Lambda, Conv1D, MaxPool1D,\
    UpSampling1D, LSTM, RepeatVector, Flatten, Reshape
from keras.optimizers import RMSprop, Adam
from keras.losses import categorical_crossentropy

from .preprocess import disembed_letters, disembed_shingles

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
                 random=False):
        """

        :param type: 'letter' or 'shingle'
        :param n: number of name-reconstruction pairs to print
        :param random: display random n if True else display first n in batch
        """
        super().__init__()
        if type == 'letter':
            self.disembed_func = disembed_letters
        elif type == 'shingle':
            self.disembed_func = disembed_shingles
        else:
            warnings.warn('Type must be "letter" or "shingle"')
        self.n = n
        self.random = random
        self.names_to_reconstruct = None

    def on_epoch_end(self, logs={}):
        pred = self.model.predict(self.model.inputs)


    # def on_epoch_end(self, epoch, logs={}):
    #     pred = self.model.predict(self.names_to_reconstruct)
    #     print('Reconstruction Check:')
    #     for i in range(self.n):
    #         print(self.disembed_func(self.names_to_reconstruct[i,:]))
    #         print(self.disembed_func(pred[i,:]))

# Helper
class BinaryEncoder():
    def __init__(self):
        self.median_mu = None

    def calculate_median_mu(self, model_path, train_data):
        encoder = keras.load_model(model_path)
        mu = encoder.predict(train_data)
        self.median_mu = np.median(mu, axis=1)

    def binary_encode(self):
        if self.median_mu == None:
            print('Median mu must be calculated first')
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


def encoder_conv(enc_input,
                 activation='relu',
                 filters=[16, 32]):
    '''Encoder with convolutional layers'''
    encode_layers = [enc_input]
    encode_layers.append(Conv1D(filters=filters[0],
                                kernel_size=2,
                                activation=activation,
                                padding='valid',
                                name='conv0')(encode_layers[-1]))
    encode_layers.append(MaxPool1D(name='pool0',
                                   pool_size=2,
                                   strides=1,
                                   padding='valid')(encode_layers[-1]))
    encode_layers.append(Conv1D(filters=filters[1],
                                kernel_size=2,
                                activation=activation,
                                padding='valid',
                                name='conv1')(encode_layers[-1]))
    encode_layers.append(MaxPool1D(name='pool1',
                                   pool_size=2,
                                   strides=1,
                                   padding='valid')(encode_layers[-1]))
    return encode_layers


def encoder_lstm(enc_inputs,
                 latent_dim,
                 encode_dim):
    encode_layers = [enc_inputs]
    for i, units in enumerate(encode_dim):
        encode_layers.append(LSTM(units)(encode_layers[-1]))
    encode_layers.append(LSTM(latent_dim)(encode_layers-1))
    return encode_layers


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


def decoder_conv(z,
                 orig_units,
                 filters=[32, 1],
                 activation='relu'):
    '''Convolutional decoder with two conv-upsample layers'''
    decode_layers = [z]
    decode_layers.append(Conv1D(kernel_size=2,
                                padding='valid',
                                filters=filters[0],
                                activation=activation,
                                name='dec_conv0')(decode_layers[-1]))
    decode_layers.append(UpSampling1D(size=1,
                                      name='upsample0')(decode_layers[-1]))
    decode_layers.append(Conv1D(kernel_size=2,
                                padding='valid',
                                filters=filters[1],
                                activation=activation,
                                name='dec_conv1')(decode_layers[-1]))
    decode_layers.append(UpSampling1D(size=2,
                                      name='upsample1')(decode_layers[-1]))
    decode_layers.append(Reshape((int(decode_layers[-1].shape[1]),))(decode_layers[-1]))
    decode_layers.append(Dense(orig_units,
                               activation='sigmoid',
                               name='reconstruction')(decode_layers[-1]))
    decode_layers.append(Reshape((int(decode_layers[-1].shape[1]),1))(decode_layers[-1]))
    return decode_layers


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
              save_path,
              optimizer='adam',
              validation_split=.2,
              earlystop=True,
              tensorboard=True,
              reconstruct=True):
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
            check_recon = CheckReconstruction(type='letter',
                                              n=5,
                                              random=False)
            callbacks.append(check_recon)
        model.fit(namesA, namesB,
                  shuffle=True,
                  epochs=epochs,
                  batch_size=self.batch_size,
                  validation_split=validation_split,
                  callbacks=callbacks)

        # Save full model
        model.save(save_path + '.h5')
        # Save enccoder
        encoder.save(save_path + '_encoder.h5')
        # Save decoder
        decoder.save(save_path + '_decoder.h5')


class ConvolutionalVAE(VAE):
    def __init__(self, batch_size, orig_dim, latent_dim):
        super().__init__(batch_size=batch_size,
                         orig_dim=orig_dim,
                         latent_dim=latent_dim,
                         encode_dim=2,
                         decode_dim=2,
                         lr=.001)

    def _build_model(self):
        K.clear_session()

        enc_inp = Input(batch_shape=(self.batch_size, self.orig_dim, 1))
        encoder_layers = encoder_conv(enc_inp, activation=self.activation)

        flatten_size = int(encoder_layers[-1].shape[1]) * int(encoder_layers[-1].shape[2])
        flatten = Reshape((flatten_size,), name='flatten')(encoder_layers[-1])

        mu = Dense(self.latent_dim, name='mu')(flatten)
        log_sigma = Dense(self.latent_dim, name='log_sigma')(flatten)
        z = Lambda(sampling)([mu, log_sigma])
        z_add_dim = Reshape((int(z.shape[1]), 1))(z)

        decoder_layers = decoder_conv(z_add_dim, self.orig_dim)
        model = Model(enc_inp, decoder_layers[-1])
        encoder = Model(enc_inp, mu)

        dec_inp = Input(batch_shape=(self.batch_size, self.latent_dim, 1))
        dec_layers = decoder_conv(dec_inp, self.orig_dim)
        decoder = Model(dec_inp, dec_layers[-1])
        print(model.summary())
        return model, encoder, decoder

    def train(self,
              namesA,
              namesB,
              epochs,
              run_id,
              save_path,
              optimizer='adam',
              validation_split=.2,
              earlystop=True,
              tensorboard=True,
              reconstruct=True):
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
    def __init__(self,
                 timesteps,
                 orig_dim,
                 lr):
        super().__init__()
        self.timesteps = timesteps
        self.orig_dim = orig_dim

    def _build_model(self):
        encoder_input = Input(shape=(self.timesteps, self.orig_dim))
        encoder = encoder_lstm(encoder_input)

        mu = Dense(self.latent_dim, name='mu')(encoder.layers[-1])
        log_sigma = Dense(self.latent_dim, name='log_sigma')(encoder.layers[-1])
        z = Lambda(sampling)([mu, log_sigma])
        decoder_layers = decoder_lstm(z)
        model = Model(encoder_input, decoder_layers[-1])

        encoder = Model(encoder_input, mu)

        decoder_input = Input(shape=(self.timesteps, self.orig_dim))
        decoder_layers = decoder_lstm(decoder_input)
        decoder = Model(decoder_input, decoder_layers[-1])

        print(model.summary())
        return model, encoder, decoder