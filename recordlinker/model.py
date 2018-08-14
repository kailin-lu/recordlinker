'''Variational autoencoder models and helpers'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

import numpy as np
import warnings

import keras
import keras.backend as K
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping
from keras import Model
from keras.layers import Input, Dense, Lambda, LeakyReLU
from keras.layers import Conv1D, MaxPool1D, UpSampling1D, Reshape
from keras.layers import LSTM, RepeatVector, Flatten
from keras.optimizers import RMSprop, Adam
from keras.losses import binary_crossentropy, categorical_crossentropy, mse

from .preprocess import disembed_letters, disembed_shingles

# Variational autoencoder sampling
class Sampling():
    '''Reparameterization trick to calculate z'''
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sampling(self,args):
        mu, log_sigma = args
        dim = int(mu.shape[1])
        epsilon = K.random_normal(shape=(self.batch_size, dim), mean=0., stddev=1.)
        return mu + (.5 * log_sigma) * epsilon


class VariationalLoss():
    '''Sum of loss and KL divergence '''
    def __init__(self, mu, log_sigma, loss_type='xent'):
        self.mu = mu
        self.log_sigma = log_sigma
        self.loss_type = loss_type

    def vae_loss(self, y_true, y_pred):
        if self.loss_type == 'xent':
            loss = K.sum(categorical_crossentropy(y_true, y_pred), axis=-1)
        elif self.loss_type == 'binary_xent':
            loss = K.sum(binary_crossentropy(y_true, y_pred), axis=-1)
        elif self.loss_type == 'mse':
            loss =  K.sum(mse(y_true, y_pred), axis=-1)
        else:
            warnings.warn('Loss type not recognized')
        kl_loss = - 0.5 * K.mean(
            1 + self.log_sigma - K.square(self.mu) - K.exp(self.log_sigma), axis=-1)
        return loss + kl_loss


# Custom callback
class CheckReconstruction(keras.callbacks.Callback):
    '''Qualitative check of name reconstruction'''
    def __init__(self,
                 train_data,
                 batch_size,
                 n=5,
                 type='letter',
                 display=20):
        """
        :param train_data: data to reconstruct
        :batch_size: size of batches used for training
        :param n: Number of reconstructions to display
        :param type: 'letter' or 'shingle'
        :param display: Display reconstruction after every `display` epochs
        """
        super().__init__()
        if type in ['letter', 'l']:
            self.disembed_func = disembed_letters
        elif type in ['shingle', 's']:
            self.disembed_func = disembed_shingles
        else:
            warnings.warn('Type must be "letter" or "shingle"')
        self.batch_size = batch_size
        self.train_data = train_data
        self.n = n
        self.display = display
        self.seen = 0

    def on_epoch_end(self, epoch, logs={}):
        ''' Every `display` epochs, print a random reconstruction of `n` names'''
        self.prediction = []
        random_index = np.random.randint(0, len(self.train_data), self.batch_size)
        if self.seen % self.display == 0:
            batch_data = self.train_data[random_index, :]
            pred = self.model.predict(batch_data)
            for i in range(self.n):
                orig_name = self.disembed_func(batch_data[i])
                pred_name = self.disembed_func(pred[i])
                self.prediction.append({'Orig': orig_name, 'Pred:': pred_name})
            print('Sample Reconstructions:')
            for pred in self.prediction:
                print(pred)
        self.seen += 1


############### Encoders ###################
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
        encode_layers.append(Dense(units,
                      activation=activation,
                      kernel_regularizer=regularizers.l2(.01),
                      name='enc_{}'.format(i))(encode_layers[-1]))
    return encode_layers


def encoder_conv(enc_input,
                 activation='relu',
                 kernel_size=3,
                 pool_size=3,
                 filters=[16,16]):
    '''Encoder with convolutional layers'''
    encode_layers = [enc_input]
    encode_layers.append(Conv1D(filters=filters[0],
                                kernel_size=kernel_size,
                                kernel_regularizer=regularizers.l2(.01),
                                activation=activation,
                                strides=1,
                                padding='same',
                                name='conv0')(encode_layers[-1]))
    encode_layers.append(Conv1D(filters=filters[1],
                                kernel_size=kernel_size,
                                kernel_regularizer=regularizers.l2(.01),
                                activation=activation,
                                strides=1,
                                padding='same',
                                name='conv1')(encode_layers[-1]))
    return encode_layers


def encoder_lstm(enc_inputs,
                 orig_dim,
                 latent_dim,
                 encode_dim):
    '''Encoder with LSTM layers'''
    encode_layers = [enc_inputs]
    encode_layers.append(LSTM(encode_dim,
                              return_sequences=True)(encode_layers[-1]))
    encode_layers.append(LSTM(encode_dim,
                              return_sequences=True)(encode_layers[-1]))
    return encode_layers


############### Decoders ###################
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
                      kernel_regularizer=regularizers.l2(.01),
                      name='dec_{}'.format(i))(decode_layers[-1])
        decode_layers.append(layer)
    decode_layers.append(Dense(orig_dim,
                               activation='sigmoid',
                               name='reconstruction')(decode_layers[-1]))
    return decode_layers


def decoder_conv(z,
                 orig_units,
                 kernel_size=3,
                 filter_size=[16,16],
                 upsample_size=2,
                 activation='relu'):
    '''Convolutional decoder with convolutional layers'''
    decode_layers = [z]
    decode_layers.append(Reshape((int(z.shape[1]),1))(z))
    decode_layers.append(Conv1D(kernel_size=kernel_size,
                                padding='same',
                                filters=filter_size[0],
                                kernel_regularizer=regularizers.l2(.01),
                                activation=activation,
                                name='dec_conv_0')(decode_layers[-1]))
    decode_layers.append(Conv1D(kernel_size=kernel_size,
                                padding='same',
                                filters=filter_size[1],
                                activation=activation,
                                kernel_regularizer=regularizers.l2(.01),
                                name='dec_conv_1')(decode_layers[-1]))
    decode_layers.append(Conv1D(kernel_size=kernel_size,
                                padding='same',
                                filters=1,
                                activation=activation,
                                kernel_regularizer=regularizers.l2(.01),
                                name='dec_conv_2')(decode_layers[-1]))
    decode_layers.append(Reshape((int(decode_layers[-1].shape[1]),))(decode_layers[-1]))
    decode_layers.append(Dense(orig_units,
                               activation='sigmoid',
                               name='reconstruction')(decode_layers[-1]))
    return decode_layers


def decoder_lstm(encoded,
                 timesteps,
                 orig_dim,
                 decode_dim=64):
    '''Decoder with LSTM layers and a softmax output'''
    decoder_layers = [RepeatVector(timesteps)(encoded)]
    decoder_layers.append(LSTM(decode_dim,
                               return_sequences=True)(decoder_layers[-1]))
    decoder_layers.append(LSTM(decode_dim,
                               return_sequences=True)(decoder_layers[-1]))
    decoder_layers.append(Dense(orig_dim,
                                activation='softmax')(decoder_layers[-1]))
    return decoder_layers


############### End-to-end models ###################
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
        enc_input = Input(shape=(self.orig_dim,))
        encoder_layers = encoder_dense(enc_input,
                                       self.batch_size,
                                       self.orig_dim,
                                       self.encode_dim,
                                       self.latent_dim,
                                       self.activation)
        mu = Dense(self.latent_dim, name='mu')(encoder_layers[-1])
        log_sigma = Dense(self.latent_dim, name='log_sigma')(encoder_layers[-1])

        samp = Sampling(batch_size=self.batch_size)
        z = Lambda(samp.sampling)([mu, log_sigma])
        decoder_layers = decoder_dense(z,
                                       self.orig_dim,
                                       self.decode_dim,
                                       self.activation)
        model = Model(enc_input, decoder_layers[-1])
        encoder = Model(enc_input, mu)

        dec_input = Input(shape=(self.latent_dim,))
        decoder_model = decoder_dense(dec_input,
                                      self.orig_dim,
                                      self.decode_dim,
                                      self.activation)
        decoder = Model(dec_input, decoder_model[-1])

        print('Full Model:', model.summary())
        loss = VariationalLoss(mu, log_sigma, loss_type='mse')

        return model, encoder, decoder, loss.vae_loss

    def train(self,
              namesA,
              namesB,
              epochs,
              run_id,
              save_path,
              optimizer='adam',
              validation_split=.2,
              earlystop=True,
              earlystop_patience=5,
              tensorboard=True,
              reconstruct=True,
              reconstruct_type='letter',
              reconstruct_display=20):
        '''
        Train the dense autoencoder model

        :param namesA: column A of known matches
        :param namesB: column B of known corresponding matches
        :param epochs: number of epochs to train
        :param run_id: name of run for Tensorboard
        :param optimizer: keras optimizer to compile model
        :param validation_split: percentage of data to use for validation
        :earlystop: EarlyStop is added as a callback if True
        :earlystop_patience: patience parameter for earlystop
        :reconstruct: CheckReconstruction is added as a callback if True
        :reconstruct_type: 'letter' or 'shingle'
        :reconstruct_display: display reconstruction every n epochs
        '''
        # Check if the save directory exists, create if not
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

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
                                       write_images=True,
                                       batch_size=self.batch_size)
            callbacks.append(tensor_board)
        if reconstruct:
            check_recon = CheckReconstruction(train_data=namesA,
                                              batch_size=self.batch_size,
                                              type=reconstruct_type,
                                              display=reconstruct_display,
                                              n=5)
            callbacks.append(check_recon)

        # Fit the model
        model.fit(namesA, namesB,
                  shuffle=True,
                  epochs=epochs,
                  batch_size=self.batch_size,
                  validation_split=validation_split,
                  callbacks=callbacks)

        # Save full model
        model.save(save_path + 'model.h5')
        print('Saved model in: {}'.format(save_path + 'model.h5'))

        # Save encoder
        encoder.save(save_path + '/encoder.h5')
        print('Saved encoder in: {}'.format(save_path + 'encoder.h5'))

        #Save decoder
        decoder.save(save_path + '/decoder.h5')
        print('Saved decoder in: {}'.format(save_path + 'decoder.h5'))

        return model, encoder, decoder


class ConvolutionalVAE(VAE):
    def __init__(self, batch_size, orig_dim, latent_dim, lr):
        super().__init__(batch_size=batch_size,
                         orig_dim=orig_dim,
                         latent_dim=latent_dim,
                         encode_dim=None,
                         decode_dim=None,
                         lr=lr)

    def _build_model(self):
        K.clear_session()

        enc_inp = Input(batch_shape=(None, self.orig_dim))
        enc_add_dim = Reshape((self.orig_dim, 1))(enc_inp)
        encoder_layers = encoder_conv(enc_add_dim, activation=self.activation)

        flatten = Flatten(name='flatten')(encoder_layers[-1])

        mu = Dense(self.latent_dim, name='mu')(flatten)
        mu = LeakyReLU(.02)(mu)
        log_sigma = Dense(self.latent_dim, name='log_sigma')(flatten)
        log_sigma = LeakyReLU(.02)(log_sigma)

        samp = Sampling(batch_size=self.batch_size)
        z = Lambda(samp.sampling, name='z')([mu, log_sigma])

        decoder_layers = decoder_conv(z, self.orig_dim)
        model = Model(enc_inp, decoder_layers[-1])
        encoder = Model(enc_inp, mu)

        dec_inp = Input(shape=(None, self.latent_dim))
        dec_layers = decoder_conv(dec_inp, self.orig_dim)
        decoder = Model(dec_inp, dec_layers[-1])
        print(model.summary())

        conv_loss = VariationalLoss(mu, log_sigma, loss_type='xent')
        return model, encoder, decoder, conv_loss.vae_loss


class LSTMVAE(VAE):
    def __init__(self,
                 batch_size,
                 timesteps,
                 orig_dim,
                 latent_dim,
                 encode_dim,
                 decode_dim,
                 lr):
        super().__init__(batch_size=batch_size,
                         orig_dim=orig_dim,
                         latent_dim=latent_dim,
                         encode_dim=encode_dim,
                         decode_dim=decode_dim,
                         lr=lr)
        self.timesteps = timesteps

    def _build_model(self):
        K.clear_session()
        encoder_input = Input(shape=(self.timesteps,self.orig_dim))
        encoder_layers = encoder_lstm(encoder_input,
                                      orig_dim=self.timesteps,
                                      encode_dim=self.encode_dim,
                                      latent_dim=self.latent_dim)
        flatten = Flatten()(encoder_layers[-1])

        mu = Dense(self.latent_dim,
                   kernel_regularizer=regularizers.l2(.005),
                   name='mu')(flatten)
        mu = LeakyReLU(.02)(mu)
        log_sigma = Dense(self.latent_dim,
                          kernel_regularizer=regularizers.l2(.005),
                          name='log_sigma')(flatten)
        log_sigma = LeakyReLU(.02)(log_sigma)

        samp = Sampling(self.batch_size)
        z = Lambda(samp.sampling, name='z')([mu, log_sigma])

        decoder_layers = decoder_lstm(z,
                                      decode_dim=self.decode_dim,
                                      orig_dim=self.orig_dim,
                                      timesteps=self.timesteps)

        model = Model(encoder_input, decoder_layers[-1])
        encoder = Model(encoder_input, mu)

        decoder_input = Input(shape=(self.latent_dim,))
        dec_layers = decoder_lstm(decoder_input,
                                  decode_dim=self.decode_dim,
                                  orig_dim=self.orig_dim,
                                  timesteps=self.timesteps)
        decoder = Model(decoder_input, dec_layers[-1])

        print(model.summary())

        lstm_loss = VariationalLoss(mu=mu, log_sigma=log_sigma, loss_type='xent')
        return model, encoder, decoder, lstm_loss.vae_loss