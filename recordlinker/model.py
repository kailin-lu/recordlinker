'''Variational Autoencoder Models'''

from __future__ import print_function
from __future__ import absolute_import

import keras.backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras import Model
from keras.layers import Input, Dense, Lambda, LSTM, Conv2D
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

# Custom Callbacks
class CheckReconstruction(K.callbacks.Callback):
    '''Qualitative check of name reconstruction'''
    def __init__(self, type):
        assert type in ['shingle', 'letter'], "Type must be 'shingle' or 'letter'"
        pass

    def on_batch_end(self, batch, logs):
        pass

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

# Helpers
class _BinaryEncoder():
    def __init__(self):
        pass

    def calculate_median_mu(self):
        pass

    def binary_encode(self):
        pass


# Encoders
def encoder_dense():
    pass

def encoder_conv():
    pass

def encoder_lstm():
    pass

def encoder_binary(encoder_func):
    pass

# Decoders
def decoder_dense():
    pass

def decoder_conv():
    pass

def decoder_lstm():
    pass


# End to end models
class VAE():
    def __init__(self):
        pass

    def _build_model(self):
        pass

    def train(self):
        pass


class ConvolutionalVAE():
    def __init__(self):
        pass

    def _build_model(self):
        pass

    def train(self):
        pass


class LSTMVAE():
    def __init__(self):
        pass

    def _build_model(self):
        pass

    def train(self):
        pass