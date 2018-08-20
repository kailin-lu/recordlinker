from unittest import TestCase

from recordlinker.model import VAE, ConvolutionalVAE, LSTMVAE

class TestDenseEncoder(TestCase):
    def test_create_dense_encoder(self):
        vae = VAE()
        pass

class TestConvEncoder(TestCase):
    def test_create_conv_encoder(self):
        conv = ConvolutionalVAE()
        pass

class TestLSTMEncoder(TestCase):
    def test_create_lstm_encoder(self):
        lstm = LSTMVAE()
        pass

