from unittest import TestCase

import os
import keras

from recordlinker.blocking import BinaryEncoder, Blocker, Linker

CUR_DIR = os.path.dirname(__file__)
TEST_MODEL_PATH = os.path.join(CUR_DIR, 'test_encoder.h5')


class TestBinaryEncoder(TestCase):
    def test_binary_encoder_create(self):
        encoder = BinaryEncoder(model_path=TEST_MODEL_PATH,
                                embed_type='l')
        self.assertTrue(isinstance(encoder, BinaryEncoder))

    def test_binary_encoder_load_model(self):
        encoder = BinaryEncoder(model_path=TEST_MODEL_PATH,
                                embed_type='l')
        self.assertTrue(isinstance(encoder.encoder, keras.engine.training.Model))

    def test_binary_encoder_input_dim(self):
        encoder = BinaryEncoder(model_path=TEST_MODEL_PATH,
                                embed_type='l')
        self.assertEqual(encoder.input_dim, (None, 12))