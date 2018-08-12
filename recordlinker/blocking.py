'''Tools for encoding and blocking'''

from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from keras.models import load_model

class BinaryEncoder():
    def __init__(self, model_path):
        self.model_path = model_path
        self.input_dim = None
        self.median_mu = None
        self.encoder = None

    def _calculate_median_mu(self, train_data):
        self.encoder = load_model(self.model_path)
        self.input_dim = self.encoder.layers[0].output_shape[1]
        try:
            mu = self.encoder.predict(train_data)
        except ValueError:
            print('Input must have feature dimension size {}!'.format(self.input_dim))
            return
        self.median_mu = np.median(mu, axis=0)
        print('Median mu has been set with size {}'.format(self.median_mu.shape))

    def binary_encode(self, new_data):
        if self.median_mu is None:
            print('median_mu must be calculated first')
        new_data_mu = self.encoder.predict(new_data)
        assert new_data_mu.shape[1] == self.median_mu.shape[0]
        return (new_data_mu > self.median_mu) * 1.

# Return grouped dataframe with pairs to be checked
def block(A, B,
          vae_col,
          date_col,
          timeframe,
          numerical,
          num_range):
    pass

