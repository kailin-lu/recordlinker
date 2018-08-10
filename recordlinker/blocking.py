'''Tools for encoding and blocking'''

from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
import numpy as np

import keras
from keras.models import load_model


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


# Return grouped dataframe with pairs to be checked

def block(A, B,
          vae_col,
          date_col,
          timeframe,
          numerical,
          num_range):
    pass

