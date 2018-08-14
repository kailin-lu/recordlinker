'''Tools for encoding and blocking'''

from __future__ import print_function
from __future__ import absolute_import

import itertools
import numpy as np
import pandas as pd

from keras.models import load_model

from . import utils

class BinaryEncoder():
    def __init__(self,
                 model_path):
        self.model_path = model_path
        self.encoder = None
        self.median_mu = None
        self.encoder = load_model(self.model_path)
        self.input_dim = self.encoder.layers[0].output_shape

    def calculate_median_mu(self, train_data):
        try:
            mu = self.encoder.predict(train_data)

        # TODO: Check input dimension warning
        except ValueError:
            print('Input must have feature dimension size {}!'.format(self.input_dim))
            return
        self.median_mu = np.median(mu, axis=0)
        print('Median mu has been set with size {}'.format(self.median_mu.shape))

    def binary_encode(self, new_data, split=False):
        if self.median_mu is None:
            print('median_mu must be calculated first')
        new_data_mu = self.encoder.predict(new_data)
        assert new_data_mu.shape[1] == self.median_mu.shape[0]
        encoded = (new_data_mu > self.median_mu) * 1.
        if split:
            encoded = np.split(encoded, encoded.shape[0], axis=0)
        return encoded

    def calculate_and_encode(self, train_data, split=False):
        self.calculate_median_mu(train_data)
        return self.binary_encode(train_data, split=split)

class Blocker():
    def __init__(self, dfA, dfB):
        self.dfA = dfA
        self.dfB = dfB
        self.num_blocks = None
        self.block_sizes = None

    @staticmethod
    def stringify(x):
        return ''.join([str(i) for i in x.astype(int).reshape(-1)])

    def _block_autoencoder(self,
                           autoencoder_model_path,
                           autoencoder_col,
                           autoencoder_colB=None,
                           embed_type='letters'):
        if autoencoder_colB is None:
            autoencoder_colB = autoencoder_col

        assert all(isinstance(name, str) for name in self.dfA[autoencoder_col])
        assert all(isinstance(name, str) for name in self.dfB[autoencoder_colB])

        encoder = BinaryEncoder(autoencoder_model_path)
        input_dim = encoder.input_dim
        if len(input_dim) == 3:
            train_data = utils.create_training_set(self.dfA,
                                                   autoencoder_col,
                                                   max_length=input_dim[1],
                                                   embed_type='letters',
                                                   normalize=False,
                                                   categorical=True)
            match_data = utils.create_training_set(self.dfB,
                                                   autoencoder_colB,
                                                   max_length=input_dim[1],
                                                   embed_type='letters',
                                                   normalize=False,
                                                   categorical=True)
        else:
            train_data = utils.create_training_set(self.dfA,
                                                   autoencoder_col,
                                                   max_length=input_dim[1],
                                                   embed_type=embed_type,
                                                   normalize=True,
                                                   categorical=False)
            match_data = utils.create_training_set(self.dfB,
                                                   autoencoder_colB,
                                                   max_length=input_dim[1],
                                                   embed_type=embed_type,
                                                   normalize=True,
                                                   categorical=False)

        train_encoded = encoder.calculate_and_encode(train_data, split=True)
        match_encoded = encoder.binary_encode(match_data, split=True)

        unique_blocks =  [self.stringify(vec) for vec in np.unique(train_encoded, axis=0)]
        blocks = dict.fromkeys(unique_blocks)
        for k, v in blocks.items():
            blocks[k] = {'A':[], 'B':[]}

        for i, vec in enumerate(train_encoded):
            key = self.stringify(vec)
            blocks[key]['A'].append(i)

        for i, vec in enumerate(match_encoded):
            key = self.stringify(vec)
            if key in blocks.keys():
                blocks[key]['B'].append(i)
        return blocks

    def _block_timeframe(self):
        pass

    def _block_exact(self):
        pass


    # TODO: Allow to same type of blocking on multiple types of columns
    def generate_blocks(self,
                        autoencoder_col=None,
                        autoencoder_model_path=None,
                        autoencoder_embedtype='letters',
                        timeframe_col=None,
                        timeframe_range=None,
                        exact_col=None):
        product = itertools.product(self.dfA.index, self.dfB.index)
        self.index_df = pd.DataFrame([(i[0], i[1]) for i in product],
                                columns=['indexA', 'indexB'])

        if exact_col is None and autoencoder_col is None and timeframe_col is None:
            print('Warning: No blocking columns selected. '
                  'Computing full cartesian product.'
                  'This will likely be inefficient for linkage.')

        if autoencoder_col is not None:
            self.blocks = self._block_autoencoder(autoencoder_model_path,
                                                  autoencoder_col,
                                                  embed_type=autoencoder_embedtype)

        if timeframe_col is not None:
            self.blocks= self._block_timeframe()

        if exact_col is not None:
            self.blocks = self._block_exact()

        pass

    def compute_block_metrics(self,
                              match_indexA=None,
                              match_indexB=None):
        if self.df_blocked is None:
            print('Blocks must be generated with generate_blocks first.')
            return

        if isinstance(self.df_blocked, list):
            num_blocks = len(self.df_blocked)
        else:
            num_blocks = 1
        print('Num Blocks:', num_blocks)

        print('Avg Block Size:' )
        print('Max Block Size')
        print('Min Block Size')

        if match_indexA is not None:
            assert len(match_indexA) == len(match_indexB), 'match indices must be same length'
            total_matches = len(match_indexA)

            # Within block % matches
            print('% Matches In Same Block:')
        pass


########## Metrics #############

# Helper to calculate L1 distance between binary encoded sets
def normalized_l1(A, B):
    '''Normalized distance between binary encoded vector A
       and binary encoded vector B'''
    assert A.shape == B.shape
    diff = np.sum(abs(A-B), axis=1)
    return 1 - (diff / A.shape[1])
