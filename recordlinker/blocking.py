'''Tools for encoding and blocking'''

from __future__ import print_function
from __future__ import absolute_import

from collections import defaultdict
import itertools
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
from keras.models import load_model
from pyjarowinkler import distance

from . import preprocess
from . import metrics


class BinaryEncoder():
    def __init__(self,
                 model_path,
                 embed_type='letters'):
        assert embed_type in ['letters', 'shingles', 'l', 's']
        self.model_path = model_path
        self.encoder = None
        self.median_mu = None
        self.embed_type = embed_type
        self.encoder = load_model(self.model_path)
        self.input_dim = self.encoder.layers[0].output_shape
        print('Loaded Model with input shape {}'.format(self.input_dim))

    def embed(self, data):
        if len(self.input_dim) == 3:
            embedded_data = preprocess.embed(data,
                                             max_length=self.input_dim[1],
                                             embed_type='letters',
                                             normalize=False,
                                             categorical=True)
        else:
            embedded_data = preprocess.embed(data,
                                             max_length=self.input_dim[1],
                                             embed_type=self.embed_type,
                                             normalize=True,
                                             categorical=False)
        return embedded_data

    def calculate_median_mu(self, train_data):
        if not isinstance(train_data, np.ndarray):
            train_data = self.embed(train_data)
        try:
            mu = self.encoder.predict(train_data)
        except ValueError:
            print('Input must have feature dimension size {}!'.format(
                self.input_dim))
            return
        self.median_mu = np.median(mu, axis=0)
        print(
            'Median mu has been set with size {}'.format(self.median_mu.shape))

    def encode(self, new_data, split=False):
        if not isinstance(new_data, np.ndarray):
            new_data = self.embed(new_data)

        if self.median_mu is None:
            print('median_mu must be calculated first')
        self.new_data_mu = self.encoder.predict(new_data)
        assert self.new_data_mu.shape[1] == self.median_mu.shape[0]
        encoded = (self.new_data_mu > self.median_mu) * 1.
        if split:
            encoded = np.split(encoded, encoded.shape[0], axis=0)
        return encoded

    def calculate_and_encode(self, train_data, split=False):
        self.calculate_median_mu(train_data)
        return self.encode(train_data, split=split)


class Blocker():
    def __init__(self, dfA, dfB):
        self.dfA = dfA
        self.dfB = dfB
        self.blocks = None
        self.num_blocks = 0
        self.block_sizes = []
        self.encoder = None
        self.block_sizes = []

    @staticmethod
    def stringify(x):
        '''Convert vector to string'''
        return ''.join([str(i) for i in x.astype(int).reshape(-1)])

    def _block_autoencoder(self,
                           autoencoder_model_path,
                           autoencoder_col,
                           autoencoder_colB=None,
                           autoencoder_embed_type='letters'):
        start_time = time.time()
        if autoencoder_colB is None:
            autoencoder_colB = autoencoder_col

        assert all(isinstance(name, str) for name in self.dfA[autoencoder_col])
        assert all(isinstance(name, str) for name in self.dfB[autoencoder_colB])
        self.encoder = BinaryEncoder(autoencoder_model_path)

        train_encoded = self.encoder.calculate_and_encode(self.dfA[autoencoder_col],
                                                          split=True)
        match_encoded = self.encoder.encode(self.dfB[autoencoder_colB],
                                            split=True)
        unique_blocks = [self.stringify(vec) for vec in
                         np.unique(train_encoded, axis=0)]
        blocks = defaultdict(dict)
        for k in unique_blocks:
            blocks[k] = {'A': [], 'B': []}

        for i, vec in enumerate(train_encoded):
            key = self.stringify(vec)
            blocks[key]['A'].append(i)

        for i, vec in enumerate(match_encoded):
            key = self.stringify(vec)
            if key in blocks.keys():
                blocks[key]['B'].append(i)

        print('Finished blocking with autoencoder in {:.4f} s'.format(time.time()-start_time))
        return blocks

    # TODO: currently only available for years - make more flexible
    def _block_timeframe(self,
                         blocks,
                         timeframe_col,
                         timeframe_colB=None,
                         timeframe_range=0,
                         timeframe_units='y'):
        start_time = time.time()
        if timeframe_colB is None:
            timeframe_colB = timeframe_col

        if len(blocks.keys()) == 1:
            block_keys = set(self.dfA[timeframe_col])
            blocks = dict.fromkeys(block_keys)
            for k, v in blocks.items():
                blocks[k] = {'A': [], 'B': []}

            for i, year in enumerate(self.dfA[timeframe_col]):
                blocks[year]['A'].append(i)

            for i, year in enumerate(self.dfB[timeframe_colB]):

                time_range = list(range(year - timeframe_range,
                                       year + timeframe_range + 1))
                matches = [x for x in time_range if x in blocks.keys()]
                if len(matches) > 0:
                    for match in matches:
                        blocks[match]['B'].append(i)
            print('Finished blocking on timerange in {:4f} s'.format(time.time()-start_time))
            return blocks
        else:
            enc_keys = list(blocks.keys())
            for enc_key in enc_keys:
                timekeys_A = set(self.dfA.iloc[blocks[enc_key]['A']][timeframe_col])
                for timekey in timekeys_A:
                    new_key = '-'.join([enc_key, str(timekey)])
                    time_range = list(range(timekey-timeframe_range,
                                           timekey+timeframe_range+1))
                    itemsA = [item for item in blocks[enc_key]['A'] if
                              self.dfA[timeframe_col].iloc[item] == timekey]
                    itemsB = [item for item in blocks[enc_key]['B'] if
                              self.dfB[timeframe_colB].iloc[item] in time_range]
                    blocks[new_key] = {
                        'A': itemsA,
                        'B': itemsB
                    }
                blocks.pop(enc_key, None)
            print('Finished blocking on timerange in {:4f} s'.format(time.time()-start_time))
            return blocks

    def _block_exact(self,
                     blocks,
                     exact_col,
                     exact_colB=None):
        start_time = time.time()
        if exact_colB is None:
            exact_colB = exact_col
        assert type(self.dfA[exact_col]) == type(self.dfB[exact_colB])

        if len(blocks.keys()) == 1:
            block_keys = list(set(self.dfA[exact_col]))
            blocks = dict.fromkeys(block_keys)

            for k, v in blocks.items():
                blocks[k] = {'A': [], 'B': []}

            for i, item in enumerate(self.dfA[exact_col]):
                blocks[item]['A'].append(i)

            for i, item in enumerate(self.dfB[exact_colB]):
                if item in blocks.keys():
                    blocks[item]['B'].append(i)
            return blocks
        else:
            enc_keys = list(blocks.keys())
            for enc_key in enc_keys:
                unique_keysA = set(self.dfA.iloc[blocks[enc_key]['A']][exact_col])
                for unique_key in unique_keysA:
                    new_key = '-'.join([enc_key, str(unique_key)])
                    itemsA = [item for item in blocks[enc_key]['A'] if
                              self.dfA[exact_col].iloc[item] == unique_key]
                    itemsB = [item for item in blocks[enc_key]['B'] if
                              self.dfB[exact_colB].iloc[item] == unique_key]
                    blocks[new_key] = {
                        'A': itemsA,
                        'B': itemsB
                    }
                blocks.pop(enc_key, None)
            print('Finished blocking exact column in {.:4f} s'.format(time.time()-start_time))
            return blocks

    # TODO: Allow to same type of blocking using multiple columns
    def block(self,
              autoencoder_col=None,
              autoencoder_colB=None,
              autoencoder_model_path=None,
              autoencoder_embedtype='letters',
              timeframe_col=None,
              timeframe_colB=None,
              timeframe_range=0,
              exact_col=None,
              exact_colB=None):
        self.blocks = defaultdict(dict)
        self.blocks['NoBlocking'] = {'A': list(self.dfA.index),
                                     'B': list(self.dfB.index)}

        if exact_col is None and autoencoder_col is None and timeframe_col is None:
            print('Warning: No blocking columns selected. '
                  'Computing full cartesian product. '
                  'This will likely be inefficient for block metric '
                  'computation and linkage.')

        if autoencoder_col is not None:
            self.blocks = self._block_autoencoder(autoencoder_model_path,
                                                  autoencoder_col,
                                                  autoencoder_colB,
                                                  autoencoder_embedtype)

        if timeframe_col is not None:
            self.blocks = self._block_timeframe(self.blocks,
                                                timeframe_col,
                                                timeframe_colB,
                                                timeframe_range)

        if exact_col is not None:
            self.blocks = self._block_exact(self.blocks,
                                            exact_col,
                                            exact_colB)

        # Drop blocks that only contain values in 'A'
        for k, v in list(self.blocks.items()):
            if len(v['B']) == 0:
                self.blocks.pop(k, None)
        self.num_blocks = len(self.blocks.keys())
        return self.blocks

    def compute_block_metrics(self,
                              match_indexA=None,
                              match_indexB=None):
        self.original_comparisons = len(self.dfA.index) * len(self.dfB.index)

        print('Num Blocks:', self.num_blocks)

        block_sizes = []
        for k, v in self.blocks.items():
            self.block_sizes.append(len(v['A']) * len(v['B']))

        self.max_block_size = np.max(self.block_sizes)
        self.min_block_size = np.min(self.block_sizes)
        self.total_pairs = np.sum(self.block_sizes)

        print('Original Comparisons Needed: {:,}'.format(self.original_comparisons))

        comparisons_needed = np.sum(self.block_sizes)
        print('Total Comparisons {:,} : {:.2f}% of original'.format(
            comparisons_needed,
            100 * comparisons_needed / self.original_comparisons))

        print('Avg Block Size: {:,.2f}'.format(np.mean(self.block_sizes)))
        print('Max Block Size: {:,}'.format(self.max_block_size))
        print('Min Block Size: {:,}'.format(self.min_block_size))

        pct_in_largest_block = self.max_block_size / self.total_pairs
        pct_in_smallest_block = self.min_block_size / self.total_pairs

        print('Balance Score (1=even sizes): {:2f}'.format(
            pct_in_smallest_block / pct_in_largest_block))

        if match_indexA is not None:
            assert len(match_indexA) == len(
                match_indexB), 'match indices must be same length'

            total_matches = len(match_indexA)
            matches = list(zip(match_indexA, match_indexB))

            matches_in_same_block = 0
            blocks_with_matches = 0

            for k, v in self.blocks.items():
                found = [item for item in matches if
                         item[0] in v['A'] and item[1] in v['B']]
                matches = [match for match in matches if match not in found]
                matches_in_same_block += len(found)
                if len(found) > 0:
                    blocks_with_matches += 1

            # Within block % matches
            print('Num Matches Found {} Out Of {} ({:.2f}%)'.format(matches_in_same_block,
                                                                    total_matches,
                                                                    100 * matches_in_same_block / total_matches))
            print('Num blocks containing matches {}, ({:.2f}%)'.format(blocks_with_matches,
                                                                       100 * blocks_with_matches / self.num_blocks))


class Linker():
    def __init__(self,
                 blocker, cols):
        self.blocker = blocker
        self.cols = cols
        self.encoder = blocker.encoder
        self.dfA = blocker.dfA
        self.dfB = blocker.dfB
        self.cpu_count = mp.cpu_count()

    @staticmethod
    def _get_pairs(args):
        k, v = args
        pairs = list(itertools.product(v['A'], v['B']))
        return pd.DataFrame(pairs, columns=['indexA', 'indexB'])

    def _possible_pairs(self):
        p = mp.Pool(processes=mp.cpu_count())
        results = p.map(self._get_pairs, self.blocker.blocks.items())
        p.close()
        return pd.concat(results, ignore_index=True)

    @staticmethod
    def jaro_winkler(arg):
        return [distance.get_jaro_distance(a, b) for (a, b) in arg]

    def enc_dist(self, args):
        a, b = args
        vecA = self.encoder.encode(
            self.blocker.preprocess(self.dfA.iloc[a], self.cols[0]))
        vecB = self.encoder.encode(
            self.blocker.preprocess(self.dfB.iloc[b], self.cols[1]))
        return metrics.normalized_l1(vecA, vecB)

    @staticmethod
    def autoencoder_dist(args):
        return np.reshape([Linker.enc_dist(arg) for arg in args], -1)

    def compare(self,
                jaro=True,
                jaro_thres=None,
                autoencoder=True,
                autoencoder_thres=None):
        start_time = time.time()
        self.comparisons = self._possible_pairs()
        print('Checkpoint {:1f}'.format(time.time() - start_time))

        A = self.dfA[self.cols[0]].iloc[self.comparisons['indexA']]
        B = self.dfB[self.cols[1]].iloc[self.comparisons['indexB']]

        print('Checkpoint {:1f}'.format(time.time() - start_time))

        if jaro:
            p = mp.Pool(processes=self.cpu_count())
            items = list(zip(A, B))
            batch_size = len(items) // 8
            items = [items[i:i + batch_size] for i in
                     range(0, len(items), batch_size)]
            results = p.map(self.jaro_winkler, items)
            self.comparisons['jaro'] = pd.Series(
                [item for result in results for item in result])
            p.close()
            print('Checkpoint {:1f}'.format(time.time() - start_time))
            if jaro_thres:
                self.comparisons['jaro'] = self.comparisons[
                                               'jaro'] >= jaro_thres
        if autoencoder:
            p = mp.Pool(processes=self.cpu_count())
            A = names_1915['lname1915'].iloc[self.comparisons['indexA']]
            B = names_1940['lname1940'].iloc[comparisons['indexB']]
            items = list(zip(comparisons['indexA'], comparisons['indexB']))
            batch_size = len(items) // 8
            items = [items[i:i + batch_size] for i in
                     range(0, len(items), batch_size)]
            results = p.map(autoencoder_dist, items)
            if autoencoder_thres:
                self.comparisons['autoencoder'] = self.comparisons[
                                                      'autoencoder'] >= autoencoder_thres

        return self.comparisons

    def fit(self):
        pass

    def predict(self):
        pass
