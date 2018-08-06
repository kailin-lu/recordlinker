from __future__ import absolute_import
from __future__ import print_function

import re
import itertools

import pandas as pd
import numpy as np

def lower_and_strip(x):
    '''Lower case, strip white space and punctuation'''
    pattern = '[^a-z|\s]'
    if isinstance(x, str):
        x = re.sub(pattern, '', x.lower()).strip()
    return x

def clean_names(dataframe, name_cols, split_on_comma=True, max_splits=None):
    '''Removes punctuation and lower cases names in a dataframe

    :return: DataFrame with cleaned names
    '''
    for col in name_cols:
        if split_on_comma:
            split_columns = dataframe[col].str.split(',', max_splits, expand=True)
            for col in split_columns.columns:
                split_columns[col] = split_columns[col].apply(lower_and_strip)
            dataframe = dataframe.merge(split_columns, left_index=True, right_index=True)
        else:
            dataframe[col] = dataframe[col].apply(lower_and_strip)
    return dataframe

def embed_letters(name, max_length, normalize=False, return_length=False):
    letters = 'abcdefghijklmnopqrstuvwxyz '
    vec_name = [0] * max_length
    num_letters = len(name)
    for i in range(min(max_length, len(name))):
        letter = name[i]
        vec_name[i] = letters.index(letter) + 1
    vec_name = np.array(vec_name)
    if normalize:
        vec_name = vec_name / 27.
    if return_length:
        return vec_name, num_letters
    else:
        return vec_name

def disembed_letters(vec_name, normalized=False):
    letters = 'abcdefghijklmnopqrstuvwxyz '
    name = []
    for i in range(len(np.trim_zeros(vec_name))):
        if normalized:
            index = int(round(vec_name[i] * 27)-1)
        else:
            index = int(vec_name[i] - 1)
        name.append(letters[index])
    return ''.join(name)