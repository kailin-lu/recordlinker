'''Preprocess string name columns in dataframes'''

from __future__ import absolute_import
from __future__ import print_function

import re
import numpy as np
import fuzzy

from keras.utils.np_utils import to_categorical

from . import utils


def lower_and_strip(x):
    '''Lower case, strip white space and punctuation'''
    pattern = '[^a-z|\s]'
    if isinstance(x, str):
        x = re.sub(pattern, '', x.lower()).strip()
    return x


def clean_names(dataframe,
                name_cols,
                split_on_comma=True,
                max_splits=None):
    '''Removes punctuation and lower cases names in a dataframe

    :return: DataFrame with cleaned names
    '''
    for col in name_cols:
        if split_on_comma:
            split_columns = dataframe[col].str.split(',',
                                                     max_splits,
                                                     expand=True)
            for col in split_columns.columns:
                split_columns[col] = split_columns[col].apply(lower_and_strip)

            dataframe = dataframe.merge(split_columns,
                                        left_index=True,
                                        right_index=True)
        else:
            dataframe[col] = dataframe[col].apply(lower_and_strip)
    return dataframe


def embed(names,
          embed_type,
          max_length=12,
          normalize=True,
          categorical=False):
    '''Create training data for autoencoder

    :return: numpy array of size [n_names, max_length]'''
    if embed_type not in ['letters', 'shingles', 'l', 's']:
        print('`embed_type` not recognized. Must be "letters" or '
              '"shingles"')
        return
    if embed_type in ['letters', 'l']:
        embed_func = embed_letters
    else:
        embed_func = embed_shingles
    embedded = [embed_func(name, max_length=max_length, normalize=normalize)
                for name in names]
    embedded = np.vstack(embedded)
    if categorical:
        if embed_type in ['letters', 'l']:
            return to_categorical(embedded, 28)
        else:
            return to_categorical(embedded, np.max(embedded) + 1)
    else:
        return embedded


def embed_letters(name,
                  max_length=12,
                  normalize=False,
                  return_length=False):
    '''Embed a string name as a vector using letters

    :param name: string to convert
    :param max_length: int for max length to convert
    :param normalize: normalize to 0-1 if True
    :param return_length: return original length of string if True

    :return: numpy vector of length max_length
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz '
    vec_name = [0] * max_length
    num_letters = len(name)
    for i in range(min(max_length, len(name))):
        letter = name[i]
        vec_name[i] = letters.index(letter) + 1
    vec_name = np.array(vec_name)
    if normalize:
        vec_name = vec_name / len(letters)
    if return_length:
        return vec_name, num_letters
    else:
        return vec_name


def disembed_letters(vec_name,
                     onehot=False):
    '''
    Convert letter-embedded names back to strings

    :param vec_name:
    :param normalized:
    :return:
    '''
    letters = 'abcdefghijklmnopqrstuvwxyz '
    if len(vec_name.shape) == 2:
        vec_name = np.argmax(vec_name, -1)
    name = []
    for i in range(len(np.trim_zeros(vec_name))):
        if np.max(vec_name) <= 1.:
            index = int(round(vec_name[i] * 27) - 1)
        else:
            index = int(vec_name[i] - 1)
        if index <= len(letters):
            name.append(letters[index])
        else:
            name.append(' ')
    return ''.join(name)


shingles = utils.k_shingles(2)


def embed_shingles(name, max_length, k=2, normalize=False):
    '''Embed string names as shingle vectors

    :param name:
    :param max_length:
    :param k:
    :param normalize:
    :param return_length:
    '''
    if k == 2:
        pairs = shingles
    else:
        pairs = utils.k_shingles(k)
    vec_name = []
    for i in range(min(len(name), max_length)):
        shingle = name[i:i + 2]
        try:
            vec_name.append(pairs.index(shingle))
        except ValueError:
            pass
    if len(vec_name) != max_length:
        remaining = [0] * (max_length - len(vec_name))
        vec_name.extend(remaining)
    vec_name = np.array(vec_name)
    if normalize:
        return vec_name / len(pairs)
    else:
        return vec_name


def disembed_shingles(vec_name, k=2, normalize=False):
    '''Disembed vectors back into string names'''
    if k == 2:
        pairs = shingles
    else:
        pairs = utils.k_shingles(k)
    name = ''
    for e, i in enumerate(vec_name):
        if np.max(vec_name) <= 1.0:
            index = int(round(i * len(pairs)))
        else:
            index = int(round(i))
        try:
            if np.sum(vec_name[e + 1:]) == 0.:
                name += pairs[index]
            else:
                name += pairs[index][0]
        except IndexError:
            pass
    return name


########## Other Embedding Methods ##############

def embed_soundex(name, max_length):
    '''Soundex algorithm described https://en.wikipedia.org/wiki/Soundex
    Modified original algorithm to include repeated codings'''
    vec_soundex = [0] * max_length
    letters = 'abcdefghijklmnopqrstuvwxyz'

    name = re.sub(r'[^a-z|\s]', '', name.lower().strip()).strip()

    # Save first letter of the name
    if len(name) > 0:
        try:
            vec_soundex[0] = letters.index(name[0]) + 1
        except ValueError:
            print(name)

    # Remove  a, e, i, o, u, y, h, w
    # Keep first letter of name
    vowels_plus = ['a', 'e', 'i', 'o', 'u', 'h', 'w']
    name = [letter for letter in name if
            letter not in vowels_plus or name.index(letter) == 0]

    # Apply soundex mapping
    for i in range(1, min(max_length, len(name))):
        # If current letter is same as previous letter, skip
        if name[i] == name[i - 1]:
            pass
        if name[i] in ['b', 'f', 'p', 'v']:
            vec_soundex[i] = 1
        elif name[i] in ['c', 'g', 'j', 'k', 'q', 's', 'x', 'z']:
            vec_soundex[i] = 2
        elif name[i] in ['d', 't']:
            vec_soundex[i] = 3
        elif name[i] == 'l':
            vec_soundex[i] = 4
        elif name[i] in ['m', 'n']:
            vec_soundex[i] = 5
        elif name[i] == 'r':
            vec_soundex[i] = 6
        else:
            pass
    return np.array(vec_soundex).astype(float)


def embed_nysiis(name):
    '''NYIIS algorithm, replacing each letter with letter index 0-25
    Cut off after 8 characters'''
    letters = 'abcdefghijklmnopqrstuvwxyz'
    nysiis = fuzzy.nysiis(name).lower()
    vec = [0] * 8
    for i in range(min(len(vec), len(nysiis))):
        vec[i] += letters.index(nysiis[i])
    return vec
