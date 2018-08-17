'''Helper functions'''

import itertools

def k_shingles(k):
    '''Returns list of k-shingles out of 26 letters + space'''
    letters = 'abcdefghijklmnopqrstuvwxyz '
    shingles = list(map(''.join, itertools.permutations(letters, k)))
    for letter in letters:
        shingles.append(letter * k)
    shingles.insert(0, '')
    return shingles


