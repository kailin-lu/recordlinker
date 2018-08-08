import re
import itertools

import numpy as np
import fuzzy

def k_shingles(k):
    '''Returns list of k-shingles out of 26 letters + space'''
    letters = 'abcdefghijklmnopqrstuvwxyz '
    shingles = list(map(''.join, itertools.permutations(letters, k)))
    for letter in letters:
        shingles.append(letter * k)
    shingles.insert(0, '.  ')
    return shingles

# def embed_consecutive_shingles(name, max_length, pairs=pairs):
#     vec_name = [''] * max_length
#     if len(name) % 2 == 1:
#         name = name + ' '
#     num_shingles = len(name)
#     idx = 0
#     for i in range(0,min(max_length, num_shingles),2):
#         try:
#             shingle = name[i:i+2]
#             vec_name[idx] = shingle
#             idx += 1
#         except:
#             print(name[i:i+2])
#     return vec_name, num_shingles // 2

def embed_soundex(name, max_length):
    """
    Soundex algorithm described https://en.wikipedia.org/wiki/Soundex
    Modified original algorithm to include repeated codings
    """
    vec_soundex=[0]*max_length
    letters = 'abcdefghijklmnopqrstuvwxyz'

    name = re.sub(r'[^a-z|\s]', '', name.lower().strip()).strip()

    # Save first letter of the name
    if len(name) >  0:
        try:
            vec_soundex[0] = letters.index(name[0])+1
        except ValueError:
            print(name)

    # Remove  a, e, i, o, u, y, h, w
    # Keep first letter of name
    vowels_plus = ['a', 'e', 'i', 'o', 'u', 'h', 'w']
    name = [letter for letter in name if letter not in vowels_plus or name.index(letter)==0]

    # Apply soundex mapping
    for i in range(1, min(max_length,len(name))):
        # If current letter is same as previous letter, skip
        if name[i]==name[i-1]:
            pass
        if name[i] in ['b','f','p','v']:
            vec_soundex[i] = 1
        elif name[i] in ['c','g','j','k','q','s','x','z']:
            vec_soundex[i] = 2
        elif name[i] in ['d', 't']:
            vec_soundex[i] = 3
        elif name[i] == 'l':
            vec_soundex[i] = 4
        elif name[i] in ['m','n']:
            vec_soundex[i] = 5
        elif name[i] == 'r':
            vec_soundex[i] = 6
        else:
            pass
    return np.array(vec_soundex).astype(float)

def embed_nysiis(name):
    """
    NYIIS algorithm, replacing each letter with letter index 0-25
    Cut off after 8 characters
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    nysiis = fuzzy.nysiis(name).lower()
    vec = [0] * 8
    for i in range(min(len(vec), len(nysiis))):
        vec[i] += letters.index(nysiis[i])
    return vec

def cos_similarity(u, v):
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    num = np.dot(u, v)
    if denom == 0:
        return num/1e-10
    else:
       return num / denom