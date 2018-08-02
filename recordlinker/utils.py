import numpy as np 
import pandas as pd
import os
import itertools
import re 

FILENAME = '/data/mil_names.csv'

def k_shingles(k): 
    """
    Returns list of k-shingles out of 26 letters + space 
    """
    letters = 'abcdefghijklmnopqrstuvwxyz '
    shingles = list(map(''.join, itertools.permutations(letters, k))) 
    for letter in letters: 
        shingles.append(letter * k)
    shingles.insert(0, '.  ')
    return shingles 

pairs = k_shingles(2)

def load_data(pairs=pairs, filename=FILENAME, name='recname1'): 
    """
    Loads names as input matrices 
    """
    path = os.getcwd() + filename
    data = pd.read_csv(path)
    
    # normalize by lower casing and removing punctuation 
    names = data[name].apply(lambda x: (re.sub(r'[^a-z|\s]', '', x.lower()).strip()).strip())
    max_length = max([len(name) for name in names])
        
    print('Embedding names into 2 shingles with length {}'.format(max_length))
    names = [embed_shingles(name, max_length) for name in names]
    data = [item[0] for item in names] 
    lengths = [item[1] for item in names] 
    data = np.vstack(data)                     
    return data, lengths


def embed_shingles(name, max_length, pairs=pairs, return_shingles=False):
    vec_name = [0] * max_length
    num_shingles = len(name)-1
    for i in range(min(max_length, num_shingles)):
        try: 
            shingle = name[i:i+2]
            if return_shingles: 
                vec_name[i] = shingle
            else:
                vec_name[i] = pairs.index(shingle)
        except: 
            print(name, name[i:i+2]) 
    if return_shingles:
        return vec_name, num_shingles
    else:
        return np.array(vec_name), num_shingles
        
def disembed_shingles(vec_name, pairs=pairs):
    name = [int(round(i)) for i in list(vec_name) if i != 0.] 
    decoded = ''
    for i in range(len(name)):
        try:
            decoded = decoded + pairs[name[i]][0]
        except:
            decoded = decoded + ' '
    decoded = decoded + pairs[name[i]-1][1]
    return decoded

def embed_consecutive_shingles(name, max_length, pairs=pairs): 
    vec_name = [''] * max_length
    if len(name) % 2 == 1: 
        name = name + ' '
    num_shingles = len(name)
    idx = 0
    for i in range(0,min(max_length, num_shingles),2):
        try: 
            shingle = name[i:i+2]
            vec_name[idx] = shingle
            idx += 1 
        except: 
            print(name[i:i+2])
    return vec_name, num_shingles // 2 
        

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
        
       
def embed_letters(name, max_length): 
    letters = 'abcdefghijklmnopqrstuvwxyz '
    vec_name = [0] * max_length 
    num_letters = len(name) 
    for i in range(min(max_length, len(name))): 
        letter = name[i] 
        vec_name[i] = letters.index(letter)+1 
    return np.array(vec_name), num_letters
    

def disembed_letters(vec_name): 
    letters = 'abcdefghijklmnopqrstuvwxyz '
    name = [] 
    for i in range(len(vec_name)): 
        index = int(vec_name[i]-1)
        name.append(letters[index])
    return ''.join(name)
    

def cos_similarity(u, v): 
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    num = np.dot(u, v)
    if denom == 0: 
        return num/1e-10
    else: 
        return num / denom