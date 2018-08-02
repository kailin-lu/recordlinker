import re
import pandas as pd

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