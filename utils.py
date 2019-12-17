# Author: Tessa Pham
# Description: Utils for loading and preprocessing data.

import pandas as pd

def load_data():
    """
    Load data from given .csv file into train and test sets, split on date condition.
    
    Returns:
        train (DataFrame) -- training set
        test (DataFrame)  -- test set
    """
    data = pd.read_csv('stocknews/Combined_News_DJIA.csv')
    train_dates = data['Date'] < '2015-01-01'
    test_dates = data['Date'] > '2014-12-31'
    train = data[train_dates]
    test = data[test_dates]
    return train, test

def preprocess(data):
    """
    Preprocess a DataFrame into a list of headlines.
    
    Parameters:
        data (DataFrame) -- a table of dates and top 25 news headlines for each date
    
    Returns:
        headlines (list) -- list of daily news headlines
                            (25 headlines concatenated for each day)
    """
    headlines = []
    for row in range(0, len(data.index)):
        headlines.append(' '.join(str(x) for x in data.iloc[row, 2:27]))
    return headlines