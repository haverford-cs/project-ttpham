# Author: Tessa Pham
# Description: Utils for loading and preprocessing data.

import pandas as pd

def load_data():
    data = pd.read_csv('stocknews/Combined_News_DJIA.csv')
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']
    return train, test

def preprocess(data):
    headlines = []
    for row in range(0, len(data.index)):
        headlines.append(' '.join(str(x) for x in data.iloc[row,2:27]))
    return headlines