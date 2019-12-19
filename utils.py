"""
Author: Tessa Pham
Utils for loading and processing data.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

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

def process(data):
    """
    Process a dataset into feature values and labels.
    
    Parameters:
        data (DataFrame) -- a dataset
    
    Returns:
        X (list) -- list of daily news headlines (25 headlines concatenated for each day)
        y (list) -- list of labels for DJIA close value changes ('0' for decrease, '1' otherwise)
    """
    X = []
    y = data['Label']
    for row in range(0, len(data.index)):
        X.append(' '.join(str(x) for x in data.iloc[row, 2:27]))
    return X, y

# code referenced from:
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    """
    A wrapper class of CountVectorizer that counts word stems instead of different
    forms of a word (e.g., 'does' and 'did' are both counted as 'do'). Words are stemmed
    using a stemmer from NLTK.
    """
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])