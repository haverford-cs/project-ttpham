# Author: Tessa Pham
# Description: Predict stock prices based on daily news headlines using different models.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics

# my imports
import utils

def main():
    train, test = utils.load_data()
    print(f'# train examples: {len(train)}')
    print(f'# test examples: {len(test)}')

    # create vectorizer and preprocess data
    vectorizer = CountVectorizer()
    train_headlines = utils.preprocess(train)
    train_bag = vectorizer.fit_transform(train_headlines)
    test_headlines = utils.preprocess(test)
    test_bag = vectorizer.transform(test_headlines)

    # print(f'# words in bag: {train_bag.shape}')
    # print(vectorizer.vocabulary_) # (key: feature, value: column index)
    # print(list(vectorizer.vocabulary_.keys())) # vocabulary
    
    # Logistic Regression
    clf = LogisticRegression()
    clf.fit(train_bag, train['Label'])

    # test
    predictions = clf.predict(test_bag)

    # confusion matrix
    confusion_matrix = pd.crosstab(test['Label'], predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)

    # feature analysis: which features are more predictive/informative
    # predict day x based on previous days

    # KNN
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_bag, train['Label'])

    # test
    predictions = clf.predict(test_bag)

    # confusion matrix
    confusion_matrix = pd.crosstab(test['Label'], predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)

    # SVM
    clf = SGDClassifier()
    clf.fit(train_bag, train['Label'])

    # test
    predictions = clf.predict(test_bag)

    # confusion matrix
    confusion_matrix = pd.crosstab(test['Label'], predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    
    # Naive Bayes


if __name__ == '__main__':
    main()