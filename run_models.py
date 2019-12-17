"""
Author: Tessa Pham
Predict stock prices based on daily news headlines using different models.
"""
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

# my imports
import utils

def main():
    train, test = utils.load_data()
    print(f'# train examples: {len(train)}')
    print(f'# test examples: {len(test)}')

    # create vectorizer and preprocess data
    vectorizer = CountVectorizer()
    X_train, y_train = utils.process(train)
    train_bag = vectorizer.fit_transform(X_train)
    n_features = train_bag.shape[1]
    print(n_features)
    X_test, y_test = utils.process(test)
    test_bag = vectorizer.transform(X_test)
    
    # Logistic Regression
    print('\n------------\nLogistic Regression\n------------')
    clf = LogisticRegression().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    print(f'accuracy = {np.mean(predictions == y_test)}')

    # feature analysis: which features are more predictive/informative
    # predict day x based on previous days

    # KNN
    print('\n------------\nKNN\n------------')
    clf = KNeighborsClassifier(n_neighbors=3).fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    print(f'accuracy = {np.mean(predictions == y_test)}')

    # Naive Bayes
    print('\n------------\nNaive Bayes\n------------')
    clf = MultinomialNB().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    print(f'accuracy = {np.mean(predictions == y_test)}')

    # SVM
    print('\n------------\nSVM\n------------')
    clf = SGDClassifier().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    print(f'accuracy = {np.mean(predictions == y_test)}')
    
    # Random Forest
    print('\n------------\nRandom Forest\n------------')
    features = int(n_features * 0.5)
    clf = RandomForestClassifier(n_estimators=20, max_features=features).fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    print(f'accuracy = {np.mean(predictions == y_test)}')


if __name__ == '__main__':
    main()