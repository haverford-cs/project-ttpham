"""
Author: Tessa Pham
Predict stock prices based on daily news headlines using different models.
"""
import pandas as pd
import numpy as np
import math
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

# my imports
import utils

def main():
    train, test = utils.load_data()

    # process data
    X_train, y_train = utils.process(train)
    X_test, y_test = utils.process(test)

    # create vectorizer
    vectorizer = CountVectorizer()
    train_bag = vectorizer.fit_transform(X_train)
    test_bag = vectorizer.transform(X_test)
    n_features = train_bag.shape[1]
    
    """
    # bag-of-words model without using TFIDF
    # Logistic Regression
    print('\n------------\nLogistic Regression\n------------')
    clf = LogisticRegression().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))

    # KNN
    print('\n------------\nKNN\n------------')
    clf = KNeighborsClassifier(n_neighbors=3).fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))

    # Naive Bayes
    print('\n------------\nNaive Bayes\n------------')
    clf = MultinomialNB().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))

    # SVM
    print('\n------------\nSVM\n------------')
    clf = SGDClassifier().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    
    # Random Forest
    print('\n------------\nRandom Forest\n------------')
    features = int(n_features * 0.5)
    clf = RandomForestClassifier(n_estimators=20, max_features=features).fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    """

    """
    # parameters for tuning
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], # choose unigram or bigram
                  'tfidf__use_idf': (True, False),
                  'base__max_features': [int(features) for features in [n_features * 0.001, n_features * 0.1, n_features * 0.5, math.sqrt(n_features)]],
                  'base__n_estimators': range(10, 100, 10)
    }
    """

    # Logistic Regression
    print('\n------------\nLogistic Regression\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', LogisticRegression(C=0.001)),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(X_test)[:, 1]
    # AUC score
    auc = roc_auc_score(y_test, probs)
    print('ROC AUC = {0:.3f}'.format(auc))
    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, marker='.', label='ROC Curve for Logistic Regression')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    
    # KNN
    print('\n------------\nKNN\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', KNeighborsClassifier(n_neighbors=32)),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(X_test)[:, 1]
    # AUC score
    auc = roc_auc_score(y_test, probs)
    print('ROC AUC = {0:.3f}'.format(auc))
    
    # Naive Bayes
    print('\n------------\nNaive Bayes\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', MultinomialNB(alpha=0.01)),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(X_test)[:, 1]
    # AUC score
    auc = roc_auc_score(y_test, probs)
    print('ROC AUC = {0:.3f}'.format(auc))
    
    # SVM
    print('\n------------\nSVM\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', SGDClassifier(loss='log', alpha=0.01)),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(X_test)[:, 1]
    # AUC score
    auc = roc_auc_score(y_test, probs)
    print('ROC AUC = {0:.3f}'.format(auc))
    
    # Random Forest
    print('\n------------\nRandom Forest\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', RandomForestClassifier(n_estimators=80, max_features=31)),
    ])
    clf.fit(X_train, y_train)
    """
    # get the best params
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf.fit(X_train, y_train)
    print(f'best score: {gs_clf.best_score_}, best params: {gs_clf.best_params_}')
    """
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(X_test)[:, 1]
    # AUC score
    auc = roc_auc_score(y_test, probs)
    print('ROC AUC = {0:.3f}'.format(auc))


if __name__ == '__main__':
    main()