"""
Author: Tessa Pham
Predict stock prices based on daily news headlines using different models.
"""
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
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

    # Logistic Regression
    print('\n------------\nLogistic Regression\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', LogisticRegression()),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))

    # KNN
    print('\n------------\nKNN\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', KNeighborsClassifier(n_neighbors=3)),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))

    # Naive Bayes
    print('\n------------\nNaive Bayes\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', MultinomialNB()),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))

    # SVM
    print('\n------------\nSVM\n------------')
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', SGDClassifier()),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    
    # Random Forest
    print('\n------------\nRandom Forest\n------------')
    features = int(n_features * 0.5)
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', RandomForestClassifier(n_estimators=20, max_features=features)),
    ])
    clf.fit(X_train, y_train)
    # test
    predictions = clf.predict(X_test)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()