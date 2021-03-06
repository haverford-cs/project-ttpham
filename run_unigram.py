"""
Author: Tessa Pham
Predict stock market changes based on daily news headlines using bag-of-words model with TFIDF
and different classifiers.
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

# my imports
import utils

def main():
    # load and process data
    train, test = utils.load_data()
    X_train, y_train = utils.process(train)
    X_test, y_test = utils.process(test)

    # create vectorizer
    vectorizer = utils.StemmedCountVectorizer()
    train_bag = vectorizer.fit_transform(X_train)
    n_features = train_bag.shape[1]

    # calculate coeffs for unigrams
    clf = LogisticRegression().fit(train_bag, y_train)
    words = vectorizer.get_feature_names()
    coeffs = clf.coef_.tolist()[0]
    df_coeff = pd.DataFrame({'Word': words, 'Coefficient': coeffs})
    df_coeff = df_coeff.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    df_coeff.head(30).to_excel('coeffs/unigram_pos_30.xlsx')
    df_coeff.tail(30).to_excel('coeffs/unigram_neg_30.xlsx')
    df_coeff.to_csv('coeffs/unigram.csv')
    
    # get features and TFIDF values
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(train_bag)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=words, columns=["idf_weights"])
    df_idf = df_idf.sort_values(by=['idf_weights'], ascending=False)
    df_idf.to_csv('idf/values.csv')
    df_idf.head(1000).to_excel('idf/most_important_1000.xlsx')
    df_idf.tail(100).to_excel('idf/least_important_100.xlsx')

    """
    # example parameter tuning for Random Forest
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], # choose unigram or bigram
                  'tfidf__use_idf': (True, False),
                  'base__max_features': [int(features) for features in [n_features * 0.001, n_features * 0.1, n_features * 0.5, math.sqrt(n_features)]],
                  'base__n_estimators': range(10, 100, 10)
    }
    """
    
    # Logistic Regression
    print('\n------------\nLogistic Regression\n------------')
    clf = Pipeline([('vect', utils.StemmedCountVectorizer()),
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
    random_probs = [0 for _ in range(len(y_test))]
    probs = clf.predict_proba(X_test)[:, 1]
    # AUC score
    random_auc = roc_auc_score(y_test, random_probs)
    auc = roc_auc_score(y_test, probs)
    print('ROC AUC = {0:.3f}'.format(auc))
    # plot ROC curve
    random_fpr, random_tpr, _ = roc_curve(y_test, random_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, probs)
    plt.title('ROC Curve Plot for Random Guessing vs. Logistic Regression (AUC = {0:.3f})'.format(auc))
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random Guessing')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figs/lr.png', format='png')
    plt.show()
    
    # KNN
    print('\n------------\nKNN\n------------')
    clf = Pipeline([('vect', utils.StemmedCountVectorizer()),
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
    # plot ROC curve
    knn_fpr, knn_tpr, _ = roc_curve(y_test, probs)
    plt.title('ROC Curve Plot for Random Guessing vs. KNN (AUC = {0:.3f})'.format(auc))
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random Guessing')
    plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN')
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figs/knn.png', format='png')
    plt.show()
    
    # Naive Bayes
    print('\n------------\nNaive Bayes\n------------')
    clf = Pipeline([('vect', utils.StemmedCountVectorizer()),
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
    # plot ROC curve
    nb_fpr, nb_tpr, _ = roc_curve(y_test, probs)
    plt.title('ROC Curve Plot for Random Guessing vs. Naive Bayes (AUC = {0:.3f})'.format(auc))
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random Guessing')
    plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes')
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figs/nb.png', format='png')
    plt.show()
    
    # SVM
    print('\n------------\nSVM\n------------')
    clf = Pipeline([('vect', utils.StemmedCountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('base', CalibratedClassifierCV(SGDClassifier(alpha=0.01), cv=5)), # calibrate to calculate probabilities
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
    svm_fpr, svm_tpr, _ = roc_curve(y_test, probs)
    plt.title('ROC Curve Plot for Random Guessing vs. SVM (AUC = {0:.3f})'.format(auc))
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random Guessing')
    plt.plot(svm_fpr, svm_tpr, marker='.', label='SVM')
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figs/svm.png', format='png')
    plt.show()
    
    # Random Forest
    print('\n------------\nRandom Forest\n------------')
    clf = Pipeline([('vect', utils.StemmedCountVectorizer()),
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
    # plot ROC curve
    rf_fpr, rf_tpr, _ = roc_curve(y_test, probs)
    plt.title('ROC Curve Plot for Random Guessing vs. Random Forest (AUC = {0:.3f})'.format(auc))
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random Guessing')
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figs/rf.png', format='png')
    plt.show()

    # plot all ROC curves
    plt.title('ROC Curve Plot for All Models and Random Guessing')
    plt.plot(random_fpr, random_tpr, linestyle='--', label='Random Guessing')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
    plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN')
    plt.plot(nb_fpr, nb_tpr, marker='.', label='Naive Bayes')
    plt.plot(svm_fpr, svm_tpr, marker='.', label='SVM')
    plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('figs/all.png', format='png')
    plt.show()


if __name__ == '__main__':
    main()