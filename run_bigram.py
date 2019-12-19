"""
Author: Tessa Pham
Predict stock market changes based on daily news headlines using bag-of-bigrams model
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
    bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
    train_bag = bigram_vectorizer.fit_transform(X_train)
    test_bag = bigram_vectorizer.transform(X_test)
    n_features = train_bag.shape[1]

    # calculate coeffs for bigrams
    clf_bigram = LogisticRegression().fit(train_bag, y_train)
    words = bigram_vectorizer.get_feature_names()
    coeffs = clf_bigram.coef_.tolist()[0]
    df_coeff = pd.DataFrame({'Word': words, 'Coefficient': coeffs})
    df_coeff = df_coeff.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
    df_coeff.head(300).to_excel('coeffs/bigram_pos_300.xlsx')
    df_coeff.tail(300).to_excel('coeffs/bigram_neg_300.xlsx')
    df_coeff.to_csv('coeffs/bigram.csv')
    
    # get TFIDF values
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(train_bag)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=words, columns=["idf_weights"])
    df_idf = df_idf.sort_values(by=['idf_weights'])
    df_idf.to_csv('idf/values_bigram.csv')
    df_idf.head(1000).to_excel('idf/bigram_most_important_1000.xlsx')
    df_idf.tail(100).to_excel('idf/bigram_least_important_100.xlsx')
    
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
    # probabilities for positives
    random_probs = [0 for _ in range(len(y_test))]
    probs = clf.predict_proba(test_bag)[:, 1]
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
    plt.savefig('figs/lr_bigram.png', format='png')
    plt.show()

    # KNN
    print('\n------------\nKNN\n------------')
    clf = KNeighborsClassifier().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(test_bag)[:, 1]
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
    plt.savefig('figs/knn_bigram.png', format='png')
    plt.show()

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
    # probabilities for positives
    probs = clf.predict_proba(test_bag)[:, 1]
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
    plt.savefig('figs/nb_bigram.png', format='png')
    plt.show()

    # SVM
    print('\n------------\nSVM\n------------')
    clf = SGDClassifier().fit(train_bag, y_train)
    calibrator = CalibratedClassifierCV(clf, cv='prefit') # calibrate to calculate probabilities
    clf = calibrator.fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(test_bag)[:, 1]
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
    plt.savefig('figs/svm_bigram.png', format='png')
    plt.show()

    # Random Forest
    print('\n------------\nRandom Forest\n------------')
    clf = RandomForestClassifier().fit(train_bag, y_train)
    # test
    predictions = clf.predict(test_bag)
    # confusion matrix
    confusion_matrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)
    accuracy = np.mean(predictions == y_test)
    print('accuracy = {0:.2f}%'.format(accuracy * 100))
    # probabilities for positives
    probs = clf.predict_proba(test_bag)[:, 1]
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
    plt.savefig('figs/rf_bigram.png', format='png')
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
    plt.savefig('figs/all_bigram.png', format='png')
    plt.show()


if __name__ == '__main__':
    main()