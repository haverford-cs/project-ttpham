# Stock Market Prediction Using Daily News
## Tessa Pham

## Lab Notebook

12/10/2019 (2 hours)
* examined data format
* found tools in pandas and sklearn for loading and processing data
* implemented util functions for processing data into training and test sets

12/11/2019 (2 hours)
* read about bag-of-words model
* played around with DataFrame
* implemented bag-of-words with CountVectorizer

12/12/2019 (1 hour)
* ran Logistic Regression
* produced confusion matrix using code from lab
* 42% accuracy only!
* selecting more ML algos to experiment with

12/13/2019 (1 hour)
* produced confusion matrix with pandas
* exploring NLP methods to get better predictions with text data

12/15/2019 (1 hour)
* implemented KNN and SVM
* read about TFIDF

12/17/2019 (8-10 hours)
* implemented Naive Bayes and Random Forest
* added pipeline with TFIDF for each algo
* tried not counting stop words => worse performance
* GridSearchCV to tune parameters for each of 5 classifiers
* A LOT of test runs for tuned parameters and fixing stuff (most time-consuming!)
* calculated AUC scores
* added stemming to vectorizer to reduce # features
* plotted ROC curves for each classifier against random guessing
* exploring bigram models

12/18/2019:

(4 hours)
* implemented bag-of-bigrams model
* generated ROC curve plots for bigram model with different classifiers
* tried bigram + TFIDF => worse performance
* calculated coefficients for words, selecting top words with highest pos/neg correlations with '1' label
* worked on presentation

(5 hours)
* refactored code for 2 separate models
* put back calculations for coeffs and TFIDF values for both models
* exported coeffs and TFIDF values (i.e., word importances) to Excel and CSV files
* final reruns, performance records, and plot figures for all model + classifier pairs
* updated README

## References

* Working With Text Data. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html.
* Shaikh, J. (2017, July). Machine Learning, NLP: Text Classification Using scikit-learn, Python and NLTK. https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a.
* Harlalka, R. (2018, June). Choosing the Right Machine Learning Algorithm. https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f.
* Gelé, A. (2016). OMG! NLP with the DJIA and Reddit!. https://www.kaggle.com/ndrewgele/omg-nlp-with-the-djia-and-reddit.
* Ganesan, K. How to Use Tfidftransformer & Tfidfvectorizer?. https://kavita-ganesan.com/tfidftransformer-tfidfvectorizer-usage-differences/#.Xfm7sdZKjBI.
* Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved [2019, November 16] from https://www.kaggle.com/aaron7sun/stocknews.

## Navigate the Repository
* `run_unigram.py`: runs 5 classifiers with the bag-of-words + TFIDF model.
* `run_bigram.py`: runs 5 classifiers with the bag-of-bigrams model.
ROC curve plots will be recreated from each run, thus may be different from the results recorded in the next section (e.g., for Random Forest).
* `stocknews`: contains labeled data.
* `figs`: contains ROC curve plot figures.
* `idf`: contains TFIDF values for all words and bigrams in the vocabulary of the given text data. A number of most and least important words are exported into separate files for ease of reference.
* `coeffs`: contains coefficients for all words and bigrams in the vocabulary of the given text data. 30 words with highest/lowest correlations with the '1' label are provided in separate files for the unigram model, and 300 words for the bigram model.

## Performance Records
### bag-of-words only
Logistic Regression: 42.59%\
KNN: 51.06%\
Naive Bayes: 49.47%\
SVM: 44.97%\
Random Forest: 44.97%

### bag-of-words + tf-idf
Logistic Regression: 48.68%\
KNN: 50.53%\
Naive Bayes: 50.79%\
SVM: 44.44%\
Random Forest: 51.59% (range 50-52%)

### parameter tuning
Logistic Regression:\
best score: 0.5418994413407822, best params: {'base__C': 0.001, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)}\
accuracy = 50.79%\
ROC AUC = 0.454

KNN: best score: 0.5543140906269398, best params: {'base__n_neighbors': 32, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}\
accuracy = 52.65%\
ROC AUC = 0.536

Naive Bayes:\
best score: 0.5406579764121664, best params: {'base__alpha': 0.01, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}\
accuracy = 51.59%\
ROC AUC = 0.488

SVM:\
best score: 0.5418994413407822, best params: {'base__alpha': 0.01, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)}
accuracy = 50.79\
ROC AUC = 0.454

Random Forest:\
best score: 0.5450031036623215, best params: {'base__max_features': 31, 'base__n_estimators': 80, 'tfidf__use_idf': False, 'vect__ngram_range': (1, 2)}\
ROC AUC = 0.526

### bag-of-words (with stemming & tuned params) + tf-idf
```
------------
Logistic Regression
------------
Predicted    1
True
0          186
1          192
accuracy = 50.79%
ROC AUC = 0.468

------------
KNN
------------
Predicted   0    1
True
0          63  123
1          57  135
accuracy = 52.38%
ROC AUC = 0.534

------------
Naive Bayes
------------
Predicted   0    1
True
0          56  130
1          52  140
accuracy = 51.85%
ROC AUC = 0.511

------------
SVM
------------
Predicted    1
True
0          186
1          192
accuracy = 50.79%
ROC AUC = 0.506

------------
Random Forest
------------
Predicted   0    1
True
0          44  142
1          44  148
accuracy = 50.79%
ROC AUC = 0.523
```

### bag-of-bigrams
```
------------
Logistic Regression
------------
Predicted   0    1
True
0          66  120
1          45  147
accuracy = 56.35%
ROC AUC = 0.591

------------
KNN
------------
Predicted  0    1
True
0          2  184
1          3  189
accuracy = 50.53%
ROC AUC = 0.507

------------
Naive Bayes
------------
Predicted   0    1
True
0          17  169
1          21  171
accuracy = 49.74%
ROC AUC = 0.481

------------
SVM
------------
Predicted   0    1
True
0          89   97
1          69  123
accuracy = 56.08%
ROC AUC = 0.554

------------
Random Forest
------------
Predicted   0    1
True
0          55  131
1          49  143
accuracy = 52.38%
ROC AUC = 0.531
```
