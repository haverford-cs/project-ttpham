# Author: Tessa Pham
# Description: Predict stock prices based on daily news headlines using different models.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# my imports
import utils

def main():
    train, test = utils.load_data()
    print(f'# train examples: {len(train)}')
    print(f'# test examples: {len(test)}')

    # train bag-of-word model
    train_headlines = utils.preprocess(train)
    vectorizer = CountVectorizer()
    train_bag = vectorizer.fit_transform(train_headlines)
    print(f'# words in bag: {train_bag.shape}')
    model = LogisticRegression()
    model = model.fit(train_bag, train['Label'])

    # test bag-of-words model
    test_headlines = utils.preprocess(test)
    test_bag = vectorizer.transform(test_headlines)
    predictions = model.predict(test_bag)

    # confusion matrix
    confusion_matrix = pd.crosstab(test['Label'], predictions, rownames=['True'], colnames=['Predicted'])
    print(confusion_matrix)

    # feature analysis: which features are more predictive/informative
    # predict day x based on previous days


if __name__ == '__main__':
    main()