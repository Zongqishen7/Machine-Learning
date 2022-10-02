from unicodedata import name
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def nb_news():
    """
    用朴素贝叶斯算法对新闻进行分类
    return none
    """
    #1:获取数据
    from sklearn.datasets import fetch_20newsgroups
    news = fetch_20newsgroups(subset="all")
    
    #2:划分数据集
    X_train, X_test, y_train, y_test = train_test_split(news.data, news.target,random_state=6)

    #3.特征工程：文本特征抽取 tf——idf
    transfer = TfidfVectorizer()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    #4:分类评估器
    estimator = MultinomialNB()
    estimator.fit(X_train, y_train)
    predicted = estimator.predict(X_test)


    #5:模型评估
    #print("confusion_matrix: \n",pd.DataFrame(confusion_matrix(y_test, predicted)))
    print("accuracy_score: \n", accuracy_score(y_test, predicted))
    print("Directly compare: \n", np.mean(y_test == predicted))

if __name__ == "__main__":
    nb_news()
