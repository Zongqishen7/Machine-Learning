from random import random
from unicodedata import name
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.tree import export_graphviz
from sklearn import tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 快速上手鸢尾花决策树
def decision_tree():
    # 用决策树对鸢尾花进行分类
    #1)获取数据集
    iris = load_iris()
    #2)划分数据集
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state= 2)
    #3)决策树预估器
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(X_train, y_train)
    #4)模型评估
    predicted = estimator.predict(X_test)
    print("Accuracy:  \n", np.mean(y_test == predicted))
    print("Score:  \n", accuracy_score(y_test, predicted))

    #5)可视化
    #export_graphviz(estimator, out_file="iris_dtree.dot", feature_names=iris.feature_names)

    return None

if __name__ == "__main__":
    decision_tree()


