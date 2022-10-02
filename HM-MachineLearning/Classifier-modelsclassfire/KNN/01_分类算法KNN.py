from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, plot_roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



###
#鸢尾花KNN模型
def knn_iris():
    """
    用KNN算法对鸢尾花进行分类
    """
    #1:获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    #2:分类训练测试集
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,random_state=6)

    #3.标准化
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    #4:分类评估器
    neigh = KNeighborsClassifier(n_neighbors = 11)
    neigh.fit(X_train,y_train)
    predicted = neigh.predict(X_test)

    #5:模型评估
    print("confusion_matrix: \n",pd.DataFrame(confusion_matrix(y_test, predicted)))
    print("accuracy_score: \n", accuracy_score(y_test, predicted))
    print("Directly compare: \n", y_test == predicted)



###
#鸢尾花增加K值调优: 使用 GridSearchCV 构建估计器
def knn_iris_gscv():
    """
    用KNN算法对鸢尾花进行分类,添加网格搜索和交叉验证
    """
    #1:获取数据
    from sklearn.datasets import load_iris
    iris = load_iris()

    #2:分类训练测试集
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,random_state=6)

    #3.标准化
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    #4:分类评估器
    neigh = KNeighborsClassifier(n_neighbors = 3)

    #加入网格搜索与交叉验证
    #参数准备
    para_dict = {"n_neighbors": [1,3,5,7,9,11]}
    neigh = GridSearchCV(neigh, para_dict, cv = 10)
    print("------------\n", neigh)
    neigh.fit(X_train,y_train)
    predicted = neigh.predict(X_test)

    #5:模型评估
    print("confusion_matrix: \n",pd.DataFrame(confusion_matrix(y_test, predicted)))
    print("accuracy_score: \n", accuracy_score(y_test, predicted))
    print("Directly compare: \n", y_test == predicted)

    #6:
    #最佳参数：best_params
    print("最佳参数\n", neigh.best_params_)
    #最佳结果: best_score_
    print("最佳结果\n", neigh.best_score_)
    #最佳估计器
    print("最佳估计器\n", neigh.best_estimator_)
    #交叉验证结果: cv_results_
    print("交叉验证结果\n", neigh.cv_results_)





if __name__ == "__main__":
    knn_iris()
    #knn_iris_gscv()



















