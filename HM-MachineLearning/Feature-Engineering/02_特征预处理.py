import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

def minmax_demo():
    "归一化"
    # 1. 获取数据
    data = pd.read_csv("/Users/shenzongqi/Desktop/PythonProject/Self_Learning/HM_Machine-Learning/Data/datingTestSet2.txt")
    data = data.iloc[:, :3]
    print(data)
    # 2. 实例化一个转化器类
    transfer = MinMaxScaler(feature_range=[2,3])
    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


def stand_demo():
    "标准化"
    # 1. 获取数据
    data = pd.read_csv("/Users/shenzongqi/Desktop/PythonProject/Self_Learning/HM_Machine-Learning/Data/datingTestSet2.txt")
    data = data.iloc[:, :3]
    print(data)
    # 2. 实例化一个转化器类
    transfer = StandardScaler()
    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


if __name__ == "__main__":
    #minmax_demo()
    stand_demo()





# Do it manualy
data = pd.read_csv("/Users/shenzongqi/Desktop/PythonProject/Self_Learning/HM_Machine-Learning/Data/datingTestSet2.txt")
#归一化
(data - np.max(data)) / (np.max(data) - np.min(data))
#标准化
(data - np.mean(data)) / np.std(data)













