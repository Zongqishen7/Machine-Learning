# (1)Filter 过滤式：方差选择：低方差过滤
#原理： 如果有两列数据的方差几乎相同那么他们很有可能就是冗余变量可以进行筛选，同是threshold = ？可以自己选择方差数字
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def variance_demo():
    data = pd.read_csv("/Users/shenzongqi/Desktop/PythonProject/Self_Learning/HM_Machine-Learning/Data/factor_returns.csv")
    data = data.iloc[:, 1:-2]
    transfer = VarianceThreshold(threshold=5)
    data_new = transfer.fit_transform(data)
    print(data_new, data_new.shape)
    return None


# (2)相关系数
#当r>0时，表示两变量正相关，r<0时，两变量为负相关
#当|r|=1时，表示两变量为完全相关，当r=0时，表示两变量间无相关关系
#当0<|r|<1时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱
#一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关
#这个符号：|r|为r的绝对值， |-5| = 5
from scipy.stats import pearsonr
def cor_coefficient():
    data = pd.read_csv(
        "/Users/shenzongqi/Desktop/PythonProject/Self_Learning/HM_Machine-Learning/Data/factor_returns.csv")
    #计算两个变量之间的相关系数
    #r = pearsonr(data["pe_ratio"], data["pb_ratio"])
    r = pearsonr(data["revenue"], data["total_expense"])

    print(r)
#pearonr 返回的第二个值为p-value


#(3) PCA主成分分析
from sklearn.decomposition import PCA
def pca_demo():
    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    #从四个特征降维成两个特征
    #transform = PCA(n_components = 2)
    #如果要保留百分之80的信息那么降维后只有两列
    transform = PCA(n_components = 0.8)
    new_data = transform.fit_transform(data)
    print(pd.DataFrame(data))
    print(pd.DataFrame(new_data))
    return None
if __name__ == "__main__":
    variance_demo()
    #cor_coefficient()
    #pca_demo()
















