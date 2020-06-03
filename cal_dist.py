# author: 龚潇颖(Xiaoying Gong)
# date： 2020/6/2 11:25  
# IDE：PyCharm 
# des: 计算样本间的距离
# input(s)：
# output(s)：
from sklearn import preprocessing
import numpy as np


def normalization(X):
    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    return X_minMax

# 计算一个点和所有点的欧式距离
def cal_point_2_point_Euclidean(X, Y):
    X_re = X.reshape(-1, 1, len(X[0]))
    X_new = np.tile(X_re, (1, len(X), 1))
    Y_re = Y.reshape(1, -1, len(Y[0]))
    Y_new = np.tile(Y_re, (len(Y), 1, 1))
    return np.sqrt(np.sum((X_new - Y_new)**2, axis=2))

# 欧式距离
def cal_euclidean(X, Y):
    return np.sqrt(np.sum((X - Y)**2, axis=1)).reshape(-1, 1)
