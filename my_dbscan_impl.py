# author: 龚潇颖(Xiaoying Gong)
# date： 2020/6/3 15:30  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import cal_dist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
# 配置文件
# 搜索的半径
eps = 0.2
# MinPts
minPts = 20
# ------------------------- 生成样本点 -------------------------
X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
# print(X)

# ------------------------- 计算样本点间的距离 -------------------------
sample_dist = cal_dist.cal_point_2_point_Euclidean(X, X)

# ------------------------ 找核心点 --------------------------
# sample_sort = np.sort(sample_dist, axis=0)
# dist_upper_bound = sample_sort[minPts, :]
# 找出每一个点的领域个数 即 小于半径的个数
eps_neighbors = np.argwhere(sample_dist < eps)
# 统计各个点的领域 以及 领域个数
eps_neighbors_2_pd = pd.DataFrame(eps_neighbors, columns=['core_id', 'neighbors_id'])
eps_neighbors_2_pd_groupby = eps_neighbors_2_pd.groupby("neighbors_id").agg({"core_id": [len, set]})
eps_neighbors_2_pd_groupby.columns = ["neighbors_cnt","neighbors"]

neighbors_cnt = np.array(eps_neighbors_2_pd_groupby["neighbors_cnt"])
neighbors = np.array(eps_neighbors_2_pd_groupby["neighbors"])
print(neighbors_cnt)
# 找到核心点的下标
core_points_idnex = np.argwhere(neighbors_cnt >= minPts).flatten()

# 和并临时的核心点，给出聚类簇
