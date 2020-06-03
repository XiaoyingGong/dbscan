# author: 龚潇颖(Xiaoying Gong)
# date： 2020/5/29 10:32  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import dbscan

X, _ = datasets.make_moons(500, noise=0.1, random_state=1)
# X = [[0.2, 0.3], [0.4, 0.5], [0.2, 0.5], [0.2, 0.4], [0.8, 0.8]]
print(X)
# eps为邻域半径，min_samples为最少点数目
core_samples, cluster_ids = dbscan(X, eps = 0.2, min_samples=20)
# core_samples, cluster_ids = dbscan(X, eps = 0.21, min_samples=2)
# cluster_ids中-1表示对应的点为噪声点

df = pd.DataFrame(np.c_[X, cluster_ids], columns = ['feature1','feature2','cluster_id'])
df['cluster_id'] = df['cluster_id'].astype('i2')

df.plot.scatter('feature1','feature2', s=100,
                c=list(df['cluster_id']), cmap='rainbow', colorbar=False,
                alpha=0.6, title='sklearn DBSCAN cluster result')
plt.show()