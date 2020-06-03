# author: 龚潇颖(Xiaoying Gong)
# date： 2020/6/3 14:42  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt


X,_ = datasets.make_moons(500, noise = 0.1, random_state=1)
df = pd.DataFrame(X,columns = ['feature1','feature2'])
df.head()


# ------------------------- 计算样本点中两两之间的距离 -------------------------
dfdata = df.copy()
dfdata.columns = ["feature1_a", "feature2_a"]
dfdata["id_a"] = range(1, len(df)+1)
dfdata = dfdata.set_index("id_a", drop=False)
dfdata = dfdata.reindex(columns=["id_a",
                                 "feature1_a",
                                 "feature2_a"])
dfdata.head()
dfpairs = pd.DataFrame(columns = ["id_a","id_b","distance_ab"])
q = dfdata.loc[:,["feature1_a","feature2_a"]].values

for i in dfdata.index:
    p = dfdata.loc[i,["feature1_a","feature2_a"]].values
    dfab = dfdata[["id_a"]].copy()
    dfab["id_b"] = i
    dfab["distance_ab"] = np.sqrt(np.sum((p -q )**2,axis = 1))
    dfpairs = pd.concat([dfpairs,dfab])

dfpairs.head()
print("------------------------- 计算样本点中两两之间的距离 -------------------------")

# ------------------------- 构造临时聚类簇 -------------------------
dfnears = dfpairs.query("distance_ab<0.2")
dfglobs = dfnears.groupby("id_a").agg({"id_b":[len,set]})
dfglobs.columns = ["neighbours_cnt","neighbours"]
dfglobs = dfglobs.query("neighbours_cnt>=20")

dfglobs = dfglobs.reset_index()

# 找到核心点id
core_ids = set(dfglobs["id_a"])
dfcores = dfglobs.copy()

# 剔除非核心点的id
dfcores["neighbours"] = [x & core_ids for x in dfcores["neighbours"]]
dfcores.head()

# ------------------------- 合并相连的临时聚类簇得到聚类簇 -------------------------
def mergeSets(set_list):
    result = []
    while len(set_list) > 0:
        cur_set = set_list.pop(0)
        intersect_idxs = [i for i in list(range(len(set_list) - 1, -1, -1)) if cur_set & set_list[i]]
        while intersect_idxs:
            for idx in intersect_idxs:
                cur_set = cur_set | set_list[idx]

            for idx in intersect_idxs:
                set_list.pop(idx)

            intersect_idxs = [i for i in list(range(len(set_list) - 1, -1, -1)) if cur_set & set_list[i]]

        result = result + [cur_set]
    return result

set_list = list(dfcores["neighbours"])
result = mergeSets(set_list)

core_clusters = {i:s for i,s in enumerate(result)}
print(core_clusters)
# ------------------------- 关联所有样本点 -------------------------
core_map = {}
for k,v in core_clusters.items():
    core_map.update({vi:k for vi in v})

cluster_map = {}
for i in range(len(dfglobs)):
    id_a = dfglobs["id_a"][i]
    neighbours = dfglobs["neighbours"][i]
    cluster_map.update({idx:core_map[id_a] for idx in neighbours})

print(len(cluster_map))

dfdata["cluster_id"] = [cluster_map.get(id_a,-1) for id_a in dfdata["id_a"]]
dfdata.head()

dfdata.plot.scatter('feature1_a','feature2_a', s = 100,
    c = list(dfdata['cluster_id']),cmap = 'rainbow',colorbar = False,
    alpha = 0.6,title = 'Hands DBSCAN Cluster Result ')
plt.show()