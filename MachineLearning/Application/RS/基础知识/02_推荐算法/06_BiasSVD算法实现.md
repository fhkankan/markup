## 基于矩阵分解的CF算法实现（二）：BiasSvd

BiasSvd其实就是前面提到的Funk SVD矩阵分解基础上加上了偏置项。

#### BiasSvd

利用BiasSvd预测用户对物品的评分，$k$表示隐含特征数量：
$$
\begin{split}
\hat {r}_{ui} &=\mu + b_u + b_i + \vec {p_{uk}}\cdot \vec {q_{ki}}
\\&=\mu + b_u + b_i + {\sum_{k=1}}^k p_{uk}q_{ik}
\end{split}
$$

#### 损失函数

同样对于评分预测我们利用平方差来构建损失函数：
$$
\begin{split}
Cost &= \sum_{u,i\in R} (r_{ui}-\hat{r}_{ui})^2
\\&=\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i -{\sum_{k=1}}^k p_{uk}q_{ik})^2
\end{split}
$$
加入L2正则化：
$$
Cost = \sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})^2 + \lambda(\sum_U{b_u}^2+\sum_I{b_i}^2+\sum_U{p_{uk}}^2+\sum_I{q_{ik}}^2)
$$
对损失函数求偏导：
$$
\begin{split}
\cfrac {\partial}{\partial p_{uk}}Cost &= \cfrac {\partial}{\partial p_{uk}}[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})^2 + \lambda(\sum_U{b_u}^2+\sum_I{b_i}^2+\sum_U{p_{uk}}^2+\sum_I{q_{ik}}^2)]
\\&=2\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})(-q_{ik}) + 2\lambda p_{uk}
\\\\
\cfrac {\partial}{\partial q_{ik}}Cost &= \cfrac {\partial}{\partial q_{ik}}[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})^2 + \lambda(\sum_U{b_u}^2+\sum_I{b_i}^2+\sum_U{p_{uk}}^2+\sum_I{q_{ik}}^2)]
\\&=2\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})(-p_{uk}) + 2\lambda q_{ik}
\end{split}
$$

$$
\begin{split}
\cfrac {\partial}{\partial b_u}Cost &= \cfrac {\partial}{\partial b_u}[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})^2 + \lambda(\sum_U{b_u}^2+\sum_I{b_i}^2+\sum_U{p_{uk}}^2+\sum_I{q_{ik}}^2)]
\\&=2\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})(-1) + 2\lambda b_u
\\\\
\cfrac {\partial}{\partial b_i}Cost &= \cfrac {\partial}{\partial b_i}[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})^2 + \lambda(\sum_U{b_u}^2+\sum_I{b_i}^2+\sum_U{p_{uk}}^2+\sum_I{q_{ik}}^2)]
\\&=2\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})(-1) + 2\lambda b_i
\end{split}
$$



#### 随机梯度下降法优化

梯度下降更新参数$p_{uk}$：
$$
\begin{split}
p_{uk}&:=p_{uk} - \alpha\cfrac {\partial}{\partial p_{uk}}Cost
\\&:=p_{uk}-\alpha [2\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})(-q_{ik}) + 2\lambda p_{uk}]
\\&:=p_{uk}+\alpha [\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})q_{ik} - \lambda p_{uk}]
\end{split}
$$
 同理：
$$
\begin{split}
q_{ik}&:=q_{ik} + \alpha[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})p_{uk} - \lambda q_{ik}]
\end{split}
$$

$$
b_u:=b_u + \alpha[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik}) - \lambda b_u]
$$

$$
b_i:=b_i + \alpha[\sum_{u,i\in R} (r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik}) - \lambda b_i]
$$

**随机梯度下降：**
$$
\begin{split}
&p_{uk}:=p_{uk}+\alpha [(r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})q_{ik} - \lambda_1 p_{uk}]
\\&q_{ik}:=q_{ik} + \alpha[(r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik})p_{uk} - \lambda_2 q_{ik}]
\end{split}
$$

$$
b_u:=b_u + \alpha[(r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik}) - \lambda_3 b_u]
$$

$$
b_i:=b_i + \alpha[(r_{ui}-\mu - b_u - b_i-{\sum_{k=1}}^k p_{uk}q_{ik}) - \lambda_4 b_i]
$$

由于P矩阵和Q矩阵是两个不同的矩阵，通常分别采取不同的正则参数，如$\lambda_1$和$\lambda_2$

**算法实现**

```python
'''
BiasSvd Model
'''
import math
import random
import pandas as pd
import numpy as np

class BiasSvd(object):

    def __init__(self, alpha, reg_p, reg_q, reg_bu, reg_bi, number_LatentFactors=10, number_epochs=10, columns=["uid", "iid", "rating"]):
        self.alpha = alpha # 学习率
        self.reg_p = reg_p
        self.reg_q = reg_q
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs
        self.columns = columns

    def fit(self, dataset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q, self.bu, self.bi = self.sgd()

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()

        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))

        for i in range(self.number_epochs):
            print("iter%d"%i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float32(r_ui - self.globalMean - bu[uid] - bi[iid] - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)
                
                P[uid] = v_pu 
                Q[iid] = v_qi
                
                bu[uid] += self.alpha * (err - self.reg_bu * bu[uid])
                bi[iid] += self.alpha * (err - self.reg_bi * bi[iid])

                error_list.append(err ** 2)
            print(np.sqrt(np.mean(error_list)))

        return P, Q, bu, bi

    def predict(self, uid, iid):

        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]

        return self.globalMean + self.bu[uid] + self.bi[iid] + np.dot(p_u, q_i)


if __name__ == '__main__':
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv("datasets/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

    bsvd = BiasSvd(0.02, 0.01, 0.01, 0.01, 0.01, 10, 20)
    bsvd.fit(dataset)

    while True:
        uid = input("uid: ")
        iid = input("iid: ")
        print(bsvd.predict(int(uid), int(iid)))

```

