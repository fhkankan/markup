# XGBoost

XGBoost属于有监督学习，是Gradient Boosting模型的一种改进版

安装

```bash
cd Destop/
# 克隆XGBoost项目
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
# 复制编译配置文件
cp make/minimum.mk ./config.mk
make -j4
# 运行文件夹
cd python-package
sudo python setup.py install
# 检验
import xgboost as sgb
```

## 模型参数

```
三大类：General、Booster、Learning Task
```

- General Parameters

包括一些比较通用的模型配置

| 变量    | 说明                                                         |
| ------- | ------------------------------------------------------------ |
| booster | XGBoost使用的单模型，可以是gbtree或gblinear，其中gbtree用得比较多 |
| silent  | 设置为1，表示在模型训练过程中不打印提示信息，为0则打印，不设定时，默认为0 |
| nthread | 训练XGBoost模型时使用的线程数量，默认为当前最大可用数量      |

- Booster Parameters

包括和单模型相关的参数，因为使用gbtree作为booster的情况比较多，因此以下介绍当booster设置为gbtree时，对应的一些Booster Parameters,和决策树、随机森林等模型类似，会涉及一些和树结构相关的参数

| 变量             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| eta              | 但模型的学习率，默认初始化为0.3,一般经过多伦迭代后会多轮迭代后会逐渐衰减到0.01~0.2 |
| min_child_weight | 树种玫瑰子节点所需的最小权重和，默认为1                      |
| Max_depth        | 树的最大深度，默认为6层(不含根节点),一般设置为3~10层         |
| max_leaf_nodes   | 树中全部叶子节点的最大数量，默认为2^5,即一颗6层完全二叉树对应的叶子节点数量 |
| gamma            | 在树结构的训练过程中，将每个节点通过判断条件拆分成两个子节点时，所需的损失函数最小优化两，默认为0 |
| subsample        | 每颗树采样时所用得记录数量比例，默认为1，一般取0.5~1         |
| colsample_bytree | 每棵树采用时所用的特征数量比例，默认为1，一般0.5~1           |
| lambda           | 单模型的L2正则化项，默认为1                                  |
| alpha            | 单模型的L1正则化项，默认为1                                  |
| scale_pos_weight | 用于加快训练的收敛速度，默认为1                              |

- Learning Task Parameters

包括一些和模型训练相关的参数

| 变量        | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| objective   | 模型训练的目标函数，默认为reg:linear，还可为binary:logistic,multi:softmax,multi:softprob等 |
| eval_metric | 模型训练的误差函数，若进行回归，则默认为rmse即均方根误差，如进行分类，则默认为error即分类误差，其他可取值包括rmse、mae、logloss、merror、mlogloss、auc等 |
| seed        | 训练过程中涉及随机操作时所用的随机数种子，默认为0            |

