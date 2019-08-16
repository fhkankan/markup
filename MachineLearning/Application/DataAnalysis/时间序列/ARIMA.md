# ARIMA

时间序列模型

## 原理

- 平稳性

```
平稳性就是要求经由样本时间序列所得到的拟合曲线在未来的一段时间内仍能顺着现有的形态"惯性"地延续下去

要求序列的均值和方差不发生明显变化
```

严平稳

```
严平稳表示的分布不随时间的改变而改变。如：白噪声(正态)，无论怎么取，都是期望为0，方差为1
```

若平稳

```
期望与相关系数(依赖性)不变。未来某时刻的t的值X_t就要依赖于它的过去信息，所以需要依赖性
```

- 差分法

时间序列在t与t-1时刻的插值

作用：可解决数据波动过大

- 自回归模型(AR)

```
- 描述当前值与历史值之间的关系，用自身的历史数据对自身进行预则
- 自回归模型必须满足平稳性的要求
- 必须具有自相关性，若果自相关系数($\varphi_i$)小于0.5，则不宜采用
- 自回归只适用于预测与自身前期相关的现象
```

p阶自回归过程的公式定义:
$$
y_t = \mu + \sum_{i=1}^p{\gamma_iy_{t-i}}+\epsilon_t
$$
其中，$y_t$是当前值，$\mu$是常数项，$p$是阶数，$\gamma_i$是自相关系数，$\epsilon_t$是误差

- 移动平均模型(MA)

```
- 移动平均模型关注的是自回归模型中的误差项的累加
- 移动平均法能有效地消除预测中的随机波动
```

q阶自回归过程的公式定义：
$$
y_t = \mu + \sum_{i=1}^q{\theta_i\epsilon_{t-i}}+\epsilon_t
$$

- 自回归移动平均模型(ARMA)

```
自回归与移动平均的结合
```

公式定义
$$
y_t = \mu + \sum_{i=1}^p{\gamma_iy_{t-i}} +\sum_{i=1}^q{\theta_i\epsilon_{t-i}}+\epsilon_t
$$

- ARIMA(p,d,q)模型

全称是差分自回归移动平均模型`Autoregressive Integrated Moving Average Model`

AR为自回归，p为自回归项，MA为移动平均，q为移动平均项数，d为时间序列成为平稳时所做的差分次数

原理

```
将非平稳时间序列转化为平稳时间序列然后将因变量仅对它的滞后值以及随机误差项的现值和滞后值进行回归所建立的模型
```

- 自相关函数ACF(`autocorrelation function`)

```
有序的随机变量序列与其自身相比较
自相关函数反映了同一序列在不同时序的取值之间的相关性
```

公式
$$
ACF(k) = \rho_k = \frac{Cov(y_t,y_{t-k})}{Var(y_t)}
$$
其中，$\rho_k$的取值范围为`[-1, 1]`

- 偏自相关函数PACF(`partial autocorrelation function`)

```python
- 对于一个平稳AR(p)模型，求出滞后k自相关系数p(k)时，实际上的得到并不是x(t)与x(t-k)之间单纯的相关关系
- x(t)同时还会受到中间k-1个随机变量x(t-1),x(t-2),...,x(t-k+1)的影响，而这k-1个随机变量又都和x(t-k)具有相关关系，所以自相关系数p(k)里实际参杂了其他变量对x(t)与x(t-k)的影响
- PACF就是剔除了中间k-1个随机变量x(t-1),x(t-2),...,x(t-k+1)的干扰之后,x(t-k)对x(t)影响的相关程度
- ACF还包含了其他变量的影响，而偏自相关系数PACF是严格这两个变量之间的相关性
```

## 构建

- 建模流程

```
1. 将序列平稳(差分法确定d)
2. p和q阶数确定(ACF/PACF)
3. ARIMA(p,d,q)
```

- ARIMA(p,d,q)阶数确定

截尾：落在置信区间内(95%的点都符合该规则)

| 模型      | ACF                             | PACF                            |
| --------- | ------------------------------- | ------------------------------- |
| AR(p)     | 衰减趋于零(几何型或振荡型)      | P阶后截尾                       |
| MA(q)     | q阶后截尾                       | 衰减趋于零(几何型或振荡型)      |
| ARMA(p,q) | q阶后衰减趋于零(几何型或振荡型) | p阶后衰减趋于零(几何型或振荡型) |

阶数确定：AR(p)看PACF，MA(q)看ACF

## 评估

- 参数选择参考

模型选择AIC和BIC：选择更简单的模型

AIC：赤池信息准则(Akaike Information Criterion)
$$
AIC= 2k-2\ln(L)
$$
BIC：贝叶斯信息准则(Bayesian Information Criterion)
$$
BIC = k\ln(n) - 2\ln(L)
$$
其中，k为模型参数个数，n为样本数量，L为似然函数

- 检测

模型残差检验：

ARIMA模型的残差是否是平均值为0且方差为常数的正态分布

QQ图：线性即正态分布

## 实现

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline
%config InlineBackend.figure_format='retina'

from __future__ import absolute_import, division, print_function
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#xgboost
import sys
import os

import pandas as pd
import numpy as np

# # Remote Data Access
# import pandas_datareader.data as web
# import datetime
# # reference: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')


#Read the data
#美国消费者信心指数
Sentiment = 'data/sentiment.csv'
Sentiment = pd.read_csv(Sentiment, index_col=0, parse_dates=[0])

# Select the series from 2005 - 2016
sentiment_short = Sentiment.loc['2005':'2016']
# 折线图显示数据波动
sentiment_short.plot(figsize=(12,8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("Consumer Sentiment")
sns.despine()

# 差分确定d
sentiment_short['diff_1'] = sentiment_short['UMCSENT'].diff(1)  # 一阶差分
sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)  # 二阶差分
sentiment_short.plot(subplots=True, figsize=(18, 12))


# ACF和PACF确定p,q
fig = plt.figure(figsize=(12,8))
# 自相关
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sentiment_short, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout();
# 偏自相关
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sentiment_short, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout();

# 散点图也可以表示
lags=9
ncols=3
nrows=int(np.ceil(lags/ncols))
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4*ncols, 4*nrows))
for ax, lag in zip(axes.flat, np.arange(1,lags+1, 1)):
    lag_str = 't-{}'.format(lag)
    X = (pd.concat([sentiment_short, sentiment_short.shift(-lag)], axis=1,
                   keys=['y'] + [lag_str]).dropna())

    X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
    corr = X.corr().as_matrix()[0][1]
    ax.set_ylabel('Original')
    ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
    ax.set_aspect('equal');
    sns.despine();
fig.tight_layout();

# 更直观一些
def tsplot(y, lags=None, title='', figsize=(14, 8)):   
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
tsplot(sentiment_short, title='Consumer Sentiment', lags=36);
```

参数选择

```

```

## 回归



## 分类

