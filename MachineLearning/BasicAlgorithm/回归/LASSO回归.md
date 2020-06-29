# LASSO Regression

## 原理

损失函数
$$
J(\theta) = MSE(y, \hat{y}; \theta) + \alpha\sum_{i=1}^n{|\theta_i|}
$$
使$J(\theta)$和$\theta$尽量小，采用了模型正则化，$\alpha$为超参数

LASSO趋向于使得一部分theta值变为0，可作为特征选择用。

缺点：可能将有效特征消除

## 实现

## sklearn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

np.random.seed(42)
x = np.random.uniform(-3.0, 3.0, size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + 3 + np.random.normal(0, 1, size=100)

plt.scatter(x, y)
plt.show()

np.ramdom.seed(666)
X_train, X_test, y_train, y_test = train_split_test(X, y)

# 多项式回归，过拟合
def PolynomialRegression(degree):
  	return Pipeline([
      	("poly", PolynomialFeatures(degree=degree)),
      	("std_scaler", StandardScaller()),
      	("lin_reg", LinearRegression())
    ])

poly20_reg = PolynomialRegression(degree=20)
poly20_reg.fit(X, y)
y20_predict = poly20_reg.predict(X)
error = mean_squared_error(y, y20_predict)
print(error)

def plot_model(model):
  	X_plot = np.linspace(-3, 3, 100).reshpe(100, 1)
		y_plot = model.predict(X_plot)
		plt.scatter(x, y)
		plt.plot(X_plot[:, 0], y_plot, color='r')
    plt.axis([-3, 3, 0, 6])
		plt.show()

plot_model(poly20_reg)

# LASSO回归
def LassoRegression(degree, alpha):
  	return Pipeline([
      	("poly", PolynomialFeatures(degree=degree)),
      	("std_scaler", StandardScaller()),
      	("ridge_reg", Lasso(alpha=alpha))
    ])

lasso1_reg = LassoRegression(20, 0.001)
lasso1_reg.fit(X_train, y_train)
y1_predict = lasso1_reg.predict(X_test)
error = mean_squared_error(y_test, y1_predict)
print(error)

plot_model(lasso1_reg)

# 改变alpha
lasso2_reg = LassoRegression(20, 0.1)
lasso2_reg.fit(X_train, y_train)
y2_predict = lasso2_reg.predict(X_test)
error = mean_squared_error(y_test, y2_predict)
print(error)

plot_model(lasso2_reg)

lasso3_reg = LassoRegression(20, 1)
lasso3_reg.fit(X_train, y_train)
y3_predict = lasso3_reg.predict(X_test)
error = mean_squared_error(y_test, y3_predict)
print(error)

plot_model(lasso2_reg)
```

