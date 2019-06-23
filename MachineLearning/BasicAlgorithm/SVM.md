# SVM

Support Vector Machine

SVM尝试寻找一个最优的决策边界

距离两个类别的最近的样本距离最远

SVM要最大化margin=2d

有Hard Margin SVM(数据线性可分)和Soft Margin SVM两种

## 原理

### Hard Margin SVM

$(x, y)$到$Ax + By +C = 0$的距离
$$
\frac{|Ax+By+C|}{\sqrt{A^2+B^2}}
$$
拓展到n维，点到$\theta^Tx_b = 0$直线，其中直线可表示为
$$
w^Tx + b = 0
$$

则距离为
$$
\frac{|w^Tx + b|}{||w||}
$$
其中
$$
||w|| = \sqrt{w_1^2+w_2^2+\ldots+w_n^2}
$$
可得到
$$
\begin{cases}
 \frac{|w^Tx^{(i)} + b|}{||w||}\geq{d} & \forall{y^{(i)}}=1 \\
 \frac{|w^Tx^{(i)} + b|}{||w||}\leq{-d} & \forall{y^{(i)}}=-1
 \end{cases}
$$
变形
$$
\begin{cases}
 \frac{|w^Tx^{(i)} + b|}{||w||d}\geq{1} & \forall{y^{(i)}}=1 \\
 \frac{|w^Tx^{(i)} + b|}{||w||d}\leq{-1} & \forall{y^{(i)}}=-1
 \end{cases}
$$
转换
$$
\begin{cases}
 {w_d^Tx^{(i)} + b_d}\geq{1} & \forall{y^{(i)}}=1 \\
 {w_d^Tx^{(i)} + b_d}\leq{-1} & \forall{y^{(i)}}=-1
 \end{cases}
$$
则三条直线分别为
$$
w_d^Tx + b_d = 1 \\
w_d^Tx + b_d = 0 \\
w_d^Tx + b_d = -1
$$
为了便于书写
$$
w^Tx + b = 1 \\
w^Tx + b = 0 \\
w^Tx + b = -1
$$

$$
\begin{cases}
 {w^Tx^{(i)} + b}\geq{1} & \forall{y^{(i)}}=1 \\
 {w^Tx^{(i)} + b}\leq{-1} & \forall{y^{(i)}}=-1
 \end{cases}
$$

可转化为
$$
y^{(i)}(w^Tx^{(i)} + b)\geq{1}
$$
对于任意支撑向量x
$$
max\frac{|w^Tx + b|}{||w||} \\
max\frac{1}{||w||} \\
min{||w||}  \\
min\frac{1}{2}||w||^2
$$
可以得到有条件的最优化问题
$$
min\frac{1}{2}||w||^2 \\
st.y^{(i)}(w^Tx^{(i)} + b)\geq{1}
$$

### Soft Margin SVM

C越小，容错空间越大，C越大，容错空间越小

L1正则
$$
min\frac{1}{2}||w||^2 + C\sum_{i=1}^m\zeta_i\\
st.y^{(i)}(w^Tx^{(i)} + b)\geq{1}-\zeta_i \\
\zeta_i\geq{0}
$$
  L2正则
$$
min\frac{1}{2}||w||^2 + C\sum_{i=1}^m\zeta_i^2\\
st.y^{(i)}(w^Tx^{(i)} + b)\geq{1}-\zeta_i \\
\zeta_i\geq{0}
$$
转换为
$$
max\sum_{i=1}^m\alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{i=1}^m\alpha_i\alpha_jy_iy_jx_ix_j\\
st. 0\leq\alpha_i\leq{C} \\
\sum_{i=1}^m\alpha_iy_i = 0
$$
使用核函数技巧$K(x, y)=x'\cdot{y}'$
$$
max\sum_{i=1}^m\alpha_i - \frac{1}{2}\sum_{i=1}^m\sum_{i=1}^m\alpha_i\alpha_jy_iy_jK(x_i, x_j)\\
st. 0\leq\alpha_i\leq{C} \\
\sum_{i=1}^m\alpha_iy_i = 0
$$

### 核函数

二次多项式核函数
$$
K(x, y) = (x\cdot{y}+1)^2 \\
K(x, y) = (\sum_{i=1}^n{x_iy_j + 1})^2
$$
扩展多元多项式核函数
$$
K(x, y) = (x\cdot{y}+c)^d \\
$$
线性核函数
$$
K(x, y) = x\cdot{y}\\
$$
高斯核函数(RBF核)
$$
K(x, y) = e^{-\gamma||x-y||^2}
$$

## sklearn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.processing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

X = iris.data
y = iris.target
# 二分类
X = X[y<2, :2]
y = y[y<2]

plt.scatter(X[y==0, 1], X[y==0, 1], color='red')
plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
plt.show()

standardScaler = StandaedScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)

svc = LinearSVC(C=1e9)
svc.fit(X_standard, y)

# 决策边界
def plot_decision_boundary(model, axis):
  	x0, x1 = np.meshgrid(
    		np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape()
      	np.linspace(axis[2], axisp3), int((axis[3]-axis[2])*100)).reshape()
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(["#EF9A9A", "#FF59D", "#90CAF9"])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1], color="red")
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1], color="blue")
plt.show()

# 调整C，C越小容错越大
svc2 = LinearSVC(C=0.01)
svc2.fit(X_standard, y)

plot_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1], color="red")
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1], color="blue")
plt.show()

svc.coef_  # 两个特征对应两个系数，支持多类
svc.intercept_  # 截距


def plot_svc_decision_boundary(model, axis):
  	x0, x1 = np.meshgrid(
    		np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1)
      	np.linspace(axis[2], axisp3), int((axis[3]-axis[2])*100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(["#EF9A9A", "#FF59D", "#90CAF9"])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
    w = model.coef_[0]
    b = model.intercept_[0]
    
    # w0 * x0 + w1 * x1 + b = 0
    # x1 = -w0 / w1 * x0 - b / w1
    plt_x = np.linspace(axis[0], axis[1], 200)
		up_y = -w[0]/w[1] * plot_x - b / w[1] + 1 / w[1]
    down_y = -w[0]/w[1] * plot_x - b / w[1] - 1 / w[1]
    
    up_index = (up_y >=axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plt_x[down_index], down_y[down_index], color='black')
    
    
plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1], color="red")
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1], color="blue")
plt.show()

plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1], color="red")
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1], color="blue")
plt.show()

```

## 多项式特征

依靠升维使得原本线性不可分的数据线性可分

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.processing import PolynomialFeatures
from sklearn.processing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

X, y = datasets.make_moons()

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

X, y = datasets.make_moons(noise=0.15, random_state=666)

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

# 使用多项式特征的SVM
def PolynomialSVC(degree, C=1.0):
  	return Pipeline([
      	("poly", PolynomialFeatures(degree=degree)),
      	("std_scaler", StandardScaler()),
      	("linearSVC", LinearSVC(C=C))
    ])
  
poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y)

# 决策边界
def plot_decision_boundary(model, axis):
  	x0, x1 = np.meshgrid(
    		np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1)
      	np.linspace(axis[2], axisp3), int((axis[3]-axis[2])*100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(["#EF9A9A", "#FF59D", "#90CAF9"])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1])
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1])
plt.show()  

```

## 多项式核函数

```python
from sklearn.svm import SVC

def PolynomialKernelSVC(degree, C=1.0):
  	return Pipeline([
      	("std_scaler", StandardScaler()),
      	("kenelSVC", SVC(kenel="poly", degree, C=C))
    ])
  
poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X, y)

# 决策边界
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X_standard[y==0, 0], X_standard[y==0, 1])
plt.scatter(X_standard[y==1, 0], X_standard[y==1, 1])
plt.show() 
```

## 高斯核函数

将一个样本点映射到一个无穷维的特征空间

- 升维

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arrange(-4, 5, 1)
y = np.array((x >= 2) & (x <= 2), dtype='int')

plt.scatter(x[y==0], [0]*len(x[y==0]))
plt.scatter(x[y==1], [0]*len(x[y==1]))
plt.show()

# 高斯核函数将一维数据映射到二维空间
def gaussian(x, l):
  	gamma = 1.0
    return np.exp(-gamma *  (x-l) ** 2)
  
l1, l2 = -1, 1
X_new = np.empty((len(x), 2))
for i, data in enumerate(x):
  	X_new[i, 0] = gaussian(data, l1)
    X_new[i, 1] = gaussian(data, l2)

plt.scatter(X_new[y==0, 0], X_new[y==0, 1])
plt.scatter(X_new[y==1, 0], X_new[y==1, 1])
plt.show()  
```

- sklearn

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

X, y = datasets.make_moons(noise=0.15, random_state=666)

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()

def RBFKernelSVC(gamma=1.0):
  	return Pipeline([
      	("std_scaler", StandardScaler()),
      	("svc", SVC(kernel="rbf", gamma=gamma))
    ])
  
svc = RBFKernelSVC(gamma=1.0)
svc.fit(X, y)

# 决策边界
def plot_decision_boundary(model, axis):
  	x0, x1 = np.meshgrid(
    		np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1)
      	np.linspace(axis[2], axisp3), int((axis[3]-axis[2])*100)).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(["#EF9A9A", "#FF59D", "#90CAF9"])
    
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
plot_decision_boundary(svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show() 

# 调整gamma
# gamma越大，覆盖区域越窄
svc_gamma_100 = RBFKernelSVC(gamma=100)
svc_gamma_100.fit(X, y)
plot_decision_boundary(svc_gamma_100, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show() 

svc_gamma_10 = RBFKernelSVC(gamma=10)
svc_gamma_10.fit(X, y)
plot_decision_boundary(svc_gamma_10, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show() 

svc_gamma_01 = RBFKernelSVC(gamma=0.1)
svc_gamma_01.fit(X, y)
plot_decision_boundary(svc_gamma_01, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.show()
```

## SVM思想解决回归问题

margin中尽可能多地包含样本

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_seletcion import train_test_split
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

boston = datasets.load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

def StandardLinearSVR(epsilon=0.1):
  	return Pipeline([
      	("std_scaler", StandardScaler()),
      	("linearSVR", LinearSVR(epsilon=epsilon))  # 存在多个超参数
    ])
  
svr = StandardLinearSVR()
svr.fit(X_train, y_train)
svr.score(X_test, y_test)
```



