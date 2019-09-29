# Statsmodels

statsmodels提供了在python环境中进行探索数据分析、统计检验以及统计模型估计的累和函数。其主要功能包括：线性回归、广义线性模型、广义估计方程、稳健线性模型、线性混合效应模型、离散因变量回归、方差分析、时间序列分析、生存分析、统计检验、非参数检验、非参数计量、广义矩方法、经验似然法、计数模型、常用分布等。此外还可以绘制你和曲线图、盒须图、相关图、时间序列图、因子图、马赛克图等用于探索性数据分析和模型构建诊断的常用图形。

statsmodels可供记性科学计算和统计分析的模块非常多，每个模块下包含的方法或函数也极其繁杂。因此，调用stasmodels进行数据分析时，常常使用其数据分析借口(api)方式来进行

```python
import statsmodels.api as sm
```

若数据分析中对中文进行处理产生乱码，可以先执行如下代码将有关信息重定位

```python
stdout=sys.stdout
stdin=sys.stdin
stderr=sys.stderr
reload(sys)
sys.stdout=stdout
sys.stdin=stdin
sys.stderr=stderr
sys.setdefaultencoding('utf-8')
```

