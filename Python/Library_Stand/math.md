# math

实现对浮点数的数学运算
一般是c语言中同名函数的简单封装

## 属性
```
math.e
# 自然常数

math.pi
# 圆周率pi
```
## 方法
```
math.ceil(x)
# 返回数字的上入整数，如math.ceil(4.1) 返回 5

math.floor(x)
# 返回数字的下舍整数，如math.floor(4.9)返回 4

math.trunc(x)
# 返回x的整数部分

math.modf(x)
# 返回x的小数和整数

math.fabs(x)
# 返回数字的绝对值，如math.fabs(-10) 返回10.0

math.fmod(x,y)
# 返回x%y

math.factorial(x)
# 返回x的阶乘

math.sqrt(x)
# 返回数字x的平方根

math.pow(x,y)
# 返回x的y次方

math.exp(x)
# 返回e的x次幂(e^x),如math.exp(1) 返回2.718281828459045

math.expm1(x)
# 返回e的x次幂(e^x)减1

math.copysign(x,y)
# 若y<0,返回-1乘以x的绝对值；否则，返回x的绝对值

math.ldexp(m,i)
# 返回m乘以2的i次方

math.log(x[,base])
# 返回x的以base为底的对数，base默认为e

math.log10(x)
# 返回以10为基数的x的对数，如math.log10(100)返回 2.0

math.hypot(x)
# 返回以x和y为直角边的斜边长
# 返回欧几里德范数 sqrt(x*x + y*y)
```


> 三角函数
```
math.degrees(x)
# 将弧度转换为角度,如degrees(math.pi/2) ， 返回90.0

math.radians(x)
# 将角度转换为弧度

sin(x)
# 返回的x弧度的正弦值

cos(x)
# 返回x的弧度的余弦值

tan(x)
# 返回x弧度的正切值

asin(x)
# 返回x的反正弦弧度值

acos(x)
# 返回x的反余弦弧度值

atan(x)
# 返回x的反正切弧度值

atan2(y, x)
# 返回给定的 X 及 Y 坐标值的反正切值
```






