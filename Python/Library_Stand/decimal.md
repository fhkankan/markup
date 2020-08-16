# decimal

## 概述

可用于金额的计算

```
1. 提供十进制数据类型，并且存储为十进制数序列；
2. 有界精度：用于存储数字的位数是固定的，可以通过decimal.getcontext（）.prec=x 来设定，不同的数字可以有不同的精度
3. 浮点：十进制小数点的位置不固定（但位数是固定的）
```

## 构建

可以通过整数、字符串或者元组构建decimal.Decimal，对于浮点数需要先将其转换为字符串

```
# 浮点数据不准确
Decimal(5.55)*100  # Decimal('554.9999999999999822364316060')

# 浮点数据转换
from decimal import *
Decimal.from_float(22.222)  # 结果为Decimal('22.2219999999999995310417943983338773250579833984375')

# 字符串
Decimal('5.55')*100  # Decimal('555.00')


```

## context

decimal在一个独立的context下工作，可以通过getcontext来获取当前环境。例如前面提到的可以通过decimal.getcontext().prec来设定小数点精度（默认为28）

```python
from  decimal import Decimal
from  decimal import getcontext

d_context = getcontext()
d_context.prec = 6  # 设定六位有效数字
print(d_context)

d = Decimal(1) / Decimal(3)
print(type(d), d)
```

## quantize

四舍五入，保留几位小数

```
from decimal import *
Decimal('50.5679').quantize(Decimal('0.00'))
# 结果为Decimal('50.57')，结果四舍五入保留了两位小数
```

Decimal 结果转化为string

```
from decimal import *
str(Decimal('3.40').quantize(Decimal('0.0')))
# 结果为'3.40'，字符串类型
```

常用参数

[参考](https://blog.csdn.net/weixin_37989267/article/details/79473706)

```
其实这里我们通过上面一组例子可以发现，正数的行为非常可预期也非常简单，负数的情况稍复杂，有些函数就是设计为负数在某些情况中使用的。正数中无法重现的ROUND_DOWN和ROUND_FLOOR的区别，ROUND_DOWN是无论后面是否大于5都不会管保持原状，而Floor在正数中的行为也是如此，但是在负数中为了倾向无穷小，所以无论是否大于5，他都会变得更小而进位。反而ROUND_UP和ROUND_DOWN的行为是最可预期的，那就是无论后面数大小，UP就进位，DOWN就始终不进位。
```

- 一组负数的后一位超过5的数据

```python
from decimal import *

x = Decimal('-3.333333333') + Decimal('-2.222222222')
print(x)   # -5.555555555
print(x.quantize(Decimal('1.0000'), ROUND_HALF_EVEN))    # -5.5556
print(x.quantize(Decimal('1.0000'), ROUND_HALF_DOWN))    # -5.5556
print(x.quantize(Decimal('1.0000'), ROUND_CEILING))      # -5.5555
print(x.quantize(Decimal('1.0000'), ROUND_FLOOR))        # -5.8599
print(x.quantize(Decimal('1.0000'), ROUND_UP))           # -5.8599
print(x.quantize(Decimal('1.0000'), ROUND_DOWN))         # -5.5555
```

说明

```
ROUND_HALF_EVENT 和 ROUND_HALF_DOWN：EVENT是quansize的默认设置值，可以通过getcontext()得到，EVENT四舍五入进了一位，DOWN为接近最近的0进了一位。

ROUND_CEILING 和 ROUND_FLOOR：CEILING超过5没有进位是因为它倾向正无穷，FLOOR为了总是变得更小所以进了一位。

ROUND_UP 和 ROUND_DOWN：UP始终进位，DOWN始终不会进位。。
```

- 一组后一位没有超过5的数据

```python
from decimal import *

x = Decimal('-3.333333333') + Decimal('-1.111111111')
print(x)   # 4.444444444
print(x.quantize(Decimal('1.0000'), ROUND_HALF_EVEN))    # -4.4444
print(x.quantize(Decimal('1.0000'), ROUND_HALF_DOWN))    # -4.4444
print(x.quantize(Decimal('1.0000'), ROUND_CEILING))      # -4.4444
print(x.quantize(Decimal('1.0000'), ROUND_FLOOR))        # -4.4445
print(x.quantize(Decimal('1.0000'), ROUND_UP))           # -4.4445
print(x.quantize(Decimal('1.0000'), ROUND_DOWN))         # -4.4444
```

说明

```
ROUND_HALF_EVENT 和 ROUND_HALF_DOWN：EVENT是quansize的默认设置值，可以通过getcontext()得到，EVENT由于达不到四舍五入所以不进位，DOWN同样也不进位。

ROUND_CEILING 和 ROUND_FLOOR：CEILING倾向正无穷不进位，FLOOR即使没有超过5，但是为了总是变得更小进了一位。

ROUND_UP 和 ROUND_DOWN：UP始终进位，DOWN始终不会进位。。
```

- 正数部分后面数大于5的情况

```
from decimal import *

x = Decimal('3.333333333') + Decimal('2.222222222')
print(x)   # 5.555555555
print(x.quantize(Decimal('1.0000'), ROUND_HALF_EVEN))    # 5.5556
print(x.quantize(Decimal('1.0000'), ROUND_HALF_DOWN))    # 5.5556
print(x.quantize(Decimal('1.0000'), ROUND_CEILING))      # 5.5556
print(x.quantize(Decimal('1.0000'), ROUND_FLOOR))        # 5.5555
print(x.quantize(Decimal('1.0000'), ROUND_UP))           # 5.5556
print(x.quantize(Decimal('1.0000'), ROUND_DOWN))         # 5.5555
```

说明

```
ROUND_HALF_EVENT 和 ROUND_HALF_DOWN：EVENT是quansize的默认设置值，可以通过getcontext()得到，EVENT由于达到四舍五入所以进位，DOWN同样进位。

ROUND_CEILING 和 ROUND_FLOOR：CEILING正数始终进位，FLOOR在正数则始终不会进位。

ROUND_UP 和 ROUND_DOWN：UP始终进位，DOWN始终不会进位。
```

- 正数部分后面数小于5的情况

```
from decimal import *

x = Decimal('3.333333333') + Decimal('1.111111111')
print(x)   # 4.444444444
print(x.quantize(Decimal('1.0000'), ROUND_HALF_EVEN))    # 4.4444
print(x.quantize(Decimal('1.0000'), ROUND_HALF_DOWN))    # 4.4444
print(x.quantize(Decimal('1.0000'), ROUND_CEILING))      # 4.4445
print(x.quantize(Decimal('1.0000'), ROUND_FLOOR))        # 4.4444
print(x.quantize(Decimal('1.0000'), ROUND_UP))           # 4.4445
print(x.quantize(Decimal('1.0000'), ROUND_DOWN))         # 4.4444
```

说明

```
ROUND_HALF_EVENT 和 ROUND_HALF_DOWN：EVENT是quansize的默认设置值，可以通过getcontext()得到，EVENT由于没有达到四舍五入所以不进位，DOWN同样不进位。

ROUND_CEILING 和 ROUND_FLOOR：CEILING正数始终进位，FLOOR在正数则始终不会进位。

ROUND_UP 和 ROUND_DOWN：UP始终进位，DOWN始终不会进位
```



