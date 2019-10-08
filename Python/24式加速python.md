# 加速python

##分析代码运行时间

- 第1式，测算代码运行时间

平凡方法

```python
import time

tic = time.time()
much_job = [x**2 for x in range(1, 1000000, 3)]
toc = time.time()
print('used {:.5}s'.format(toc-tic))
```

快捷方法(jupyter环境)

```python
%%time
much_job = [x**2 for x in range(1, 1000000, 3)]
```

- 第2式，测算代码多次运行平均时间

平凡方法

```python
from timeit import timeit

g = lambda x:x**2 + 1
def main():
    return(g(2)**120)

#timeit('main()', setup='from __main__ import main', number =10)
timeit('main()', globals={'main':main}, number=10)
```

快捷方法(jupyter环境)

```python
%%timeit -n 10
g = lambda x:x**2+1
def main():
  	return(g(2)**120)
main()
```

- 第3式，按调用函数分析代码运行时间

平凡方法

```python
def relu(x):
		return(x if x>0 else 0)

def main():
  	result = [relu(x) for x in range(-100000, 100000, 1)]
		return result
```

```python
import profile
profile.run('main()')
```

快捷方法(jupyter)

```python
%prun main()
```

- 第4式，按行分析代码运行时间

平凡方法

```python
!pip install line_profiler
%load_ext line_profiler
```

```python
def relu(x):
		return(x if x>0 else 0)

def main():
  	result = [relu(x) for x in range(-100000, 100000, 1)]
		return result
```

```python
from line_profiler import LineProfiler

lprofile = LineProfiler(main, relu)
lprofile.run('main()')
lprofile.print_stats()
```

快捷方法(jupyter环境)

```python
%lprun -f main -f relu main()
```

## 加速查找

- 第5式，用set而非list进行查找

低速方法

```python
data = (i**2 + 1 for i in range(1000000))
list_data = list(data)
set_data = set(data)
```

```python
%%time
1098987 in list_data
```

高速方法

```python
%%time
1098987 in set_data
```

- 第6式，用dict而非两个list进行匹配查找

低速方法

```python
list_a = [2**i-1 for i inrange(1000000)]
list_b = [i**2 for i in list_a]
dict_ab = dict(zip(list_a, list_b))
```

```python
%%time
print(list_b[list_a.index(876567)])
```

高速方法

```python
%%time
print(dict_ab.get(876567, None))
```

## 加速循环

- 第7式，优先使用for循环而不是while循环

低速方法

```python
%%time
s, i = 0, 0
while i<10000:
  	i = i + 1
    s = s + i
print(s)
```

高速方法

```python
%%time
s = 0
for i in range(1, 10001):
  	s = s + i
print(s)
```

- 第8式，在循环提中避免重复计算

低速方法

```python
a = [i**2+1 for i in range(2000)]
```

```python
%%time
b = [i/sum(a) for i in a]
```

高速方法

```python
%%time
sum_a = sum(a)
b = [i/sum_a for i in a]
```

## 加速函数

- 第9式，用循环机制代替递归函数

低速方法

```python
%%time
def fib(n):
  	return(i if n in (1,2) else fib(n-1)+fib(n-2))
print(fib(30))
```

高速方法

```python
%%time
def fib(n):
  	if n in (1,2):
      	return(1)
    a, b = 1, 1
    for i in range(2, n):
      	a, b = b, a+b
    return(b)
  
print(fib(30))
```

- 第10式，用缓存机制加速递归函数

低速方法

```python
%%time
def fib(n):
  	return(i if n in (1,2) else fib(n-1)+fib(n-2))
print(fib(30))
```

高速方法

```python
%%time
from functools import lru_cache

@lru_cache(100)
def fib(n):
  	return(i if n in (1,2) else fib(n-1)+fib(n-2))
print(fib(30))
```

- 第11式，用numba加速python函数

低速方法

```python
%%time
def my_power(x):
  	return(x**2)
  
def my_power_sum(n):
  	s = 0
    for i in range(1, n+1):
      	s = s + my_power(i)
    return(s)
  
print(my_power_sum(1000000))
```

高速方法

```python
%%time
from numba import jit

@jit
def my_power(x):
  	return(x**2)
  
@jit
def my_power_sum(n):
  	s = 0
    for i in range(1, n+1):
      	s = s + my_power(i)
    return(s)
  
print(my_power_sum(1000000))
```

## 使用标准库函数

- 第12式，使用`collections.Counter`加速计数

低速方法

```python
data = [x**2%1989 for x in range(2000000)]
```

```python
%%time
values_count = {}
for i in data:
  	i_cnt = values_count.get(i, 0)
    values_count[i] = i_cnt + 1
print(values_count.get(4, 0))
```

高速方法

```python
%%time
from collections import Counter

values_count = Counter(data)
print(values_count.get(4, 0))
```

- 第13式，使用`collections.ChainMap`加速字典合并

低速方法

```python
dic_a = {i:i+1 for i in range(1, 1000000, 2)}
dic_b = {i:2*i+1 for i in range(1, 1000000, 3)}
dic_c = {i:3*i+1 for i in range(1, 1000000, 5)}
dic_d = {i:4*i+1 for i in range(1, 1000000, 7)}
```

```python
%%time
result = dic_a.copy()
result.update(dic_b)
result.update(dic_c)
result.update(dic_d)
print(result.get(9999, 0))
```

高速方法

```python
%%time
from collections import ChainMap
chain = ChainMap(dic_a, dic_b, dic_c, dic_d)
print(chain.get(9999, 0))
```

## 使用高阶函数

- 第14式，使用map代替推导式进行加速

低速方法

```python
%%time
result = [x**2 for x in range(1, 1000000, 3)]
```

高速方法

```python
%%time
result = map(lambda x:x**2, range(1, 1000000, 3))
```

- 第15式，使用filter代替推导式进行加速

低速方法

```python
%%time
result = [x for x in range(1, 1000000, 3) if x%7==0]
```

高速方法

```python
%%time
result = filter(lambda x:x%7==0, range(1, 1000000, 3))
```

## 使用numpy向量化

- 第16式，使用`np.array`代替list

低速方法

```python
%%time
a = range(1, 1000000， 3)
b = range(1000000, 1, -3)
c = [3*a[i]-2*b[i] for i in range(0, len(a))]
```

高速方法

```python
%%time
import numpy as np
array_a = np.arange(1, 1000000, 3)
array_b = np.arange(1000000, 1, -3)
array_c = 3*array_a - 2*array_b
```

- 第17式，使用`np.ufunc`代替`math.func`

低速方法

```python
%%time
import math

a = range(1, 1000000, 3)
b = [math.log(x for x in a)]
```

高速方法

```python
%%time
import numpy as np

array_a = np.arange(1, 1000000, 3)
array_b = np.log(array_a)
```

- 第18式，使用`np.where`代替if

低速方法

```python
import numpy as np
array_a = np.arange(-100000, 1000000)
```

```python 
%%time
# np.vectorize可以将普通函数转换成支持向量化的函数
relu = np.vectorize(lambda x:x if x>0 else 0)
array_b = relu(array_a)
```

高速方法

```python
%%time
relu = lambda x:np.where(x>0, x, 0)
array_b = relu(array_a)
```

## 加速pandas

- 第19式，使用csv文件读写代替excel文件读写

低速方法

```python
%%time
df.to_excel('data.xlsx')
```

高速方法

```python
%%time
df.to_csv('data.csv')
```

- 第20式，使用pandas多进程工具pandarallel

低速方法

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randint(-10, 11, size=(10000, 26)), columns=list('abcdefghijklmnopqrstuvwxyz'))
```

```python
%%time
result = df.apply(np.sum, axis=1)
```

高速方法

```python
%%time
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=4)
result = df.parallel_apply(np.sum. axis=1)
```

## 使用Dask加速

- 第21式，使用dask加速dataframe

低速方法

```python
import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0, 6, size=(100000000, 5)), columns=list('abcde'))

%time df.groupby('a').mean()
```

高速方法

```python
import dask.dataframe as dd

df_dask = dd.from_pandas(df, npartitions=40)

%time df_dask.groupy('a').mean().compute()
```

- 第22式，使用`dask.delayed`加速

低速方法

```python
import time 

def muchjob(x):
  	time.sleep(5)
    return(x**2)
```

```python
%%time
result = [muchjob(i) for i in range(5)]
result
```

高速方法

```python
%%time
from dask import delayed, compute
from dask import threaded, multiprocessing

values = [delayed(muchjob)(i) for i in range(5)]
result = compute(*values, scheduler='multiprocessing')
```

## 使用多线程多进程加速

- 第23式，应用多线程加速IO密集型任务

低速方法

```python
%%time
def writefile(i):
  	with open(str(i) + '.txt', 'w') as f:
      	s = ('hello %d' % i) * 10000000
        f.write(s)
        
# 串行任务
for i in range(10):
  	writefile(i)
```

高速方法

```python
%%time
import threading

def writefile(i):
  	with open(str(i) + '.txt', 'w') as f:
      	s = ('hello %d' % i) * 10000000
        f.write(s)
 
# 多线程任务
thread_list = []
for i in range(10):
  	t = threading.Thread(target=writefile, args=(i,))
    t.setDaemon(True)  # 设置为守护线程
    thread_list.append(t)
    
for t in thread_list:
  	t.start()  # 启动线程
    
for t in thread_lsit:
  	t.join()  # 等待子线程结束
```

- 第24式，应用多进程加速CPU密集型任务

低速方法

```python
%%time
import time

def muchjob(x):
  	time.sleep(5)
    return(x**2)
  
# 串行任务
ans = [muchjob(i) for i inrange(8)]
print(ans)
```

高速方法

```python
%%time
import time
import multiprocessing

data = range(8)

def muchjob(x):
  	time.sleep(5)
    return(x**2)
  
# 多进程任务
pool = multiprocessing.Pool(processes=4)
result = []
for i in range(8):
		result.append(pool.apply_async(muchjob, (i,)))
pool.close()
pool.join()
ans = [res.get() for res in result]
print(ans)
```

