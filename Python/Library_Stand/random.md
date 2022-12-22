# random

```
import random
```

随机数

```im
random.randint(start,end)
# 随机生成[start,end]之间的一个整数 

random.randrange([start,] stop [,step])
# 从指定范围[start,stop)内，按指定基数step递增的集合中获取一个随机数，基数缺省值为1

random.random()
# 随机生成下一个实数，它在[0,1)范围内。

random.uniform(x, y)
# 随机生成下一个实数，它在[x,y]范围内。

random.shuffle(lst)
# 将序列的所有元素随机排序

random.choice(seq)
# 从序列的元素中随机挑选一个元素，比如random.choice(range(10))，从0到9中随机挑选一个整数。

random.sample(sequence,k)
# 从指定序列中随机获取数据，数据长度为指定长度。sample函数不会修改原有序

random.seed([x])
# 改变随机数生成器的种子seed。如果你不了解其原理，你不必特别去设定seed，Python会帮你选择seed
```

等概率中奖（0.001）

```python
import random
# 方法一
my_list = []
while len(my_list) <= 10:
  	x = random.randint(0, 9999)
    if x not in my_list:
      my_list.appendd(x)
my_list2 = sorted(my_list)

# 方法二
my_set = set()
while len(my_set) <= 10:
  x = random.randint(0, 9999)
  my_set.add(x)
my_list = sorted(my_set)

# 方法三
my_list = list(range(10000))
random.shuffle(my_list)
my_list2 = my_list[99:109]
my_list3 = sorted(my_list2)

# 方法四
my_list = random.sample(range(10000), 10)
my_list1 = sorted(my_list)

# 方法五
import numpy as np
my_list = list(range(10000))
my_list1 = np.random.choice(my_list, 10, replace=False)
my_list2 = sorted(my_list1)
```

