# random

```
import random
```

随机数

```
random.randint(start,end)
# 随机生成[start,end]之间的所有整数 

random.randrange ([start,] stop [,step])
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
# 从指定序列中随机获取指定长度的片段。sample函数不会修改原有序

random.seed([x])
# 改变随机数生成器的种子seed。如果你不了解其原理，你不必特别去设定seed，Python会帮你选择seed
```