# 二分法

算法描述

```
注意：二分查找的前提必须待查找的序列有序。

（设查找的数组期间为array[low, high]）
（1）确定该期间的中间位置K
（2）将查找的值T与array[k]比较。若相等，查找成功返回此位置；否则确定新的查找区域，继续二分查找。区域确定如下：
a.array[k]>T 由数组的有序性可知array[k,k+1,……,high]>T;故新的区间为array[low,……，K-1]
b.array[k]<T 类似上面查找区间为array[k+1,……，high]。每一次查找与中间值比较，可以确定是否查找成功，不成功当前查找区间缩小一半。递归找，即可。
```

时间复杂度

```
O(log2n)
```

实现

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

def BinarySearch(array,t):
    # 最低位的索引
    low = 0
    # 最高位的索引
    height = len(array)-1
    # 只要范围没有缩小到只包含1个元素，继续循环
    while low < height:
        # 判断中间位置
        mid = (low+height)/2
        if array[mid] < t:
            low = mid + 1
        elif array[mid] > t:
            height = mid - 1
        else:
            return array[mid]
    # 若没有指定元素，返回空
    return None


my_list = [1,3,5,7,9]
result1 = binary_search(my_list,3)
result2 = binary_search(my_list,-1)
print(result1)
print(result2
```

# 简单查找

时间复杂度
```
O(n)
```

# 快速查找

时间复杂度
```
O(nlogn)
```