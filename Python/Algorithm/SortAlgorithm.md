# 选择排序

算法描述

```
对于一组关键字{K1,K2,…,Kn}， 首先从K1,K2,…,Kn中选择最小值，假如它是 Kz，则将Kz与 K1对换；
然后从K2，K3，… ，Kn中选择最小值 Kz，再将Kz与K2对换。
如此进行选择和调换n-2趟，第(n-1)趟，从Kn-1、Kn中选择最小值 Kz将Kz与Kn-1对换，最后剩下的就是该序列中的最大值，一个由小到大的有序序列就这样形成。
```

时间复杂度

```
O(n*n)
```

实现

```python
def Selection_sort(a):
    for i in range(len(a) -1):
        min = i
        for j in range(i + 1, len(a)):
            if a[min] > a[j]:
                min = j
            if min != i:
                a[min], a[i] = a[i], a[min]
    return a
```

实现2

```python
def findSmallest(arr):
    # 设定第一个值为最小值
    smallest = arr[0]
    # 最小值的索引
    smallest_index = 0
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index
    
def selectionSort(arr):
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))
    return newArr
    
print(selectionSort([5,4,3,7,9]))
```



# 冒泡排序

算法描述：

```
共循环 n-1 次
每次循环中，如果 前面的数大于后面的数，就交换
设置一个标签，如果上次没有交换，就说明这个是已经好了的。
```

时间复杂度

```
n(n-1)/2
```

实现

```python
def Dubble_sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if a[i] > a[j]:
                a[i],a[j] = a[j],a[i]
    return a
```

实现2

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

def bubble(l):
    flag = True
    for i in range(len(l)-1, 0, -1):
        if flag: 
            flag = False
            for j in range(i):
                if l[j] > l[j + 1]:
                    l[j], l[j+1] = l[j+1], l[j]
                    flag = True
        else:
            break
    print l

li = [21,44,2,45,33,4,3,67]
bubble(li)
```





# 插入排序

算法描述

```
设有一组关键字｛ K 1 ， K 2 ，…， K n ｝；排序开始就认为 K 1 是一个有序序列；让 K 2 插入上述表长为 1 的有序序列，使之成为一个表长为 2 的有序序列；然后让 K 3 插入上述表长为 2 的有序序列，使之成为一个表长为 3 的有序序列；依次类推，最后让 K n 插入上述表长为 n-1 的有序序列，得一个表长为 n 的有序序列。
```

时间复杂度

````
O(n*n)
````

实现

```python
def Insertion_sort(a):
    for i in range(1,len(a)):
        j = i
        while j>0 and a[j-1]>a[i]:
            j -= 1
        a.insert(j,a[i])
        a.pop(i+1)
    return a
```

# 快速排序

算法描述

```
先从数列中取出一个数作为基准数。

分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它的左边。

再对左右区间重复第二步，直到各区间只有一个数。
```

时间复杂度

```
计算速度的快慢取决于基准值的选取
```

实现

```python
def sub_sort(array,low,high):
    key = array[low]
    while low < high:
        while low < high and array[high] >= key:
            high -= 1
        while low < high and array[high] < key:
            array[low] = array[high]
            low += 1
            array[high] = array[low]
    array[low] = key
    return low

def quick_sort(array,low,high):
     if low < high:
        key_index = sub_sort(array,low,high)
        quick_sort(array,low,key_index)
        quick_sort(array,key_index+1,high)


if __name__ == '__main__':
    array = [8,10,9,6,4,16,5,13,26,18,2,45,34,23,1,7,3]
    print array
    quick_sort(array,0,len(array)-1)
    print array
```

实现2

```python
def quick_sort(array):
    # 基线条件：为空或者只包含一个元素的数组是‘有序’的
    if len(array) < 2:
        return array
    # 递归条件
    else:
        # 设定基准值
        pivot = array[0]
        # 由所有小于等于基准值的元素构成的子数组
        less = [i for i in array[1:] if i <= pivot]
        # 由所有大鱼基准值的匀速构成的子数组
        greeter = [i for i in array[1:] if  i > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greeter)
 

print(quick_sort([10,4,6,8]))
```

