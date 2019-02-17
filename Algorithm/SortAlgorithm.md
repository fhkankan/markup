# 排序

排序算法是一种将一串无规律数据依照特定顺序进行排列的一种方法

关键

```
有序对列：有序区刚开始没有任何数据，逐渐变多

无序队列：无序区刚开始存放所有数据，主键变少
```

稳定性

```
对列中有相同的元素，排序前后，这两个相同元素的顺序有没有发生变化

无变化，表示算法有稳定性

有变化，表示算法没稳定性
```

常见排序算法

```
初级：冒泡、选择、插入

中级：快速

高级：堆、归并

其他：基数、希尔、桶
```

技术

```
以从小到大进行排序为例：

冒泡排序：在无序队列中选择 最小的移动到最左侧,

选择排序：定一个有序队列，从无序队列中 选择 最小的元素 追加到有序队列的末尾

插入排序：定一个有序队列，从无序队列中选择 第一个元素， 插入到到有序队列的 合适位置

希尔排序：通过对无序队列进行 分组，然后再采用 插入的排序方法

快速排序：指定一个元素，将无序队列 拆分为 大小两部分，然后 层级递进，最终实现有序队列

归并排序：是将无序队列 拆分，然后小 组内排序，组间元素比较后在 新队列中进行排序

堆 排 序：顺序表方式构造堆，首尾替换调整堆
```

成本

| 方法 | 最坏     | 平均          | 最好          | 稳定性 | 代码复杂度 |
| ---- | -------- | ------------- | ------------- | ------ | ---------- |
| 冒泡 | O(n**2)  | O(n**2)       | O(n**2)       | 是     | 简单       |
| 选择 | O(n**2)  | O(n**2)       | O(n**2)       | 否     | 简单       |
| 插入 | O(n**2)  | O(n**2)       | O(n**2)       | 是     | 简单       |
| 希尔 | O(n**2)  | O(nlogn~n**2) | O(nlogn~n**2) | 否     | 中下等     |
| 堆   | O(nlogn) | O(nlogn)      | O(nlogn)      | 否     | 中等       |
| 归并 | O(nlogn) | O(nlogn)      | O(nlogn)      | 是     | 中等       |
| 快速 | O(n**2)  | O(nlogn)      | O(nlogn)      | 否     | 中下等     |

# 冒泡排序

算法描述：

```
相邻的元素两两比较，升序的话：大的在右，小的在左，降序反之

经过数次比较循环，最终达到一个从小到大或从大到小的有序序列

在整个冒泡排序过程中，有一个标识指向两个元素的最大值，当最大值移动的时候，标识也随之移动，过程跟踪。
```

次数

```
比较次数： 无序队列元素个数-1
冒泡次数： 无序对列元素个数-1
```

时间复杂度

```
O(n**2)
```

稳定性

```
稳定
```

分析

```
最基本元素比较
每一次冒泡排序，内层元素比较次数
执行多少次冒泡排序
特殊情况：已排好序
```

实现

```python
def bubble_sort(alist):
    # 获取列表元素总数量
    n = len(alist)
    # 外部冒泡循环
    for j in range(n-1, 0, -1):
        # 开始比较前，定义计数器为0
        count = 0
        # 内部比较循环
        for i in range(j):
            # 元素比较后替换,过程跟踪
            if alist[i] > alist[i+1]:
                alist[i], alist[i+1] = alist[i+1], alist[i] 
                # 数据替换完毕，计数器加1
                count += 1
        # 若计数器的值为0，表示没有发生任何替换，就退出当前循环 
        if count == 0:
            break
```

实现2

```python
def Dubble_sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if a[i] > a[j]:
                a[i],a[j] = a[j],a[i]
    return a
```

# 选择排序

算法描述

```
首先在未排序序列中找到最小(大)元素，和无序对垒的第一个元素替换位置
以此类推，知道所有元素全部进入有序对列，即排序完毕
```

时间复杂度

```
O(n**2)
```

稳定性

```
稳定
```

分析

```python
无序队列查找最小元素 即“比较循环”

最小元素和无序队列第一个元素替换位置 即“元素替换”

需要进行多少次替换，才能形成队列 即“选择循环”
```

实现

```python
def selection_sort(alist):
    # 当前列表的长度
    n = len(alist)
    # 外部选择循环
    for j in range(n-1):
        # 定义min_index标签初始值
        min_index = j
        # cur标签元素下标移动范围
        for i in range(min_index+1, n):
            # 让下标为min_index和i的数值进行比较,找到最小元素
            if alist[i] < alist[min_index]:
                min_index = i
        # mix标签元素和无序队列首位置元素替换
        if min_index != j:
            alist[j], alist[min_index] = alist[min_index], alist[j]
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

# 插入排序

算法描述

```
构建有序空序列

选择无序对列的第一个元素，向放在有序队列末尾，然后进行冒泡排序，放到指定的位置

循环上步，直到无序对列中所有元素全部进入有序序列的合适位置
```

时间复杂度

````
O(n**2)
````

稳定性

```
稳定
```

实现

```python
def insert_sort(alist):
    # 无序队列元素数量
    n = len(alist)
    # 无序队列元素放置到有序队列，有序对列循环次数
    for i in range(n):
        # 有序队列冒泡
        for j in range(i, 0, -1):
            if alist[j] < alist[j-1]:
                # 大小数值元素进行替换
                alist[j], alist[j-1] = alist[j-1], alist[j]
            else:
                break
```

实现2

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

# 希尔排序

算法描述

```
首先根据下标的范围进行分组
i和i+len/2是一组，len/2为小标偏移量

对同一组的元素进行插入排序

对下标偏移量/2，重新进行分组后，对同一组进行插入排序

按照上步循环下去，直到下标偏移量为1的分组插入结束
```

时间复杂度

```
根据步长序列的不同而不同
O(n**2)
```

稳定性

```
不稳定
```

实现

```python
def shell_sort(alist):
    # 获取列表的长度
    n = len(alist)
    # 获取gap的偏移值
    gap = n//2
    # 只要gap在合理范围内
    while gap >= 1:
        # 指定i下标的取值范围
        for i in range(gap, n):
            # 对移动元素的下标进行条件判断
            while (i-gap) >= 0:
                # 组内大小元素替换
                if alist[i] < alist[i-gap]:
                    alist[i], alist[i-gap] = alist[i-gap], alist[i]
                    # 更新迁移元素的下标值为最新值
                    i = i-gap
                else:
                    break
        # 每执行完毕一次分组内的插入排序，对gap进行/2细分
        gap = gap//2

def main():
    li = [36, 44, 57, 86, 76, 76]
    print(li)

if __name__ == '__main__':
    main()
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
最优：O(nlogn)
最差：O(n**2)
```

稳定性

```
不稳定
```

实现

```python
def quick_sort(alist, start, end):
    """快速排序"""
    if start < end:
        # 定义三个标签
        mid = alist[start]
        left = start
        right = end
        # 定义拆分条件
        while left < right:
            # 右标签移动
            if right > left and alist[right] >= mid:
                right -= 1
            alist[left] = alist[right]
            # 左标签移动
            if right > left and alist[left] < mid:
                left += 1
            alist[right] = alist[left]
        # 中间元素归位
        alist[left] = mid

        # 对切割后左边的小组进行快排
        quick_sort(alist, start, left-1)
        # 对切割后右边的小组进行快排
        quick_sort(alist, left+1, end)
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
        # 由所有大于基准值的元素构成的子数组
        greeter = [i for i in array[1:] if  i > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greeter)
```

# 归并排序

算法描述

```
分组排序阶段
1.将无序队列alist，拆分成两个小组A和B
2.分别对两个小组进行同样的冒泡排序
3.用标签left和right，分表对小组A和小组B进行管理

合并新队列
4.两个标签所在的元素比较大小
5.将小的元素放到一个新队列中，然后小元素所在标签向右移
6.多次执行4和5，最终肯定有一个小组先为空
7.把不为空的小组元素，按顺序全部移到新队列的末尾
8.无序队列中的所有元素就在新队列中形成有序队列了
```

时间复杂度

```
最优:O(nlogn)
最差:O(nlogn)
```

稳定性

```
稳定
```

实现

```python
def group(alist):
    n = len(alist)
    if n <= 1:
        return alist
    else:
        mid = n // 2
        left = group(alist[:mid])
        right = group(alist[mid:])
        return merge(left, right)

def merge(left, right):
    # 准备工作
    l, r = 0, 0
    # 新列表放置新队列
    result = []
    # 获取分组的长度
    left_len = len(left)
    right_len = len(right)

    # 元素比较条件
    while l < left_len and r < right_len:
        # 左队列移动元素到新队列
        if left[l] <= right[r]:
            result.append(left[l])
            l += 1
        # 右队列移动元素到新队列
        else:
            result.append(right[r])
            r += 1

    # 剩余内容添加到result中
    result += left[l:]
    result += right[r:]
    # 返回result表
    return result

if __name__ == "__main__":
    li = [54,26,93,17,77,31,44,77,20]
    print("处理前: %s" % li)
    sortlist = group(li)
    print("处理后: %s" % li)
    print("新列表: %s" % sortlist)
```

# 堆排序

算法描述

```
堆是采用顺序表存储的一种近似完全二叉树的结构，本质上是一种选择排序

1.根据完全二叉树结构，将无序队列
```

时间复杂度

```
O(nlogn)
```

稳定性

```
不稳定
```

实现

```python
def shift(alist, low, high):
    """堆调整"""
    # 堆顶节点的下标
    i = low
    # 上移节点标号j,临时指向左侧子节点标号
    j = 2 * i + 1
    # 把堆顶节点移动到临时队列
    tmp = alist[i]
    # 左侧子节点小于堆的最大范围值
    while j <= high:
        # 右侧节点元素若大于左侧节点元素
        if j + 1 <= high and alist[j] < alist[j+1]:
            # 上移节点标号j指向右侧节点
            j += 1
        # 若上移节点标号的元素大于移出的堆顶元素
        if alist[j] > tmp:
            # 把上移节点元素移动到堆顶位置
            alist[i] = alist[j]
            # 调整空位置节点标号i指向最新的空位置节点标号
            i = j
            # 上移节点标号j,临时指向空位置节点的左侧子节点标号
            j = 2 * i + 1
        # 若子节点标号不在队列中，就退出操作
        else:
            break
    # 设置堆顶节点为原来的内容
    alist[i] = tmp

def heap_sort(alist):
    """堆排序"""
    # 获取队列的长度
    n = len(alist)
    # 构建初始堆结构
    # 调整节点范围，从n/2-1开始，倒序到0
    for i in range(int(n/2)-1, -1, -1):
        shift(alist, i, n-1)
    # 堆顶元素排序
    # 遍历队列最小元素
    for i in range(n-1, -1, -1):
        # 堆顶元素和堆最小元素进行替换
        alist[0], alist[i] = alist[i], alist[0]
        # 调整新队列
        shift(alist, 0, i-1)
    # 返回排序后的队列
    return alist

if __name__ == '__main__':
    a = [0, 2, 6, 98, 34, 5, 23, 11, 89, 100, 7]
    print("排序之前：%s" % a)
    c = heap_sort(a)
    print("排序之后：%s" % c)
```

