# 选择排序
"""
将数组中的元素按从小到大的顺序排序
"""
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