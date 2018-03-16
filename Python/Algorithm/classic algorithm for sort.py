# 选择排序
def Selection_sort(a):
    for i in range(len(a) -1):
        min = i
        for j in range(i + 1, len(a)):
            if a[min] > a[j]:
                min = j
            if min != i:
                a[min], a[i] = a[i], a[min]
    return a



# 冒泡排序
def Dubble_sort(a):
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            if a[i] > a[j]:
                a[i],a[j] = a[j],a[i]
    return a


# 插入排序
def Insertion_sort(a):
    for i in range(1,len(a)):
        j = i
        while j>0 and a[j-1]>a[i]:
            j -= 1
        a.insert(j,a[i])
        a.pop(i+1)
    return a


# 快速排序
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