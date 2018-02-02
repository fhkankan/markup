# 二分法
"""
采用二分法猜测列表中是否有目的值
"""
# 定义二分法搜有序数列的函数
def binary_search(list, item):
    # 最低位的索引
    low = 0
    # 最高位的索引
    high = len(list)-1
    # 只要范围没有缩小到只包含1个元素，继续循环
    while low <= high:
        # 判断中间位置    
        mid = (low + high)//2
        guess = list[mid]
        if guess == item:
            return mid
        elif guess > item:
            high = mid - 1
        elif guess < item:
            low = mid + 1
    # 若没有指定元素，返回空
    return None
    
my_list = [1,3,5,7,9]
result1 = binary_search(my_list,3)
result2 = binary_search(my_list,-1)
print(result1)
print(result2)