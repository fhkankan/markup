# 快速排序法
# 递归思想，sum()函数即是如此思想计算
# 计算速度的快慢取决于基准值的选取
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
        
    
    