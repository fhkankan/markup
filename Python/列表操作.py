# list1 = [1, 2, 3, 4]
# 将list2关联到list1中的列表，均指向同一列表
# list2 = list1
# 将list1的副本复制一份给list3。
# list3 = list1[:]
# print(id(list1))
# print(id(list2))
# print(id(list3))

"""
多维列表
"""
# 二维列表即其他语言的二维数组
# list1 = [[],[]]
# list1.[0][1]

# 定义3行6列的二维列表，打印出元素值
rows = 3
cols = 6
matrix = [[0 for col in range(cols)] for row in range(rows)]
for i in range(rows):
    for j in range(cols):
        matrix[i][j] = i * 3 + j
        print(matrix[i][j],end = '')
    print('\n')
print(matrix)

"""
操作符
"""
# len
# 列表元素的个数

# +
# 列表的组合

# *
# 列表元素的重复

# in/not in 
# 判断元素是否在列表中

# for x in [1,2,3]: print(x)
# 列表的解析
 


"""
遍历
"""
# for循环
# list1 = [1, 2, 5, 4]
# for var in list1:
#     print(var)
# print(list1)

# enumerate遍历
# for i, value in enumerate(list1):
#     print(i, value)

"""
添加元素
"""

# append(obj)
# 在末尾追加对象(整体)
# list1.append([3, 4])
# list1.append('c')
# print(list1)

# extend(seq)
# 在末尾添加可迭代的对象元素
# list1 = ['a']
# list2 = ['c','d']
# 在末尾追加列表
# list1.append(list2)
# print(list1)

# insert(index,obj)
# 在索引前面添加对象(整体)
# list1.insert(2,'x')
# print(list1)

"""
修改元素
"""

# list1[2] = 'a'
# print(list1)

"""
查找元素
"""

# in/not in
# if 'a' in list1:
#     print("有")
# else:
#     print("没")

# find(obj)
# 检查是否在字符串中，若有返回索引，若无返回-1
# print(list1.find('b', 0, 10))

# index(obj)
# 存在则返回下标，不存在则报异常
# 若避免异常，则可加if-in判定
# print(list1.index('b', 0, 5))

# len(list)
# 返回列表中元素的个数

# count(obj)
# 输出列表所含字符的个数
# print(list1.count('a'))

# max(list)
# 内置函数，返回列表元素最大值

# min(list)
# 内置函数，返回列表元素最小值

"""
删除元素
"""

# del
# 内置函数，删除列表或列表中的元素
# del list1[0]
# del+空格等价del()
# del(list1[0])
# 干预对象提前结束
# del list1
# print(list1)

# remove(obj)
# 根据元素值删除
# list1.remove("b")
# print(list1)

# pop([index])
# 默认删除最后一个元素
# 删除列表中指定位置的元素后，返回删除的元素
# list1.pop()
# print(list1)
# print(list1.pop(0))
# print(list1)

# clear
# 把列表中的元素全部清空，等价于[],list()
# list1.clear()

"""
排序
"""

# 要求组内为同类数据
# sort([func])
# 将list按特定顺序重新排列，默认从小到大，参数reverse=True可改为倒序
# list1.sort()
# list1.sort(reverse = True)
# print(list1)

# sorted
# 将list临时按特定顺序排序
# a = sorted(list1)
# b = sorted(list1,reverse = True)
# print(a)
# print(b)

# reverse
# 将list按照逆序排列
# list1.reverse()
# print(list1)

# reversed
# 将list临时逆序
# a = reversed(list1)
# print(a)
# print(list(a))

