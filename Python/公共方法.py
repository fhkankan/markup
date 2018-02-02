"""
遍历
"""
# 适用于字符串、列表、元组可遍历的数据对象

# for item in  any:
    print(item)
    
# for i char in enumerate(any)：
    print(i,char)


"""
运算符
"""
# +
# 合并，适用于：字符串、列表、元组
# print("123"+"abc")
# print([1, 2, 'a']+['b', True])
# print((1, 2, True)+('a', 3))

# *
# 复制，适用于：字符串、列表、元组
# print("1"*4)
# print([1,2]*2)
# print((1,2)*3)

# in
# 元素是否存在，存在返回True,不存在返回False
# 适用于：字符串、列表、元组、字典，集合
# if 'a' in "abcde":
#     print("yes")
# if 'a' in ["a",2]:
#     print("yes")
# if "a" in ("a","b"):
#     print("yes")
# in 判断的是键
# if "a" in {'a':1 }:
#     print("yes")

# not in
# 元素是否不存在，不存在返回True， 存在返回False
# 适用于：字符串、列表、元组、字典，集合

"""
内置函数
"""
# len(item)
# 计算容器中元素个数
# 适用于：字符串、列表、元组、字典
# print(len([1,[1,[2]]]))
# print(len({"a":1,"b":[1,2]}))

# max(item)
# 返回容器中元素最大值
# print(max("abcde"))
# print(max([1,2,3]))
# print(max((1,2,3)))
# print(max({"a":1,"b":2}))

# min(item)
# 返回容器中元素最小值

# del+ 空格或del(item)
# 删除变量
# 适用于：字符串、列表、元组、字典
# variType = "abcde"
# print(variType)
# del variType
# 上下等价
# del(variType)
# 删除后输出会报错
# print(variType)

# id(item)
# 返回变量的地址
# 适用于：字符串、列表、元组、字典

# type(item)
# 返回变量的类型
# 适用于：字符串、列表、元组、字典


# help()
# 帮助，输出与数据类型相关的方法与属性
# 适用于：字符串、列表、元组、字典

# range(a,b,p)
# 从a到b，以p作步距的序列

# round(number[.ndigits])
# 保留指定的小数点后位数的四舍五入的值


"""
切片截取
"""
# 适用于：字符串、列表、元组
# 格式：[起始：结束：步长]
# 区间从"起始"开始，到"结束"位的前一位结束（不包含结束位本身)，步长表示选取间隔

#字符串不可变，列表可变，当进行切片操作时，其实是操作一个副本
# 切片语法：[起始：结束：步长]
# 左闭右开,第一个下标为0，倒数第一个下标-1
# a = 'abcdef'
# a[:3],a[0:3],a[0:3:1]等价
# fedcba
# print(a[-1::-1])
list1 = ['a',3,4,5,6]
print(id(list1))
print(id(list1[:]))
print(id(list1[2:]))


"""
自带函数
"""
# 适用于字符串、列表、元组

# index
# 存在，返回索引，不存在，报错

# count
# 存在，返回数目，不存在，返回0
