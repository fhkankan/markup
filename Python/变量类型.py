"""
变量名命名规则：
"""
# 只能是一个词，不包含空格
# 只能包含字母、数字、下划线
# 不能以数字开头
# 不要将Python关键字和函数名用作变量名
# 慎用小写字母l和大写字母O，因为可能被人看错为数字1和0

# 注意：变量名区分大小写

# 常量名所有字母大写，由下划线连接各个单词。（通用习惯）

# 类名首字母大写

"""
变量的类型
"""
# None
# 空值，不支持任何运算也没有任何内置函数方法
# None和任何其他数据类型比较永远返回False
# 在python中，未指定返回值的函数自动返回None

# bool
# True/False

# string
# name = '张三'
# print(type(name))
# print(name[0])
# print(name[:])
# print('Ru\noob')
# print(r'Ru\noob')


# int
# print(type(10))

# float
# print(type(5.20))

# complex
# print(complex(1,2))

# bool
# print(type(True))

# list
# 可变有序序列
# list1 = []
# list1 = list()
# list2 = [1,2,'c']
# list3 = list('abcde')
# list4 = list(range(1,5,2))
# list5 = list(x*2 for x in range(5))
# print(list2.[1])
# print(list2.[0:])

# dictionary
# 可变无序序列
# dict1 = {}
# dict1 = dict()
# dict2 = {'name':'lilei','age':18}
# print(dict2['name'])

# set
# 可变无序不重复的序列
# set1 = set()
# set2 = set('abcde')
# set3 = {1,2,3}


# tuple
# 不可变的有序序列
# tuple1 = ()
# tuple1 = tuple()
# tuple2 = (1,2,'a')
# print(tuple3[0])

"""
Python中的变量值均是引用的，故不可变类型的重新赋值需要重新分配内存空间

可变类型：list，dictionary，set
不可变类型：number，sting，tuple
有序的：string, list, tuple
无序的：dictionary, set

"""
