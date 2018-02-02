"""
输出打印
"""
# print()
# 默认print()输出换行,等价于print(,end="\n")
# print("你好" ,end="")
# print("中国",'啊！')
# print("啊！")

"""
读取键盘输入
"""

# input()
# 输入内容以字符串格式返回
# a = input("请输入任意数字：")
# print(type(a))

"""
Python2.X：
"""
# input
#2.x中：input函数返回的类型由输入值所用的界定符来决定
# a = input("请输入任意数字：")
# print type(a)------》1，'1'---->int ,str

# raw_input()
# 2.x中,此函数与3.x中的input类似，返回均是字符串

# print
# 2.x中,采用print 语句输出，而3.x中采用print()函数输出


"""
输出格式美化
"""
# 方法一：
# str.format() 

# 括号及其里面的字符 (称作格式化字段) 将会被 format() 中的参数替换。
# print('{}网址： "{}!"'.format('菜鸟教程', 'www.runoob.com'))

# 在括号中的数字用于指向传入对象在 format() 中的位置，如下所示：
# print('{0} 和 {1}'.format('Google', 'Runoob'))
# print('{1} 和 {0}'.format('Google', 'Runoob'))

# 如果在 format() 中使用了关键字参数, 那么它们的值会指向使用该名字的参数。
# print('站点列表 {0}, {1}, 和 {other}。'.format('Google', 'Runoob', other='Taobao'))

# '!a' (使用 ascii()), '!s' (使用 str()) 和 '!r' (使用 repr()) 可以用于在格式化某个值之前对其进行转化
# import math
# print('常量 PI 的值近似为： {!r}。'.format(math.pi))

# 可选项 ':' 和格式标识符可以跟着字段名。 这就允许对值进行更好的格式化
# 将 Pi 保留到小数点后三位
# print('常量 PI 的值近似为 {0:.3f}。'.format(math.pi)

# 在 ':' 后传入一个整数, 可以保证该域至少有这么多的宽度。 用于美化表格时很有用。
# table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
# for name, number in table.items():
    # print('{0:10} ==> {1:10d}'.format(name, number))

# 如果你有一个很长的格式化字符串, 而你不想将它们分开, 那么在格式化时通过变量名而非位置会是很好的事情。
# 传入一个字典, 然后使用方括号 '[]' 来访问键值 
# table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
# print('Runoob: {0[Runoob]:d}; Google: {0[Google]:d}; Taobao: {0[Taobao]:d}'.format(table))
# 也可以通过在 table 变量前使用 '**' 来实现相同的功能
# table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
# print('Runoob: {Runoob:d}; Google: {Google:d}; Taobao: {Taobao:d}'.format(**table))

# 方法二：
# 旧式字符串格式化：%，参见“字符串运算符”




