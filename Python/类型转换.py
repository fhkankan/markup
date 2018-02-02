"""
整数的转换
"""
# float(x) 将x转换为一个浮点数
print(float('5'))
print(type(float('5')))
# 浮点型字符串不能直接转为int类型，需要先转为浮点型
print(type(int(float('10.00'))))

# complex(real[,imag])
# 将 real 和 imag转换到一个复数，实数部分为 real，虚数部分为 imag。real和 imag是数字表达式,imag缺失时默然为0

# chr(x)
# 将一个整数转换为一个Unicode字符

# ord(x)
# 将一个字符转换为它的ASCII整数值

# hex(x)
# 将一个整数转换为十六进制字符串

# oct(x)
# 将一个整数转换为八进制字符串

# bin(x)
# 将一个整数转换为一个二进制字符串

# int(x[,base]) 
# 将x转换为一个十进制整数

print(float(10.2))
print(complex(3))
print(char(65))
print(ord('A'))
print(bin(5))
print(oct(20))
print(hex(20))
print(int("16"))
print(int("11", 8))


"""
字符串与表达式的转换
"""

# str(x)
# 将对象x转换为字符串，给程序员看
# str_num = str(123)
str_num = str('123')
print(type(str_num))
print(str_num)

# repr(x)
# 将对象x转换为表达式字符串，给计算机看
b = repr(123)
b = repr('123')
print(type(b))
print(b)

# eval(str)
# 用来计算在字符串中的有效表达式，并返回一个对象
a = eval("5")
print(a)
print(type(a))
print(eval("'abc'"))
# 报错
# print(eval('abc'))
#
print(repr(4+3))
print(eval("4+3"))

"""
set/list/tuple
"""
# s可以为元组、列表或集合
# tuple(s)
# 将序列s转换为一个元组

# list(s)
# 将序列s转换为一个列表

# set(s)
# 将序列s转换为一个集合






