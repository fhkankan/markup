"""
三种命名规则
"""
# 小驼峰
myName = "张三"
# 大驼峰
MyName = '张三'
#下划线（建议）
my_name = "张三"




"""
注释
"""
# 单行注释
# #

# 多行注释
# '''   '''
# """    """


"""
帮助
"""
# help(对象)

# 内置函数和类型
help(max)
help(list)

# 模块
import math
help(math)

# 模块中成员函数
import os
help(os.fdopen)

"""
语句过长时
"""
# 方法一：(推荐)
# 使用（中间换行）

a = ('这是一个很长很长很长很长很长很长很'
       '长很长很长很长很长很长的字符串')
if (width == 0 and height ==0 and
    color == 'red' and emphasis == 'strong')

# 方法二：
# 使用‘\’

a = '这是一个很长很长很长很长很长很长很\
       长很长很长很长很长很长的字符串'