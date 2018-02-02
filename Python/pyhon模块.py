"""
keyword
"""
# 记录关键字

keyword.kwlist
# 返回所有Python关键字的列表

iskeyword(字符串)
# 判断是否是python关键字，是则返回True，否则返回False

"""
copy
"""
# 复制

copy.copy(object)
# 浅拷贝,对内存地址进行复制，目标对象和源对象都指向同一片内存空间

copy.deepcopy(object)
# 深拷贝，目标对象和源对象分别有各自的内存空间，内存地址是自主分配的


"""
sys
"""
# 控制shell程序

sys.vesion
# 获取解释器的版本信息

sys.path
# 获取模块的搜索路径，初始化时使用PYTHONPATH环境变量的值

sys.platform
# 获取操作系统平台的名称

sys.maxint
# 最大的int值

sys.maxunicode
# 最大的Unicode值

sys.stdin
# 获取信息到shell程序中

sys.stdout
# 向shell程序输出信息

sys.exit()
# 退出shell程序



"""
random
"""
# 随机数


"""
time
"""
# 时间相关

"""
calendar
"""
# 日历

'''
date
'''
# 日期和时间


"""
os
"""
# 对文件夹进行操作


"""
科学计算
"""
# NumPy
# 快速数组处理

# SciPy
# 数值运算

# Matplotlib
# 绘图

"""
pdb
"""
# 调试模块

"""
doctest
"""
# 测试模块