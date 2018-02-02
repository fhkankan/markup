# 集合
# 无序无重复的一组元素
"""
创建
"""
# set
# 可变无序不重复的序列
# set1 = set()
# set2 = set('abcde')
# set3 = {1,2,3}

"""
成员测试
"""
# in
# 元素是否存在，存在返回True,不存在返回False

# not in
# 元素是否不存在，不存在返回True， 存在返回False

"""
方法
"""
# pop()
# 由于无序，删除元素随机


"""
运算
"""
# -
# 差集

# |
# 并集

# &
# 交集

# ^
# 不同时存在的元素

a = set('abcd')
b = set('cdef')
print('a和b的差集：',a-b)
print('a和b的并集：',a|b)
print('a和b的交集：',a&b)
print('a和b的不同时存在的元素：',a^b)