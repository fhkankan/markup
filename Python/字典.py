# 字典是无序，但是是一一对应的，key值不能重复
# 格式： 字典名 = {key1:value1,key2:value2,...}
# 字典中的key可以是任意类型，但是不能是可变的数据类型,如列表、字典等
my_dict = {"name": "老王", "age": 18}

"""
内置的函数
"""
# len(dict)
# 计算字典元素的个数

# str(dict)
# 以字符串形式输出字典

# type(variable)
# 返回变量的类型，若是字典则返回字典类型

# key in dict
# 若键在字典里，返回True,否则返回False

"""
字典的方法
"""
# dict1.clear()
# 删除字典中的所有元素

# dict1.copy()
# 返回一个字典的副本

# dict1.update(dict2)
# 把字典dict2中的键/值对更新到dict1中

# dict1.fromkeys(seq,value)
# 创建一个新字典，以序列seq中元素作字典的键，value为字典所有键对应的初始值

# dict1.get(key,default = None)
# 返回指定键的值，若键或值不存在，返回默认值

# dict1.setdefault(key,default = None)
# 类似get(),若键不存在于字典，把key和value添加到字典中

# keys
# 以列表返回字典所有的键

# dict1.values()
# 以列表返回字典所有的值

# dict1.items()
# 以列表返回可遍历的（键，值）元组数组


"""
遍历
"""
# keys
# for key in my_dict.keys():
    # print(key)
    
# values
# for value in my_dict.values():
    # print(value)
    
# items
# 输出为元组
# for item in my_dict.items():
    # print(item)
    
# key-value
# for key, value in my_dict.items():
    # print(key, value)

"""
查看
"""
# my_name = my_dict["name"]
# print(my_name)

"""
修改
"""
# 如果key存在，修改对应key对应大的value；
# my_dict["name"] = "老张"
# print(my_dict)

"""
添加
"""
# 如果key不存在，就添加
# my_dict["sex"] = "男"
# print(my_dict)

# 复制(浅复制)
my_dict2 = my_dict.copy()
print(my_dict2)

"""
删除
"""
# del
# del my_dict['name']
# del my_dict

# pop
# 由于字典无序，删除是随机的
# my_dict.popitem()
# my_dict.pop('name')
# print(my_dict)

# 清空
# clear()
# my_dict.clear()
# print(my_dict)

#去除重复的值,用set()
# for value in set(my_dict.values()):
	# print(value.title())



