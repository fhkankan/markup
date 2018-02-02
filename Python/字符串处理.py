mystr = 'hello world itcast and itcastcpp'
"""
遍历
"""
for char in mystr:
    print(char)

"""
查找
"""
# len(string)
# 返回字符串的长度

# count(str,beg,end)
# 返回str在start和end之间在mystr中出现的次数
# print(mystr.count('itcast',0,10))

# max(str)
# 返回字符串中最大的字母

# min(str)
# 返回字符串中最小的字母

# index(str,beg,end)
# 检查是否在字符串中，若有返回索引，若无则报异常
# try:
    # print(mystr.index('itcast',0,10))
# except ValueError:
    # print("不存在")
    
# rindex
# 类似index()函数，从右边开始检索
# print(mystr.rindex('itcast'))

# find(str,beg,end)
# 检查是否在字符串中，若有返回索引，若无返回-1
# print(mystr.find('itcast', 0, 10))

# rfind
# 类似find()函数，从右边开始检索
# print(mystr.rfind('itcast'))


"""
判断
"""
# starstwith(str,beg,end)
# 检查是否以str开头，是则返回True,否则返回False
# print(mystr.startswith("hello"))

# endswith(str,beg,end)
# 检查是否以str结束，是则返回True，否则返回False
# print(mystr.endswith("cpp"))

# isaplha
# 字符全是字母返回True,否则返回False
# print(mystr.isalpha())

# isdigit
# 字符全是数字返回True,否则返回False
# print(mystr.isdigit())

# isnumeric
# 字符全是数字返回True,否则返回False

# isdecimal
# 字符串是否只包含十进制字符，如果是返回true，否则返回false。

# isalnum
# 字符全是字母或数字则返回True,否则返回False
# print(mystr.isalnum())

# isspace
# 字符全是空格则返回True,否则返回False
# print(mystr.isspace())

# istitle
# 如果字符串是标题化的则返回 True，否则返回 False

# islower
#字符串中至少一个区分大小写的字符，并且所有这些字符都是小写，则返回 True，否则返回 False

# isupper
#字符串中至少一个区分大小写的字符，并且所有这些字符都是大写，则返回 True，否则返回 False


"""
对齐
"""
# ljust(width[,fillchar])
# 左对齐，用fillchar填充右空位，默认空格
# print(mystr.ljust(50))

# rjust(width[,fillchar])
# 右对齐，用fillchar填充左空位，默认空格
# print(mystr.rjust(50))

# center(width,fillchar)
# 指定宽度，居中对齐，fillchar为填充字符，默认空格

# zfill()
# 在数字的左边填充 0
# '12'.zfill(5)
# '3.14159265359'.zfill(5)

"""
转换
"""
# capitalize
# 将字符串的第一个字符大写
# print(mystr.capitalize())

# title
# 把字符串的每个单词首字母大写
# print(mystr.title())

# lower
# 转换为全小写
# print(mystr.lower())

# upper
# 转换为全大写
# print(mystr.upper())

# swapcase
# 将字符串中大写转换为小写，小写转换为大写
# print(mystr.swapcase())

# expandtabs(tabsize=8)
# 把字符串 string 中的 tab 符号转为空格，tab 符号默认的空格数是 8 。

"""
修改
"""
# replace(old,new[,count])
# 用字符串new替换成old,次数不超过count,返回一个新字符串
# print(mystr.replace("itcast","Itcast"))

# join(seq)
# 以指定字符串作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串
# print('*'.join(mystr))


# lstrip
# 删除左边的特定字符(一个个)，默认空格
# print(mystr.center(50).lstrip())
# print(mystr.center(50).lstrip(" he"))

# rstrip
# 删除右边的特定字符(一个个)，默认空格
# print(mystr.center(50).rstrip())
# print(mystr.center(50).lstrip(" cp"))

# strip
# 删除两边的特定字符(一个个)，默认空格
# print(mystr.center(50).strip())

"""
分隔
"""
# split(str[,maxsplit])
# 以str为分隔符,最多maxsplit次，返回各个片段作为元素的列表
# print(mystr.split(" ",2))
# print(mystr.split("l"))

# splitlines([keepends])
# 按照行('\r', '\r\n', \n')分隔，返回包含各行作为元素的列表,如果keepends为False,则不包含换行符，否则保留换行符
# mystr="hello\nworld"
# print(mystr.splitlines())

# partition
# 将字符串分成三部分,返回一个元组
# print(mystr.partition('itcast'))

# rpartition
# 类似partition，从右边开始
# print(mystr.rpartition('itcast'))






