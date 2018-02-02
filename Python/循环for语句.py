# 一般形式
# for 循环索引值 in 序列：
    # 循环体
    



# 可以遍历任何序列的项目，如字符串、列表、字典、元组
# for 循环把字符串中的每个字符串循环打印出来
# str1 = 'hello'
# for c in str1:
#     print(c)

# range(x,y,z)为[x,y)中以z做递增
# range(x) 从0开始，以1做递增，至x-1
# for i in range(10):
#     print("第%d次" % (i + 1))
#     print("媳妇儿，我错了")
# for i in range(5, 10, 2):
#     print(i)
#     print("我错了")



# for+break
#　break 语句用于跳出当前循环体：
for i in range(5):
    if i == 4:
        break
    print(i)

# for+continue
# 遇到Continue,跳至循环开头，执行下一循环
# for i in range(5):
#     if i == 4:
#         continue
#     print(i)

# for-else
# 只要在for语句中未遇到return/break/continue,for语句执行完执行else语句
# for i in range(5):
#     print(i)
# else:
#     print("for-else")
# print("ha")

# for-else+break
# 遇到break,跳出循环，不执行else
# for i in range(5):
#     print(i)
#     if i == 2:
#         break
# else:
#     print("for-else")
# print("ha")

# for-else+continue，
# 遇到Continue,跳至循环开头，循环执行完之后执行else
# for i in range(5):
#     if i == 2:
#         continue
#     print(i)
# else:
#     print("for-else")
# print("ha")


