# 一般形式 
# while 判断条件：
     # 执行语句
     
# i = 0
# while i < 10:
#     print("good!")
#     print("当前是第%d次执行循环"%(i+1))
#     i += 1

# 无限循环
while True:
    name = input("enter your name or 'q' to quit:")
    if name = "q":
        break

# 类似do...while的实现
"""
while True:    
    #dosomthing
    if fail_condition:
        break
"""


# while+break
# 遇到break,不执行后面，结束循环
# i =1
# while i <= 5:
#     i += 1
#     if i ==4:
#         break
#     print(i)

# while+continue
# 遇到Continue,不执行后面，跳至循环开始下次循环
# i = 1
# while i <= 5:
#     i += 1
#     if i ==4:
#         continue
#     print(i)

# while-else 
# 循环执行完时,执行else的语句块
# i = 0
# while i < 5:
#     i += 1
#     print(i)
# else:
#     print('haha')

# while-else+break
# 遇到break,结束循环，不执行else语句
# i = 0
# while i < 5:
#     i += 1
#     if i == 2:
#         break
#     print(i)
# else:
#     print('haha')
# print("nini")

# while-else+continue
# 遇到continue,跳出本次循环，循环执行完执行else语句
# i = 0
# while i < 5:
#     i += 1
#     if i == 2:
#         break
#     print(i)
# else:
#     print('haha')
# print("nini")






