# 一般形式
# if condition_1:
    # statement_block_1
# elif condition_2:
    # statement_block_2
# else:
    # statement_block_3
"""
1、每个条件后面要使用冒号（:），表示接下来是满足条件后要执行的语句块。
2、使用缩进来划分语句块，相同缩进数的语句在一起组成一个语句块。
3、在Python中没有switch – case语句。
"""

# if
# flag = True
# if flag:
#     print("您灭有危险品，可以进如车站")
# print("测试")


# if...else
# age = 15
# if age >=18:
#     print("已成年，可以进入网吧！")
# else:
#     print("未成年，禁止进入网吧")
# print("测试")

# if-elif 
# 发现有条件满足，将对应的缩进代码，以后的判断条件将不再执行
# 范围由小变大，可取中间
# score =88
# if score >= 90 :
#     print("优")
# elif score >= 80:
#     print("良")
# elif score >=60:
#     print("中")
# elif score >= 0:
#     print("差")

#  if嵌套
# 公交卡余额
# money = eval(input("please enter the money left in the card:"))
# # 是否有座位
# flag = True
# if money >= 2:
#     print("go on ")
#     if flag:
#         print("please sit down")
#     else:
#         print("please stand on")
# else:
#     print("get off")




