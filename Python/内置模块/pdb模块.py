import pdb

# 调试模块
# 采用命令交互方式，可以设置断点、单步执行、查看变量等。

# 语句块调试
# run()函数可以对语句块进行调试
import pdb
pdb.run('''
for i in range(1,3):
    print(i)
''')
# 运行后会出现调试明亮提示，输入如下命令
# h/help:查看命令列表
# b/break:设置断点
# j/jump:跳转至指定行
# n/next：执行下一条语句，不进入函数
# r/return：运行到函数返回
# s/step：执行下一条语句，遇到函数返回
# q/quit：退出pdb

# 函数调试
# runcall()可对函数进行调试
import pdb
def sum(a,b):
    return a+b
    
pdb.runcall(sum,10,6)


