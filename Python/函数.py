"""
pass语句                                          
"""
# pass是空语句，是为了保持程序结构的完整性。一般用作占位符。

"""
创建与调用
"""
# 创建函数一般格式：
# def 函数名（参数列表）:
    # 函数体
    
# 调用函数一般格式：
# 函数名(参数列表)

# 函数 
# 可重复使用的，用来实现单一，或相关联功能的代码段
# 同一个程序中函数名不能相同，变量名不能与函数名相同 
# 在python中，采用def关键字进行函数定义，不用指定返回值类型
# 函数参数可以是零个、一个或多个，也不用指定参数类型。因为python中变量都是弱类型的，会自动根据值来维护其类型



'''
return [表达式]
'''
#无论循环嵌套多少层，只要遇到一个return返回表达式后，就退出整个函数函数。没有return语句或不带参数值的return语句返回None
# return后面可以是元组，列表、字典等，只要是能够存储多个数据的类型，就可以一次性返回多个数据
 # def function():
          # return [1, 2, 3]
          # return (1, 2, 3)
          # return {"num1": 1, "num2": 2, "num3": 3}
# 如果return后面有多个数据，那么默认是元组
# def s(a,b,c):
    # return a,b,c
  
# print(s('a',2,3))

"""
返回的数据拆包
"""
# 拆包时要注意，需要拆的数据的个数要与变量的个数相同，否则程序会异常
# 拆包适用于元组、列表、字典(获得key)等
# def get_my_info():
    # high = 178
    # weight = 100
    # age = 18
    # return high, weight, age
# my_high, my_weight, my_age = get_my_info()
# print(my_high)
# print(my_weight)
# print(my_age)


"""
函数类型
"""
# 无参数无返回
# def 函数名():
    # 语句

# 无参数有返回
# def 函数名():
    # 语句
    # return 需要返回的数值
    
# 有参数无返回
# def 函数名(形参列表):
    # 语句

# 有参数有返回
# def 函数名(形参列表):
    # 语句
    # return 需要返回的数值

"""
形参与实参
"""
# 定义时小括号中的参数，用来接收参数用的，称为 “形参”
# 调用时小括号中的参数，用来传递给函数用的，称为 “实参”


"""
参数的传递
"""
# 在python中，一切皆对象，变量中存放的是对象的引用
# 在python中，参数传递的是值(实参的id值)传递，对于绝大多数情况，函数内部直接修改形参的值是不会影响实参


"""
参数类型
"""
# 必需参数(位置参数)
# 必需参数须以正确的顺序传入函数。调用时的数量必须和声明时的一样。
# 默认情况下，参数值和参数名称是按函数声明中定义的的顺序匹配起来的

# 关键字参数
# 使用关键字参数允许函数调用时参数的顺序与声明时不一致，因为 Python 解释器能够用参数名匹配参数值。

# 默认参数(缺省参数)
# 调用函数时，如果缺省传递参数，则会使用默认参数。
# 注意：默认参数位于最后面

# 不定长参数
# 能处理比当初声明时更多的参数
# 加*的变量args会存放所有未命名的变量参数(依位置传参)，args为元组
# 加**的变量kwargs会存放命名参数(依key传参),即形如key=value的参数,kwargs为字典.
# def functionname([formal_args,] *args, **kwargs):
   # """函数_文档字符串"""
   # function_suite
   # return [expression]
   
"""
变量作用域
"""
# 程序的变量并不是在哪个位置都可以访问的，访问权限决定于这个变量是在哪里赋值的。 变量的作用域决定了在哪一部分程序可以访问哪个特定的变量名称
# L （Local） 局部作用域
# E （Enclosing） 闭包函数外的函数中
# G （Global） 全局作用域
# B （Built-in） 内建作用域
# 以 L –> E –> G –>B 的规则查找

# x = int(2.9)  # 内建作用域 
# g_count = 0  # 全局作用域
# def outer():
    # o_count = 1  # 闭包函数外的函数中
    # def inner():
        # i_count = 2  # 局部作用域
# Python 中只有模块（module），类（class）以及函数（def、lambda）才会引入新的作用域，其它的代码块（如 if/elif/else/、try/except、for/while等）是不会引入新的作用域的，也就是说这这些语句内定义的变量，外部也可以访问

# 全局变量和局部变量
# 定义在函数内部的变量拥有一个局部作用域，定义在函数外的拥有全局作用域。
# 局部变量只能在其被声明的函数内部访问，而全局变量可以在整个程序范围内访问。调用函数时，所有在函数内声明的变量名称都将被加入到作用域中
# 全局变量不需要形参传值,可以直接在函数内使用，但是若想在内部更改全局变量值，需用global关键字

# 全局变量和局部变量名相同时
# 当需要时，先在函数内部找，找到后使用；若函数内部未有，在函数外部找，找到后使用；若函数外部也无，报错未定义

# global 和 nonlocal关键字
# 当内部作用域想修改为外部作用域的变量时，就要用到global和nonlocal关键字
# 如果要修改嵌套作用域(enclosing作用域,外层非全局作用域)中的变量则需要nonlocal关键字

# a = 10
# def test():
    # global a
    # a = a +1
    # print(a)
# test()
# print(a)


# def outer():
    # num = 10
    # def inner():
        # nonlocal num   # nonlocal关键字声明
        # num = 100
        # print(num)
    # inner()
    # print(num)
# outer()


"""
匿名函数
"""
# python 使用 lambda 来创建匿名函数。
# 所谓匿名，意即不再使用 def 语句这样标准的形式定义一个函数。
# lambda的主体是一个表达式，而不是一个代码块。仅仅能在lambda表达式中封装有限的逻辑进去。
# lambda 函数拥有自己的命名空间，且不能访问自己参数列表之外或全局命名空间里的参数。
# 虽然lambda函数看起来只能写一行，却不等同于C或C++的内联函数，后者的目的是调用小函数时不占用栈内存从而增加运行效率。
# 语法形式：lambda [arg1 [,arg2,.....argn]]:expression

# print((lambda a,b,c:(a+b+c)/3 )(3,76,555))

# def fun(a, b, opt):
    # ret = opt(a,b)
    # print(ret)

# fun(1, 2, lambda x,y:x+y)

# stus = [{"name": "zhangsan", "age": 18},{"name": "lisi", "age": 19}]
# stus.sort(key = lambda x: x['name'])
# stus.sort(key = lambda x: x['age'])

"""
闭包(函数的嵌套)
"""
# 可以在函数你不定义一个嵌套函数，将嵌套函数视为一个对象
# def func():
    # def add(x,y):
        # return x+y
    # return add
    
# fadd = func()
# print(fadd(3,5)

"""
递归函数
"""
# 函数在内部调用自己本身
# 递归必须有结束条件，递归向结束条件发展
# 计算输入数字的阶乘
# def factorial(num):
    # if num > 1:
        # result = num*factorial(num-1)
    # else:
        # result = 1
    # return result
    
# print(factorial(3))



# ***可更改与不可更改类型***
"""
不可更改的对象：numbers ，strings, tuples 
可以修改的对象：list,dict,set
有序的对象：strings,list,tuples
不可变类型：变量赋值 a=5 后再赋值 a=10，这里实际是新生成一个 int 值对象 10，再让 a 指向它，而 5 被丢弃，不是改变a的值，相当于新生成了a。
可变类型：变量赋值 la=[1,2,3,4] 后再赋值 la[2]=5 则是将 list la 的第三个元素值更改，本身la没有动，只是其内部的一部分值被修改了。
python 函数的参数传递：
不可变类型：类似 c++ 的值传递，如fun（a），传递的只是a的值，没有影响a对象本身。比如在 fun（a）内部修改 a 的值，只是修改另一个复制的对象，不会影响 a 本身。
可变类型：类似 c++ 的引用传递，如 fun（la），则是将 la 真正的传过去，修改后fun外部的la也会受影响
python 中一切都是对象，严格意义我们不能说值传递还是引用传递，我们应该说传不可变对象和传可变对象。
"""
# def ChangeInt( a ):
    # a = 10
# b = 2
# ChangeInt(b)
# print( b )

# 可变变量，地址不变
def f(a,L=[]):
    L.append(a)
    return L

print(f(2))
print(f(2,[1,2]))
print(f(2))


# 可变变量需要注意+=和= +在可变变量中的区别
# def func(b):
    # b += b  # 是直接对b指向的空间进行修改，而不是让b指向一个新的
    # b = b +b #先计算“=”右边的结果，之后对左边变量进行赋值，指向了新的内存空间

# a = [1,2]
# a = 10
# func(a)
# print(a)




