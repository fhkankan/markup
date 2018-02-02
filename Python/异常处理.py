"""
异常处理
"""
try:
    # 可能引发异常的代码
except 异常类型:
    # 异常时所要执行的代码
else:
    # 无异常时所要执行的代码
finally：
    # 有无异常均需执行的代码(清理行为)
 
# try语句按照如下方式工作；
# 首先，执行try子句（在关键字try和关键字except之间的语句）
# 如果没有异常发生，忽略except子句，try子句执行后结束。
# 如果在执行try子句的过程中发生了异常，那么try子句余下的部分将被忽略。如果异常的类型和 except 之后的名称相符，那么对应的except子句将被执行。

    
# 多个except子句
# 一个 try 语句可能包含多个except子句，分别来处理不同的特定的异常。最多只有一个分支会被执行。

# try-except嵌套   
# 如果一个异常没有与任何的except匹配，那么这个异常将会传递给上层的try中。


# 函数嵌套中的异常
# 异常处理并不仅仅处理那些直接发生在try子句中的异常，而且还能处理子句中调用的函数（甚至间接调用的函数）里抛出的异常



"""
抛出异常
"""
# 若是只想知道是否抛出了一个异常，并不处理它，可用raise
# 格式：raise 被抛出的异常
# 要被抛出的异常必须是一个异常的实例或者是异常的类（也就是 Exception 的子类）。
 raise NameError('HiThere')

"""
自定义异常
"""
# 通过创建一个新的exception类来拥有自己的异常。异常应该继承自 Exception 类，或者直接继承，或者间接继承  
class MyError(Exception):
        def __init__(self, value):
            self.value = value
        def __str__(self):
            return repr(self.value)
            
# 显示异常信息            
raise MyError('oops!')  
# 调用自定义异常          
try:
        raise MyError(2*2)
    except MyError as e:
        print('My exception occurred, value:', e.value)
  
    