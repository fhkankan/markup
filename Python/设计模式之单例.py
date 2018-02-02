# 确保某一个类只有一个实例，而且自行实例化并向整个系统提供这个实例，这个类称为单例类
# 单例模式是一种对象创建型模式


# 当对一个类创建多个实例对象时，开辟了不同的内存空间，若想实现合理使用内存，而且可以不重复开辟新的内存，最终达到的效果就是重复使用对象名，但是地址都一样
# 如果调用对象名时，若对象已开辟空间则直接使用若没有开辟地址，将开辟地址并报错这个对象并返回这个对象


 # 保持对象的唯一性（类属性和new方法配合使用）
class Person(object):
    # 定义一个类属性保存实例对象
    __instance = None
    # 定义一个类属性保存对象属性是否是第一次赋值或实例化
    __is_first = True
    # 监听person类创建对象
    
    def __new__(cls, *args, **kwargs):
        # 如果__instance没有值，就代表没有使用这个类创建过对象
        if not cls.__instance:
            print('因为没有值')
            cls.__instance = object._new__(cls)
        return cls.__instance
        
    def __init__(self,name,age):
        # 如果是第一次初始化，则进行初始化，之后则不再进行初始化
       if Person.__is_first:
           self.name = name
           self.age = age
           # 对__is_first重新赋值为False
           Person.__is_first = False

    # 对象方法，求和操作
    def add2num(self,a,b):
        return a + b

# 对象地址一样，对象身上的属性值也一样
# 业务需求（对象地址唯一或单例对象属性唯一）
p1= Person('小红',19)
print(p1)
print(p1.name,p1.age)
p2 = Person('小明',20)
print(p2)
print(p2.name,p2.age)