# functools

模块应用于高阶函数，即参数或（和）返回值为其他函数的函数。 通常来说，此模块的功能适用于所有可调用对象。

## cache

简单轻量级未绑定函数缓存。返回值与 `lru_cache(maxsize=None)` 相同，创建一个查找函数参数的字典的简单包装器。 因为它不需要移出旧值，所以比带有大小限制的 [`lru_cache()`](https://docs.python.org/zh-cn/3/library/functools.html#functools.lru_cache) 更小更快。

```python
@cache
def factorial(n):
    return n * factorial(n-1) if n else 1

>>> factorial(10)      # no previously cached result, makes 11 recursive calls
3628800
>>> factorial(5)       # just looks up cached value result
120
>>> factorial(12)      # makes two new recursive calls, the other 10 are cached
479001600
```

## cached_property

将类的方法转换为一个属性，该属性的值计算一次，然后在实例的生命周期中将其缓存作为普通属性。与 property() 类似，但添加了缓存，对于在其他情况下实际不可变的高计算资源消耗的实例特征属性来说该函数非常有用。

```python
class F00():
    @cached_property
    def test(self):
        # cached_property 将会把每个实例的属性存储到实例的__dict__中, 实例获取属性时, 将会优先从__dict__中获取，则不会再次调用方法内部的过程
        print(f'运行test方法内部过程')
        return 3
    
    @property
    def t(self):
        print('运行t方法内部过程')
        return 44
 
 
f = F00()
print(f.test)  # 第一次将会调用test方法内部过程
print(f.test)  # 再次调用将直接从实例中的__dict__中直接获取，不会再次调用方法内部过程
print(f.t)     # 调用方法内部过程取值
print(f.t)     # 调用方法内部过程取值
```

## cmp_to_key

在 list.sort 和 内建函数 sorted 中都有一个 key 参数

```python
x = ['hello','worl','ni']
x.sort(key=len)
print(x)
# ['ni', 'worl', 'hello']
```

## lru_cache

允许我们将一个函数的返回值快速地缓存或取消缓存。
该装饰器用于缓存函数的调用结果，对于需要多次调用的函数，而且每次调用参数都相同，则可以用该装饰器缓存调用结果，从而加快程序运行。
该装饰器会将不同的调用结果缓存在内存中，因此需要注意内存占用问题。

```python
from functools import lru_cache

# 静态数据
@lru_cache(maxsize=32)
def get_pep(num):
    'Retrieve text of a Python Enhancement Proposal'
    resource = f'https://peps.python.org/pep-{num:04d}'
    try:
        with urllib.request.urlopen(resource) as s:
            return s.read()
    except urllib.error.HTTPError:
        return 'Not Found'

>>> for n in 8, 290, 308, 320, 8, 218, 320, 279, 289, 320, 9991:
...     pep = get_pep(n)
...     print(n, len(pep))

>>> get_pep.cache_info()
CacheInfo(hits=3, misses=8, maxsize=32, currsize=8)


# 使用动态规划计算
@lru_cache(maxsize=30)  # maxsize参数告诉lru_cache缓存最近多少个返回值
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
print([fib(n) for n in range(10)])
fib.cache_clear()   # 清空缓存
```

## partial

用于创建一个偏函数，将默认参数包装一个可调用对象，返回结果也是可调用对象。
偏函数可以固定住原函数的部分参数，从而在调用时更简单。

```python
from functools import partial
 
int2 = partial(int, base=8)
print(int2('123'))
# 83
```

## partialmethod

对于python 偏函数partial理解运用起来比较简单，就是对原函数某些参数设置默认值，生成一个新函数。而如果对于类方法，因为第一个参数是 self，使用 partial 就会报错了。

partialmethod 返回一个新的 partialmethod 描述器，其行为类似 partial 但它被设计用作方法定义而非直接用作可调用对象。

```python
from functools import partialmethod
 
class Cell:
    def __init__(self):
        self._alive = False
    @property
    def alive(self):
        return self._alive
    def set_state(self, state):
        self._alive = bool(state)
 
    set_alive = partialmethod(set_state, True)
    set_dead = partialmethod(set_state, False)
 
    print(type(partialmethod(set_state, False)))
    # <class 'functools.partialmethod'>
 
c = Cell()
c.alive
# False
 
c.set_alive()
c.alive
# True
```

## reduce

函数的作用是将一个序列归纳为一个输出reduce(function, sequence, startValue)

```python
from functools import reduce
 
l = range(1,50)
print(reduce(lambda x,y:x+y, l))
# 1225
```

## singledispatch

单分发器，用于实现泛型函数。根据单一参数的类型来判断调用哪个函数。

```python
from functools import singledispatch

@singledispatch
def fun(text):
	print('String：' + text)
 
@fun.register(int)
def _(text):
	print(text)
 
@fun.register(list)
def _(text):
	for k, v in enumerate(text):
		print(k, v)
 
@fun.register(float)
@fun.register(tuple)
def _(text):
	print('float, tuple')
    
    
    
fun('i am is hubo')
fun(123)
fun(['a','b','c'])
fun(1.23)
print(fun.registry)	# 所有的泛型函数
print(fun.registry[int])	# 获取int的泛型函数
# String：i am is hubo
# 123
# 0 a
# 1 b
# 2 c
# float, tuple
# {<class 'object'>: <function fun at 0x106d10f28>, <class 'int'>: <function _ at 0x106f0b9d8>, <class 'list'>: <function _ at 0x106f0ba60>, <class 'tuple'>: <function _ at 0x106f0bb70>, <class 'float'>: <function _ at 0x106f0bb70>}
# <function _ at 0x106f0b9d8>
```

## singledispatchmethod

与泛型函数类似，可以编写一个使用不同类型的参数调用的泛型方法声明，根据传递给通用方法的参数的类型，编译器会适当地处理每个方法调用。

```python
class Negator:
    @singledispatchmethod
    def neg(self, arg):
        raise NotImplementedError("Cannot negate a")
 
    @neg.register
    def _(self, arg: int):
        return -arg
 
    @neg.register
    def _(self, arg: bool):
        return not arg
```

## total_ordering

是针对某个类如果定义了**lt**、**le**、**gt**、**ge**这些方法中的至少一个，使用该装饰器，则会自动的把其他几个比较函数也实现在该类中

```python
from functools import total_ordering
 
class Person:
    # 定义相等的比较函数
    def __eq__(self,other):
        return ((self.lastname.lower(),self.firstname.lower()) == 
                (other.lastname.lower(),other.firstname.lower()))
 
    # 定义小于的比较函数
    def __lt__(self,other):
        return ((self.lastname.lower(),self.firstname.lower()) < 
                (other.lastname.lower(),other.firstname.lower()))
 
p1 = Person()
p2 = Person()
 
p1.lastname = "123"
p1.firstname = "000"
 
p2.lastname = "1231"
p2.firstname = "000"
 
print p1 < p2  # True
print p1 <= p2  # True
print p1 == p2  # False
print p1 > p2  # False
print p1 >= p2  # False
```

## update_wrapper

使用 `partial` 包装的函数是没有`__name__`和`__doc__`属性的。
`update_wrapper` 作用：将被包装函数的`__name__`等属性，拷贝到新的函数中去。

```python
from functools import update_wrapper

def wrap2(func):
	def inner(*args):
		return func(*args)
	return update_wrapper(inner, func)
 
@wrap2
def demo():
	print('hello world')
 
print(demo.__name__)
# demo
```

## wraps

`warps` 函数是为了在装饰器拷贝被装饰函数的`__name__`。
就是在`update_wrapper`上进行一个包装 

```python
from functools import wraps

def wrap1(func):
	@wraps(func)	# 去掉就会返回inner
	def inner(*args):
		print(func.__name__)
		return func(*args)
	return inner
 
@wrap1
def demo():
	print('hello world')
 
print(demo.__name__)
# demo
```