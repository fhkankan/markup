# dockets

一个文本中

```
def sum(a, b):
    """
    >>> sum(1,4)
    5
    >>> sum(100,11)
    133
    """
    return a + b


if __name__ == '__main__':
    import doctest
    doctest.testmod()
```
测试代码单列
```
//sum.py
def sum(a,b):
    return a+b  111
    
//testsum.py
>>> from sum import sum1
>>> sum(1,4)
5
>>> sum1(100,11)
133

//test.py
import doctest
doctest.testfile('testsum.txt')
```