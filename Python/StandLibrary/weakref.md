# 弱引用

和许多其它的高级语言一样，Python使用了垃圾回收器来自动销毁那些不再使用的对象。每个对象都有一个引用计数，当这个引用计数为0时Python能够安全地销毁这个对象。

引用计数会记录给定对象的引用个数，并在引用个数为零时收集该对象。由于一次仅能有一个对象被回收，引用计数无法回收循环引用的对象。

一组相互引用的对象若没有被其它对象直接引用，并且不可访问，则会永久存活下来。一个应用程序如果持续地产生这种不可访问的对象群组，就会发生内存泄漏。

在对象群组内部使用弱引用（即不会在引用计数中被计数的引用）有时能避免出现引用环，因此弱引用可用于解决循环引用的问题。

在计算机程序设计中，弱引用，与强引用相对，是指不能确保其引用的对象不会被垃圾回收器回收的引用。一个对象若只被弱引用所引用，则可能在任何时刻被回收。弱引用的主要作用就是减少循环引用，减少内存中不必要的对象存在的数量。

使用weakref模块，你可以创建到对象的弱引用，Python在对象的引用计数为0或只存在对象的弱引用时将回收这个对象。

##创建弱引用

你可以通过调用weakref模块的ref(obj[,callback])来创建一个弱引用，obj是你想弱引用的对象，callback是一个可选的函数，当因没有引用导致Python要销毁这个对象时调用。回调函数callback要求单个参数（弱引用的对象）。

一旦你有了一个对象的弱引用，你就能通过调用弱引用来获取被弱引用的对象。

```
>>>>　import　sys
>>>　import　weakref
>>>　class　Man:
　　def　__init__(self,name):
　　　　print　self.name = name
　　　　
>>>　o　=　Man('Jim')
>>>　sys.getrefcount(o)   
2
>>>　r　=　weakref.ref(o)　#　创建一个弱引用
>>>　sys.getrefcount(o)　#　引用计数并没有改变
2
>>>　r
<weakref　at　00D3B3F0;　to　'instance'　at　00D37A30>　#　弱引用所指向的对象信息
>>>　o2　=　r()　#　获取弱引用所指向的对象
>>>　o　is　o2
True
>>>　sys.getrefcount(o)
3
>>>　o　=　None
>>>　o2　=　None
>>>　r　#　当对象引用计数为零时，弱引用失效。
<weakref　at　00D3B3F0;　dead>de>
```

上面的代码中，我们使用sys包中的`getrefcount()`来查看某个对象的引用计数。需要注意的是，当使用某个引用作为参数，传递给`getrefcount()`时，参数实际上创建了一个临时的引用。因此，getrefcount()所得到的结果，会比期望的多1。

一旦没有了对这个对象的其它的引用，调用弱引用将返回None，因为Python已经销毁了这个对象。 注意：大部分的对象不能通过弱引用来访问。

weakref模块中的getweakrefcount(obj)和getweakrefs(obj)分别返回弱引用数和关于所给对象的引用列表。

弱引用对于创建对象(这些对象很费资源)的缓存是有用的。

##创建代理对象

代理对象是弱引用对象，它们的行为就像它们所引用的对象，这就便于你不必首先调用弱引用来访问背后的对象。通过weakref模块的proxy(obj[,callback])函数来创建代理对象。使用代理对象就如同使用对象本身一样：

```
import weakref

class Man:
    def __init__(self, name):
        self.name = name
    def test(self):
        print "this is a test!"

def callback(self):
    print "callback"
    
o = Man('Jim')
p = weakref.proxy(o, callback)
p.test()
o=None
p.test()
```

callback参数的作用和ref函数中callback一样。在Python删除了一个引用的对象之后，使用代理将会导致一个weakref.ReferenceError错误。

##循环引用

前面说过，使用弱引用，可以解决循环引用不能被垃圾回收的问题。
首先我们看下常规的循环引用，先创建一个简单的Graph类，然后创建三个Graph实例：

```
# -*- coding:utf-8 -*-
import weakref
import gc
from pprint import pprint


class Graph(object):
    def __init__(self, name):
        self.name = name
        self.other = None

    def set_next(self, other):
        print "%s.set_next(%r)" % (self.name, other)
        self.other = other

    def all_nodes(self):
        yield self
        n = self.other
        while n and n.name !=self.name:
            yield n
            n = n.other
        if n is self:
            yield n
        return

    def __str__(self):
        return "->".join(n.name for n in self.all_nodes())

    def __repr__(self):
        return "<%s at 0x%x name=%s>" % (self.__class__.__name__, id(self), self.name)

    def __del__(self):
        print "(Deleting %s)" % self.name

def collect_and_show_garbage():
    print "Collecting..."
    n = gc.collect()
    print "unreachable objects:", n
    print "garbage:",
    pprint(gc.garbage)


def demo(graph_factory):
    print "Set up graph:"
    one = graph_factory("one")
    two = graph_factory("two")
    three = graph_factory("three")
    one.set_next(two)
    two.set_next(three)
    three.set_next(one)

    print
    print "Graph:"
    print str(one)
    collect_and_show_garbage()

    print
    three = None
    two = None
    print "After 2 references removed"
    print str(one)
    collect_and_show_garbage()

    print
    print "removeing last reference"
    one = None
    collect_and_show_garbage()


gc.set_debug(gc.DEBUG_LEAK)
print "Setting up the cycle"
print 
demo(Graph)
print
print "breaking the cycle and cleaning up garbage"
print
gc.garbage[0].set_next(None)
while gc.garbage:
    del gc.garbage[0]
print collect_and_show_garbage()
```

这里使用了python的gc库的几个方法， 解释如下：

- gc.collect() 收集垃圾
- gc.garbage 获取垃圾列表
- gc.set_debug(gc.DBEUG_LEAK) 打印无法看到的对象信息

运行结果如下：

```
Setting up the cycle

Set up graph:
one.set_next(<Graph at 0x25c9e70 name=two>)
two.set_next(<Graph at 0x25c9e90 name=three>)
three.set_next(<Graph at 0x25c9e50 name=one>)

Graph:
one->two->three->one
Collecting...
unreachable objects:g 0
garbage:[]

After 2 references removed
one->two->three->one
Collecting...
unreachable objects: 0
garbage:[]

removeing last reference
Collecting...
unreachable objects: 6
garbage:[<Graph at 0x25c9e50 name=one>,
 <Graph at 0x25c9e70 name=two>,
 <Graph at 0x25c9e90 name=three>,
 {'name': 'one', 'other': <Graph at 0x25c9e70 name=two>},
 {'name': 'two', 'other': <Graph at 0x25c9e90 name=three>},
 {'name': 'three', 'other': <Graph at 0x25c9e50 name=one>}]

breaking the cycle and cleaning up garbage

one.set_next(None)
(Deleting two)
(Deleting three)
(Deleting one)
Collecting...
unreachable objects: 0
garbage:[]
None
[Finished in 0.4s]c: uncollectable <Graph 025C9E50>
gc: uncollectable <Graph 025C9E70>
gc: uncollectable <Graph 025C9E90>
gc: uncollectable <dict 025D3030>
gc: uncollectable <dict 025D30C0>
gc: uncollectable <dict 025C1F60>
```

从结果中我们可以看出，即使我们删除了Graph实例的本地引用，它依然存在垃圾列表中，不能回收。
接下来创建使弱引用的WeakGraph类：

```
class WeakGraph(Graph):
    def set_next(self, other):
        if other is not None:
            if self in other.all_nodes():
                other = weakref.proxy(other)
        super(WeakGraph, self).set_next(other)
        return
demo(WeakGraph)
```

结果如下：

```
Setting up the cycle

Set up graph:
one.set_next(<WeakGraph at 0x23f9ef0 name=two>)
two.set_next(<WeakGraph at 0x23f9f10 name=three>)
three.set_next(<weakproxy at 023F8810 to WeakGraph at 023F9ED0>)

Graph:
one->two->three
Collecting...
unreachable objects:Traceback (most recent call last):
  File "D:\apps\platform\demo\demo.py", line 87, in <module>
    gc.garbage[0].set_next(None)
IndexError: list index out of range
 0
garbage:[]

After 2 references removed
one->two->three
Collecting...
unreachable objects: 0
garbage:[]

removeing last reference
(Deleting one)
(Deleting two)
(Deleting three)
Collecting...
unreachable objects: 0
garbage:[]

breaking the cycle and cleaning up garbage

[Finished in 0.4s with exit code 1]
```

上面的类中，使用代理来指示已看到的对象，随着demo()删除了对象的所有本地引用，循环会断开，这样垃圾回收期就可以将这些对象删除。

因此我们我们在实际工作中如果需要用到循环引用的话，尽量采用弱引用来实现。

##缓存对象

`ref`和`proxy`都只可用与维护单个对象的弱引用，如果想同时创建多个对象的弱引用咋办？这时可以使用`WeakKeyDictionary`和`WeakValueDictionary`来实现。

`WeakValueDictionary`类，顾名思义，本质上还是个字典类型，只是它的值类型是弱引用。当这些值引用的对象不再被其他非弱引用对象引用时，那么这些引用的对象就可以通过垃圾回收器进行回收。
下面的例子说明了常规字典与`WeakValueDictionary`的区别。

```
# -*- coding:utf-8 -*-
import weakref
import gc
from pprint import pprint

gc.set_debug(gc.DEBUG_LEAK)


class Man(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '<Man name=%s>' % self.name

    def __del__(self):
        print "deleting %s" % self


def demo(cache_factory):
    all_refs = {}
    print "cache type:", cache_factory
    cache = cache_factory()
    for name in ["Jim", 'Tom', 'Green']:
        man = Man(name)
        cache[name] = man
        all_refs[name] = man
        del man
    print "all_refs=",
    pprint(all_refs)
    print
    print "before, cache contains:", cache.keys()
    for name, value in cache.items():
        print "%s = %s" % (name, value)
    print "\ncleanup"
    del all_refs
    gc.collect()

    print
    print "after, cache contains:", cache.keys()
    for name, value in cache.items():
        print "%s = %s" % (name, value)
    print "demo returning"
    return

demo(dict)
print

demo(weakref.WeakValueDictionary)
```

结果如下所示：

```
cache type: <type 'dict'>
all_refs={'Green': <Man name=Green>, 'Jim': <Man name=Jim>, 'Tom': <Man name=Tom>}

before, cache contains: ['Jim', 'Green', 'Tom']
Jim = <Man name=Jim>
Green = <Man name=Green>
Tom = <Man name=Tom>

cleanup

after, cache contains: ['Jim', 'Green', 'Tom']
Jim = <Man name=Jim>
Green = <Man name=Green>
Tom = <Man name=Tom>
demo returning
deleting <Man name=Jim>
deleting <Man name=Green>
deleting <Man name=Tom>

cache type: weakref.WeakValueDictionary
all_refs={'Green': <Man name=Green>, 'Jim': <Man name=Jim>, 'Tom': <Man name=Tom>}

before, cache contains: ['Jim', 'Green', 'Tom']
Jim = <Man name=Jim>
Green = <Man name=Green>
Tom = <Man name=Tom>

cleanup
deleting <Man name=Jim>
deleting <Man name=Green>

after, cache contains: []
demo returning

[Finished in 0.3s]
```