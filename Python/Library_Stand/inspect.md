[TOC]

# inspect

测试对象类型

## func

| func                             | dec                                                          |
| -------------------------------- | ------------------------------------------------------------ |
| `getmembers(object[,predicate])` | 返回由object的成员的(name,value)构成的列表，并且根据name进行排序。如果可选参数predicate是一个判断条件，只有value满足predicate条件的成员才会被返回。 |
| `getmodulename(path)`            | 通过输入一个路径返回模块名称。在Python中，一个py文件就是一个module，这需要与包(package)相区别，如果输入的path是一个package，则该方法会返回None。 |
| `ismodule(object)`               | 如果object是一个module就返回True，反之则返回False            |
| `isclass(object)`                | 如果object是一个class就返回True，反之则返回False             |
| `ismethod(object)`               | 如果object是一个方法则返回True，反之则返回False              |

