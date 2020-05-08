# C语言扩展

## 简单扩展

- C语言

Extest.c程序

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// 绝对路径
#include "/usr/include/python3.5/Python.h"
// 另一种方式
#include <Python.h>

// 在C语言的Extest_hello函数中打印Hello,Python-IoT!字符后，通过Py_RETURN_NONE返回None
static PyObject * Extest_hello(PyObject*self, PyObject*args){
    print("Python-IoT\n");
    Py_RETURN_NONE;
}

static PyMethodDef ExtestMethods[] = {
    // hello为python调用函数名称，Extest_hello为C语言内部真实的函数名称
    {"hello", Extest_hello, METH_VARARGS},
    {NULL, NULL},
};

static struct PyModuleDef ExtestModule = {
    PyModuleDef_HEAD_INIT,
    // Extest是C语言模块暴露给Python的接口名称
    "Extest",
    NULL,
    -1,
    // ExtestMethods为python提供给模块C函数名称的映射表
    ExtestMethods
};

PyMODINIT_FUNC PyInt_Extest(void){
    return PyModule_Create(&ExtestModule)
}
```

- 编译

```
// 在当前目录中编译生成Extest.c的动态文件Extest.so
gcc -fPIC -shared Extest.c -o Extest.so
```

- 安装

需要将Extest.so文件安装到sys.path路径中

```
# 编写setup.py文件
from distutils.core import setup, Extension
setup(name='Extest', version='1.0', ext_modules=[Extension('Extest',['Extest.c'])])

# 执行命令安装
python ./setup.py install
```

执行完成后，将Extest安装到`/usr/local/lib/python3.5/dist-packages`目录，此目录是sys.path路径之一，可导入Extest模块

- 使用扩展库

```
import Extest

Extest.hello()
```

## 传递整型参数

Extest.c程序

```python
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// 绝对路径
#include "/usr/include/python3.5/Python.h"
// 另一种方式
#include <Python.h>

int add(int n){
    return 100+n;
}

// 在C语言的Extest_hello函数中打印Hello,Python-IoT!字符后，通过Py_RETURN_NONE返回None
static PyObject * Extest_hello(PyObject*self, PyObject*args){
    print("Python-IoT\n");
    Py_RETURN_NONE;
}

static PyObject * Extest_add(PyObject*self, PyObject*args){
    int res;
    int num;
    PyObject*retval;
    // i表示需要传递进来的参数类型为整型，如果是，就赋值给num，若不是，返回NULL；
    res = PyArg_ParseTuple(args, "i", &num);
    if(!res){
        return NULL;
    }
    res = add(num);
    // 需要把C语言中计算的结果转成python对象，i代表整数对象类型
    retval = (PyObject *)Py_BuildValue("i", res);
    return retval;
}

static PyMethodDef ExtestMethods[] = {
    // hello为python调用函数名称，Extest_hello为C语言内部真实的函数名称
    {"hello", Extest_hello, METH_VARARGS},
    {"add", Extest_add, METH_VARARGS},
    {NULL, NULL},
};

static struct PyModuleDef ExtestModule = {
    PyModuleDef_HEAD_INIT,
    // Extest是C语言模块暴露给Python的接口名称
    "Extest",
    NULL,
    -1,
    // ExtestMethods为python提供给模块C函数名称的映射表
    ExtestMethods
};

PyMODINIT_FUNC PyInt_Extest(void){
    return PyModule_Create(&ExtestModule)
}
```

测试

```python
import Extest

Extest.add(1)
Extest.add(2)
Extest.hello()
```

## 传递字符串参数

Extest.c程序

```python
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// 绝对路径
#include "/usr/include/python3.5/Python.h"
// 另一种方式
#include <Python.h>

int add(int n){
    return 100+n;
}

char *reverse(char *s){
    register char t;
    char *p = s;
    char *q = (s + (strlen(s) - 1))
    while(p < q){
        t = *p;
        *p++ = *q;
        *q-- = t;
    }
    return s;
}

// 在C语言的Extest_hello函数中打印Hello,Python-IoT!字符后，通过Py_RETURN_NONE返回None
static PyObject * Extest_hello(PyObject*self, PyObject*args){
    print("Python-IoT\n");
    Py_RETURN_NONE;
}

static PyObject * Extest_add(PyObject*self, PyObject*args){
    int res;
    int num;
    PyObject*retval;
    // i表示需要传递进来的参数类型为整型，如果是，就赋值给num，若不是，返回NULL；
    res = PyArg_ParseTuple(args, "i", &num);
    if(!res){
        return NULL;
    }
    res = add(num);
    // 需要把C语言中计算的结果转成python对象，i代表整数对象类型
    retval = (PyObject *)Py_BuildValue("i", res);
    return retval;
}

static PyObject * Extest_reverse(PyObject*self, PyObject*args){
    char *original;
    // s表示需要传递进来的参数类型为字符串，如果是，就赋值给original，若不是，返回NULL
    if(!PyArg_ParseTuple(args, "s", &original)){
        return NULL;
    }
    // 需要把结果转换成python对象，s代表字符串对象类型
    return (PyObject *)Py_BuildValue("s", reverse(original));
}

static PyMethodDef ExtestMethods[] = {
    // hello为python调用函数名称，Extest_hello为C语言内部真实的函数名称
    {"hello", Extest_hello, METH_VARARGS},
    {"add", Extest_add, METH_VARARGS},
    {"reverse", Extest_reverse, METH_VARARGS},
    {NULL, NULL},
};

static struct PyModuleDef ExtestModule = {
    PyModuleDef_HEAD_INIT,
    // Extest是C语言模块暴露给Python的接口名称
    "Extest",
    NULL,
    -1,
    // ExtestMethods为python提供给模块C函数名称的映射表
    ExtestMethods
};

PyMODINIT_FUNC PyInt_Extest(void){
    return PyModule_Create(&ExtestModule)
}
```

测试

```
import Extest

Extest.reverse('abcdefg')
Extest.add(1)
Extest.hello()
dir(Extest)
```

