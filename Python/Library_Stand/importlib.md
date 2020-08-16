# importlib

旨在提供Python的import语法和(`__import__()`函数)的实现。另外，`importlib`提供了开发者可以创建自己的对象(即`importer`)来处理导入过程

还有一个`imp`模块提供了`import`语句接口，不过这个模块在Python3.4已经`deprecated`了。建议使用`importlib`来处理。

## 动态导入

`importlib`模块支持传递字符串来导入模块。我们先来创建一些简单模块一遍演示。我们在模块里提供了相同接口，通过打印它们自身名字来区分。我们分别创建了`foo.py`和`bar.py`，代码如下：

```python
def main():
    print(__name__)
```

现在我们尽需要使用importlib导入它们。我们来看看代码是如何实现的，确保该代码在刚才创建的两个文件的相同目录下。

```python
#importer
import importlib

def dynamic_import(module):
    return importlib.import_module(module)


if __name__ == "__main__":
    module = dynamic_import('foo')
    module.main()

    module2 = dynamic_import('bar')
    module2.main()
```

这里我们导入`importlib`模块，并创建了一个非常简单的函数`dynamic_import`。这个函数直接就调用了`importlib`的`import_module`方法，并将要导入的模块字符串传递作为参数，最后返回其结果。然后在主入口中我们分别调用了各自的`main`方法，将打印出各自的name.

```shell
$ python3 importer.py 
foo
bar
```

也许你很少会代码这么做，不过在你需要试用字符串作为导入路径的话，那么`importlib`就有用途了。

## 模块导入检查

Python有个众所周知的代码风格EAFP: Easier to ask forgiveness than permission.它所代表的意思就是总是先确保事物存在(例如字典中的键)以及在犯错时捕获。如果我们在导入前想检查是否这个模块存在而不是靠猜。 使用mportlib就能实现。

```python
import importlib.util

def check_module(module_name):
    """
    检查这个包是否可以导入而不必真的导入它
    """
    # 当传入不存在的模块时，find_spec函数将返回 None,否则返回模块的specification
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        print("Module: {} not found".format(module_name))
        return None
    else:
        print("Module: {} can be imported".format(module_name))
        return module_spec

def import_module_from_spec(module_spec):
    """
    在通过检查后导入模块，并返回模块名
    """
    # 导入模块
    # 方法一：直接将字符串作为参数调用import_module函数
    # 方法二：使用模块specification方式导入
    # import_module_from_spec函数接受check_module提供的模块specification作为参数，返回导入模块
    module = importlib.util.module_from_spec(module_spec)
    # 先导入后执行
    module_spec.loader.exec_module(module)
    return module

if __name__ == '__main__':
    module_spec = check_module('fake_module')
    module_spec = check_module('collections')
    if module_spec:
        module = import_module_from_spec(module_spec)
        # 使用dir来确保得到预期模块
        print(dir(module))
```

## 从源代码导入

可以使用util通过模块的名字和路径来导入模块。

```python
import importlib.util

def import_source(module_name):
    # 通过导入的模块获取到实际的路径和名字
    module_file_path = module_name.__file__
    module_name = module_name.__name__
	# 将信息传递给sec_from_file_location函数,返回模块的specification
    module_spec = importlib.util.spec_from_file_location(
        module_name, module_file_path
    )
    # 通过模块的specification导入模块
    module = importlib.util.module_from_spec(module_spec)
    # 执行模块
    module_spec.loader.exec_module(module)
    print(dir((module)))

    msg = 'The {module_name} module has the following methods {methods}'
    print(msg.format(module_name=module_name, methods=dir(module)))


if __name__ == "__main__":
    # 导入logging
    import logging
    # 将模块传递给了import_source函数
    import_source(logging)
```