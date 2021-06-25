# python调用shell

[参考](https://www.jb51.net/article/186301.htm)

## os

- `os.system(cmd)`

使用os模块的system方法：`os.system(cmd)`，其返回值是shell指令运行后返回的状态码，int类型，0表示shell指令成功执行，256表示未找到，该方法适用于shell命令不需要输出内容的场景。

示例

```python
import os

# 1. 执行命令
val = os.system('ls -al')
print(val)

# 2. 执行脚本
"""
hello.sh
"""
#!/bin/bash
echo "hello world ${1} ${2}"
exit 0

"""
hello.py
"""
import os
import sys

if(len(sys.argv) < 3):
    print("please input two arguments")
    sys.exit(1)
arg0 = sys.argv[1]
arg1 = sys.argv[2]

os.system("./hello.sh" + arg0 + " " + arg1)
 
"""
执行
"""
python hello.py zhang san
```

- `os.popen()`

使用`os.popen()`，该方法以文件的形式返回shell指令运行后的结果，需要获取内容时可使用`read()`或`readlines()`方法。

```python
import os

val = os.popen('ls -al')
for temp in val.readlines():
    print(temp)
```

## commands

有三个方法

```python
commands.getstatusoutput(cmd)  # 其以字符串的形式返回的是输出结果和状态码，即（status,output）。

commands.getoutput(cmd)  # 返回cmd的输出结果。

commands.getstatus(file)  # 返回ls -l file的执行结果字符串，调用了getoutput，不建议使用此方法
```

示例

```python
import commands

(status, output) = commands.getstatusoutput("ls -l")
print(status, output, 'sep'="\n")

output = commands.getoutput("ls -l")
print(output)

status = commands.getstatus("mysql")
print(status)
```

## subprocess

允许创建很多子进程，创建的时候能指定子进程和子进程的输入、输出、错误输出管道，执行后能获取输出结果和执行状态。

```python
subprocess.run()  # python3.5中新增的函数， 执行指定的命令， 等待命令执行完成后返回一个包含执行结果的CompletedProcess类的实例。

subprocess.call()  # 执行指定的命令， 返回命令执行状态， 功能类似os.system（cmd）。

subprocess.check_call()  # python2.5中新增的函数, 执行指定的命令, 如果执行成功则返回状态码， 否则抛出异常。
```

