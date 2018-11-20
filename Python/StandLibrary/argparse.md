# argparse

命令行解析包,为py文件封装好可以选择的参数，使他们更加灵活，丰富

```
import argparse
```

## 基础

prog.py

```
import argparse

parser = argparse.ArgumentParser()
parser.parse_args()
```

运行

```
$ python3 prog.py # 运行脚本，无输出
$ python3 prog.py --help	# 系统默认帮助信息
$ python3 prog.py --verbose # 报错
$ python3 prog.py foo  # 报错
```

## 位置参数

prog.py

```
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("echo")
args = parser.parse_args()
print args.echo
```

运行

```
$ python prog.py  # 报错
$ python prog.py -h # 系统默认帮助信息
$ python prog.py hahaha # 输出输入信息hahaha
```

## 可选参数

```
# 方式一：
通过一个-来指定的短参数，如-h；
# 方式二：
通过--来指定的长参数，如--help
可以同存，也可只存一个
```

prog.py

```
import argparse


parser = argparse.ArgumentParser()
# 定义可选参数-v或--verbosity，通过解析后，其值保存在args.verbosity变量中
parser.add_argument("-v", "--verbosity", help="increase output verbosity")  
args = parser.parse_args()
if args.verbosity:
        print "verbosity turned on"
```

运行

```
$ python prog.py -v 1 # 通过-v指定参数值
$ python prog.py --verbosity # 通过--verbosity指定参数值
$ python prog.py -h # 通过-h来打印帮助信息
$ python prog.py -v # 没有给-v指定参数值，报错
```

