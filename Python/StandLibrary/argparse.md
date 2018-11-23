# argparse

argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。当然，Python 也有第三方的库可用于命令行解析，而且功能也更加强大，比如 [docopt](http://docopt.org/)，[Click](http://click.pocoo.org/5/)

```
import argparse
```

## 基础

三步骤

```
创建 ArgumentParser() 对象
调用 add_argument() 方法添加参数
使用 parse_args() 解析添加的参数
```

prog.py

```
import argparse

parser = argparse.ArgumentParser()
parser.parse_args()
```

运行

```
$ python3 prog.py #运行脚本，无输出
$ python3 prog.py --help	#系统默认帮助信息
$ python3 prog.py --verbose #报错
$ python3 prog.py foo  #报错
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
$ python prog.py  #报错
$ python prog.py -h #系统帮助信息
$ python prog.py hahaha #输出输入信息hahaha
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
$ python prog.py -v 1 #通过-v指定参数值
$ python prog.py --verbosity 2 #通过--verbosity指定参数值
$ python prog.py -h #通过-h来打印帮助信息
$ python prog.py -v #没有给-v指定参数值，报错
```

## 混合使用

prog.py

```python
import argparse
 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max,help='sum the integers (default: find the max)')
 
args = parser.parse_args()
print args.accumulate(args.integers)
```

运行

```
$ python prog.py  # 报错
$ python prog.py 1 2 3 4  #输出4
$ python aprog.py 1 2 3 4 --sum  #输出10
```

## 互斥组

prog.py

```python
# encoding: utf-8
import argparse

# 整个程序的帮助文档
parser = argparse.ArgumentParser(description="calculate X to the power of Y")
# 定义了互斥组
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("x", type=int, help="the base")
parser.add_argument("y", type=int, help="the exponent")
args = parser.parse_args()
answer = args.x**args.y

if args.quiet:
    print answer
elif args.verbose:
    print "{} to the power {} equals {}".format(args.x, args.y, answer)
else:
    print "{}^{} == {}".format(args.x, args.y, answer)
```

运行

```
$ python prog.py -h  #输出自定义帮助信息
$ python prog.py 4 2  #输出4^2 == 16
$ python prog.py 4 2 -v  #输出4 to the power 2 equals 16
$ python prog.py 4 2 -q  #输出16
$ python prog.py 4 2 -q -v  #报错
```

## add_argument()

```
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

参数定义

```
name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。

action - 命令行遇到参数时的动作，默认值是 store。
	store_const，表示赋值为const；
 	append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
	append_const，将参数规范中定义的一个值保存到一个列表；
	count，存储遇到的次数；此外，也可以继承 argparse.Action 自定义参数解析；
	
nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于 Positional argument用 default，对于 Optional argument 使用 const；或者是 * 号，表示 0 或多个参数；或者是 + 号表示 1 或多个参数。

const - action 和 nargs 所需要的常量值。
default - 不指定参数时的默认值。
type - 命令行参数应该被转换成的类型。
choices - 参数可允许的值的一个容器。
required - 可选参数是否可以省略 (仅针对可选参数)。
help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
```

## parse_args()

