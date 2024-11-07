# Shell脚本高级

## 创建函数

### 简单函数

定义

```shell
# 方式一
function name {
    commands
}
# 方式二
name() {
	commands
}
```

使用

```shell
name
```

### 变量作用域

默认情况下，在脚本中定义的任何变量都是全局变量。全局变量是在shell脚本中任何地方都有效的变量。

函数中的变量和函数外的变量都是全局变量可能会带来变量值使用的混乱。在函数中使用局部变量方法

```shell
function func1 {
	local temp=$[ $value + 5 ]  # temp局部变量
	result=$[ $temp * 2]  # result全局变量
}
temp=4
value=6
func1
echo "The result is $result"
if [ $temp -gt $value ]
then
	echo "temp is larger"
else
  echo "temp is smaller"
fi
```

### 参数返回值

- 参数

普通参数

```shell
fucntion addem {
		if [ $# -eq 0 ] || [ $# -gt 2 ]
		then
			echo -1
		elif [ $# -eq 1]
		then
			echo $[ $1 + $1]
		else
			echo $[ $1 + $2]
		fi
}
echo -n "Adding 10 and 15: "
value=$(addem 10 15)
echo $value
```

命令行参数

```shell
# 命令行参数传入函数中
function func1 {
	echo $[ $1 * $2]
}
value=$(func1 $1 $2)  # 将命令行中参数传递给函数
```

数组参数

```shell
# 将数组变量当作单个参数传递不会起作用，需要使用*
function test {
  local newarray
  newarray=('echo "$@"')
  echo "The new arrsy value is: ${newarray[*]}"
}
myarray=(1 2 3 4)
echo "The original array is ${myarray[*]}"
test ${myarray[*]}
```

- 返回值

默认退出状态码

```shell
# 默认情况下，函数的退出状态码时函数中最后一条命令返回的退出状态码。
# 在函数执行结束后，可以使用标准标量`$?`来确定函数的退出码。

function db1{
	echo "hi"
}
db1
echo "the value is $?"
```

使用retrun修改退出状态码

```shell
# reutrn命令允许指定一个整数值来定义函数的退出状态码。
# 若要取得函数返回值，需要在函数刚结束时，使用`$?`，退出状态码必须是0~255。

function db1{
	echo "hi"
	return 22
}
db1
echo "the value is $?"
```

使用函数输出

```shell
# 普通输出
function db1{
	echo "hi"
}
result = $(db1)
echo "the value is $result"

# 数组输出
function test {
	local origarray
  local newarray
  local elements
  local i
  origarray=($(echo "$@"))
  newarray=($(echo "$@"))
  elements=$[ $# - 1]
  for ((i=0; i <= $elements; i++))
  {
  	newarray[$i]=$[ ${origarray[$i]} * 2 ]
  }
  echo ${newarray[*]}
}
myarray=(1 2 3 4)
echo "The original array is ${myarray[*]}"
arg1=$(echo ${myarray[*]})
resutl=($(test $arg1))
echo "The new array is: ${result[*]}"
```

### 公共函数

- 创建公共函数脚本

在不同的脚本中使用共同的函数，可以在脚本中引用公共函数文件。

和环境变量一样，shell函数仅在定义它的shell会话中有效。若在shell命令行界面提示符下运行公共函数文件，shell会创建一个新的shell并在其中运行这个脚本。当你运行另外一个需要用到这些函数的脚本时，是无法使用这些公共函数的。

```
# 会在不同的shell会话中
./common.sh 
./test.sh
```

使用库函数的方式是使用`source`命令，会在当前shell上下文中执行命令，而不是创建一个新的shell。其快捷方式`../common.sh`

```
. ./common.sh
```

- 创建命令行函数

命令行上的函数可以被临时公共使用，但是退出shell时，函数就消失了。

```shell
# 单行方式
$ fucntion divm {echo $[ $1 / $2]}
$ divm 100 5
# 多行方式
$ function divm {
> echo $[ $1 / $2]
> }
> divm 100 5
```

- 在`.bashrc`文件创建函数

bash shell在每次启动时都会在主目录下查找`.bashrc`，不论是交互式还是从现有shell中启动新的shell，所以在此创建的函数，可以简单公用。

```shell
# 直接定义
# 在文件末尾加上自定义函数
function func1 {}
# 读取函数文件
. /home/demo...
```

## 其他部分

图形化桌面环境、sed、gawk、其他shell

