# shell脚本基础

## 概述

### shell类型

```shell
cat /etc/default/useradd
# 里面有
SHELL=/bin/bash
```

### 创/执/退

- 常规创建

```shell
# 创建工具
vim/vi

# 脚本注释
单行注释	---> # 注释内容
多行注释	---> :<<! 注释内容！

# 示例
#!/bin/bash
# demo
echo '1'
:<<!
echo '2'
!
echo '3'
```

- 脚本执行

```shell
# 脚本文件本身没有可执行权限或者脚本首行没有命令解释器(推荐)
bash /path/to/script-name	或 /bin/bash /path/to/script-name

# 脚本文件具有可执行权限时使用
/path/to/script-name		或 ./script-name

# 加载shell脚本文件内容
source script-name			或 . script-name
```

- 退出脚本

退出状态码

```shell
# 查看上个命令退出状态码
date
echo $?	

# 退出状态码值
0	# 命令成功结束
1  	# 一般性未知错误
2	# 不适合的shell命令
126	# 命令不可知性
127	# 没找到命令
128	# 无效的退出参数
128+x	# 与linux信号x相关的严重错误
130		# 通过Ctrl+C终止的命令
255		# 正常范围之外的退出状态码
```

脚本中指定退出状态码

```shell
exit 5
```

### 开发规范

```
- 脚本命令要有意义，文件后缀.sh
- 脚本文件首行是且必须是脚本解释器`#!bin/bash`
- 脚本文件解释器后面要有脚本的基本信息等内容
- 脚本文件常见执行方式`bash 脚本名`
- 脚本内容执行：从上到下，一次执行
- 代码书写优秀习惯(成对内容一次性写完，[]两端要有空格，流程控制语句一次性写完 )
- 通过缩进让代码容易易读
```
## 使用变量

- 常用变量

环境变量

```shell
# 查看
env
set

# 定义
# 方法一：
变量=值
export 变量
# 方法二：
export 变量=值
```

用户变量

```shell
# 临时存储数据并在整个脚本中使用，可以是任何由字母数字或下划线组成的文本字符串，长度不超过20个
# 整体,中间无特殊字符
变量名=变量值
# 原样输出
变量名='变量值'
# 先解析，将结果与其他合成整体
变量名="变量值"
```

内置变量

```shell
$0	# 获取当前执行的shell脚本文件名，包括脚本路径
$n	# 获取当前执行的shell脚本的第n个参数值，0表示文件名，大于9就$(10)
$#	# 获取当前shell命令中参数的总个数
$?	# 获取执行上一个指令的返回值(0表示成功，非0为失败)
$$  # 当前shell的PID
```

数组变量

```shell
# 数组是能够存储多个值的变量

# 创建
# 把值放在括号里，值与值之间用空格分隔
mytest=(one two three four five)

# 取值
echo $mytest  # 只能取出第一个元素one
echo ${mytest[2]}  # 取出特定的第3个元素three
echo ${mytest[*]}  # 取出整个变量的所有值

# 修改
mytest[2]=seven
echo ${mytest[*]}

# 删除
unset mytest  # 删除整个数组
unset mytest[2]  # 将数组上索引为2位置上的值置为空
```

- 变量操作

```shell
# 查看变量
# 方式一
$变量名
# 方式二
"$变量名"
# 方式三
${变量名}
# 方式四(推荐)
"${变量名}"

# 命令替换
变量名=`命令`
变量名=$(命令)
# 执行流程
1.执行'后者$()范围内的命令
2.将命令执行后的结果，赋值给新的变量名

# 取消变量
unset 变量名

# 默认值
# 若变量有内容，输出变量值；若变量内无内容，输出默认值
${变量名:-默认值}
# 无论变量是否有内容，均输出默认值
${变量名+默认值}

# 字符串精确截取
${变量名：起始位置：截取长度}
```

常量与变量拼接

```
a = "12${i}"
```

## 表达式

### 测试语句

```shell
# 方式一
test 条件表达式
# 方式二
[ 条件表达式 ]

# 条件成立，状态返回值是0，条件不成立，状态返回值是1
```

### 条件表达式

- 逻辑表达式

```shell
命令1 && 命令2
# 若命令1执行成功，才执行命令2；若命令1执行失败，那么命令2也不执行

命令1 || 命令2
# 若命令1执行成功，则命令2不执行；若命令1执行失败，则命令2执行
```

- 文件比较

```shell
-d file		# 检查file是否存在并是一个目录
-e file		# 检查file是否存在
-f file		# 检查file是否存在并是一个文件
-r file		# 检查file是否存在并可读
-s file		# 检查file是否存在并非空
-w file		# 检查file是否存在并可写
-x file		# 检查file是否存在并可执行
-O file		# 检查file是否存在并属当前用户所有
-G file		# 检查file是否窜在并默认组与当前用户相同

file1 -nt file2	# 检查file1是否比file2新
file1 -ot file2	# 检查file1是否比file2旧
```

- 数值比较

```shell
# 主要根据给定的两个值，判断第一个与第二个数的关系，如是否大于、小于、等于第二个数。常见选项如下：
n1 -eq n2	# 相等
n1 -ge n2	# 大于等于
n1 -gt n2 	# 大于
n1 -le n2   # 小于等于
n1 -lt n2 	# 小于
n1 -ne n2 	# 不等于
```

- 字符串比较

```shell
str1 = str2		# str1和str2字符串内容一致
str1 != str2	# str1和str2字符创内容不一致
str1 < str2		# 检查str1是否比str2小
str1 > str2		# 检查str1是否比str2大
-n str1			# 检查str1的长度是否为非0
-z str1			# 检查str1的到长度是否为0

# 注意
# 大于小于号必须转义，否则shell会把他们当作重定向符号，把字符串值当作文件名
# 大于小于顺序和sort命令所采用不同：比较测试中大写字母小于小写字母，sort中相反
```

### 计算表达式

```shell
# 方式一：
$(( 计算表达式 ))	# $((100/5))

# 方式二：
let 计算表达式		# let i=i+7
```

## 常见符号

- 重定向

```shell
# 输出重定向
>		# 表示将符号左侧的内容，以覆盖的方式输入到右侧文件中
>>		# 表示将符号左侧的内容，以追加的方式输入到右侧文件的末尾行中

# 输入重定向
<		# 表示将文件的内容重定向到命令
<<		# 内联输入重定向，无需使用文件进行重定向，只需在命令行中指定用于输入重定向的数据即可
```

- 管道符

```shell
|		# 管道符左侧命令执行的结果，传递给管道右侧的命令使用
```

- 后台展示

```
&		---> 将一个命令从前台转到后台执行
```

- 全部信息符号

```
2>&1
1		--->表示正确输出的信息
2		--->表示错误输出的信息
2>&1	--->代表所有输出的信息
```

- 括号

```shell
((expression))
# 双括号允许在比较过程中使用高级数学表达式
val++	# 后增
val--	# 后减
++val	# 先增
--val	# 先减
!		# 逻辑求反
~		# 位求反
**		# 幂运算
<<		# 左位移
>>		# 右位移
&		# 位布尔和
|		# 位布尔或
&&		# 逻辑和
||		# 逻辑或

[[expression]]
# 双方括号提供了对字符串比较的高级特性
```

## 数学运算

```shell
# expr
expr 1 + 5

# 方括号
var1 = $[1 + 5]
echo var1

# 浮点解决方案
# bash脚本中数学运算符只支持整数运算，若要进行浮点，需要使用内建的bash计算器
# 在命令中使用
bc		# 进入bash计算器
quit	# 退出bash计算器
scale=4	# 设置小数点位数，默认为0
# 在脚本中使用
# 简单计算
variable = $(echo "options; expression" | bc)  # 格式
var1 = $(echo "scale=4, 3.44/5" | bc)
echo the answer is $var1
# 复杂计算
# 方式一：将表达式存放到文件中，使用<将一个文件重定向到bc命令
var1=$(bc < test)
# 方式二：使用<<直接在命令行中重定向数据
variable=$(bc << EOF
options
statements
expressions
EOF  # EOF文本字符串标识了内联重定向数据的起止
)
# 示例
var1=10.46
var2=43.67
var3=33.2
var4=71
var5=$(bc <<EOF
sacle = 4
a1 = ($var1 * $var2)
b1 = ($var3 * $var4)
a1 + b1
EOF
)
```

## 流程控制

### 选择

- 单分支if

```
if [条件]
then
	指令
fi
```

- 双分支if

```
if [条件]
then
	指令1
else
	指令2
fi	
```

- 多分支if

```
if	[条件1]
then
	指令1
elif [条件2]
then
	指令2
else
	指令3
fi
```

- case选择

```shell
case 变量名 in
	值1）
		指令1
			;;
	值2）
		指令2
			;;
	值3）
		指令3
			;;
esac
```

### 循环

- 循环

for循环

```shell
for 值 in 列表
do
	执行语句
done

# 例子
#！/bin/bash
for i in {1..9}
do
	echo "202${i}"
done

#例子2
cmd=$1
num=$2
for((i=0; i<num; i++))
do
  echo "${cmd}:${i}"
done
```

while循环

```shell
while 条件
do
	执行语句
done

# 示例
#!/bin/bash
declare -i i=1
declare -i sum=0
while ((i<=10))
do
  let sum+=i
  let ++i
done
echo $sum
```

until循环

```shell
until 条件
do
	执行语句
done

# 示例
#!/bin/bash
declare -i i=10
declare -i sum=0
until ((i>10))
do
  let sum+=i
  let ++i
done
echo $sum
```

- 打断

```shell
break
# 自动终止所在的最内层循环
break n
# 跳出外部循环，n制定了要跳出的循环逻辑

continue
# 提前中止某次循环中的命令
```

## 处理输入

### 多命令

```shell
# ; 各命令结果互不影响
# && 前面成功才执行后面
# || 前面失败才执行后面
date; who
date&&who
date||who 
```



### 命令行参数

读取参数

```shell
# bash shell会将一些成为位置参数的特殊变量分配给输入到命令行的所有参数。

$0
# 位置参数是标准的数字：$0是脚本程序名，$1是第一个参数，$2是第二个参数，依次类推直到第9个参数$9
# 对于第9个以后的命令行参数，在获取时使用${10}
$#
# 参数的个数
${!#}
# 最后一个参数的获取
$*
# 将命令行上提供的所有参数当作一个单词保存
$@
# 将命令行上提供的所有参数当作同一个字符串中的多个独立的单词，可通过for命令遍历参数
```

参数测试

```shell
# 方法一：测试指定的参数
if [ -n "$1"]
then
	...
else
	...
fi

# 方法二：测试参数的总数
if [ $# -ne 2 ]
then
	...
else
	...
fi
```

移动变量

```shell
# shift命令默认情况下会将每个参数变量向左移动一个位置，所以变量$2会移动到$1，而$1的值会被删除，$0就是程序名不会变
echo
count=1
while [-n "$1"]
do
  echo "params #$count = $1"
  count=$[ $count + 1 ]
  shift
done
```

### 处理选项

查找选项

```shell
# 处理简单选项，使用shift和case
# 命令行：./demo.sh -a -b -c -d
echo
while [ -n "$1"]
do
  case "$1" in
    -a) echo "Found the -a option";;
    ...
    *) echo "$1 is not an option";;
  esac
  shift
done
 
# 分离参数和选项，命令行中用--做分隔
# 命令行：./demo.sh -a -b -c -- test1 test2 test3
echo
while [ -n "$1"]
do
  case "$1" in
    -a) echo "Found the -a option";;
    ...
    -b) echo "Found the -b option";;
    --) shift
    	break;;
    *) echo "$1 is not an option";;
  esac
  shift
done
#
count=1
for param in $@
do
  echo "Parameter #$count: $param"
  count=$[ $count + 1 ]
done

# 处理带值的选项
# 命令行：./demo.sh -a -b test1 -c
echo
while [ -n "$1"]
do
  case "$1" in
    -a) echo "Found the -a option";;
    ...
    -b) param="$2"
        echo "Found the -b option, with parameter value $param"
        shift ;;
    --) shift
    	break;;
    *) echo "$1 is not an option";;
  esac
  shift
done
#
count=1
for param in $@
do
  echo "Parameter #$count: $param"
  count=$[ $count + 1 ]
done
```

使用`getopt/getopts`命令

```shell
getopt
# 命令是一个在处理命令行选项和参数时方便的工具。

# 命令格式：
getopt optstring parameters
# optstring中列出要在脚本中用到的每个命令行选项字母，然后在每个需要参数值的额选项字母后加一个冒号

# 命令行中使用
getopt ab:cd -a -b test1 -cd test2 test3

# 在脚本中使用
# set --将命令行参数替换为set命令的命令行值
set -- $(getopt -q ab:cd "$@")
echo
while [ -n "$1" ]
...

getopts
# getopt并不擅长处理带有空格和引号的参数，它会将空格当作参数分隔符，而不是根据引号将二者当作一个参数，geopts可以解决

# 命令格式
getopts optstring variable
# 有效的选项字母会列在optstring中，如果选项字母要求有参数值，加一个冒号。
# 要去掉错误消息对的话，可以在optstring之前加一个冒号。
# getopts将当前参数保存在命令行中定义的variable中
# getopts使用了两个环境变量。如果选项需要跟一个参数值，OPTARG环境变量就会保存这个值。OPTIND环境变量保存了参数列表中getopts正在处理的参数位置。这样就能在处理完选项之后继续处理其他命令行参数

# 脚本中使用
# 命令行：./demo.sh -ab test1 -c
echo
while getopts :ab:c opt
do
  case "$opt" in
    a) echo "found the -a option" ;;
    b) echo "found the -b option, with value $OPTARG";;
    ...
    *) echo "unknown option: $opt";;
  esac
done
```

### 交互输入

基本读取

```shell
read
# 从标准输入或另一个文件描述符中接受输入，在收到输入后，read命令会将数据放进一个变量
echo -n "Enter your name: "
read name
echo "hellp $name, welcome to my program"
```

超时继续

```shell
# -t选项可以避免一直等待用户输入，指定一个计时器表示read命令等待输入的秒数，当计时器过期后，read命令会返回一个非0退出状态码
if read -t 5 -p "Please enter your name: " name
then
  echo "hello $name, welcome to my script."
else
  echo "sorry, too slow"
fi
```

隐藏方式读取

```shell
# -s选项可以避免在read命令中输入的数据出现在显示器上（实际上read命令将文本颜色设置为背景色）
read -s -p "Enter your password: " pass
echo
echo "Is your password readlly $pass?"
```

从文件中读取

```shell
# 对文件使用cat命令，将结果通过｜直接传给含有read的while命令
count=1
cat test | while read line  # while循环会持续通过read命令处理文件中的行，直到read命令以非0退出状态码退出
do
  echo "Line $count: $line"
  count=$[ $count + 1]
done
echo "Finished processing the file"
```

## 呈现数据

### 输入输出

- 输出命令

```shell
echo
# 显示消息
echo This is a test

# 显示内容中有引号
echo "Let's go!"

# 需要date和标题在一行，无-n时自动换行
echo -n "The time and date are: "
date
```

- 标准文件描述符

Linux系统将每个对象当作文件处理。这包括输入和输出进程。Linux用文件描述符来标识每个文件对象。文件描述符是一个非负整数，可以唯一标识会话中打开的文件。每个进程一次最多可以有9个文件描述符。处于特殊目的，bash shell保留了前三个文件描述符

| 文件描述符 | 缩写   | 描述     |
| ---------- | ------ | -------- |
| 0          | STDIN  | 标准输入 |
| 1          | STFOUT | 标准输出 |
| 2          | STDERR | 标准错误 |

`STDIN`

```
STDIN文件描述符代表shell的标准输入。对于终端界面来说，标准输入是键盘。shell从STDIN文件描述符对应的键盘获得输入，在用户输入时处理每个字符。
在使用输入重定向符号(<)时，Linux会用重定向指定的文件来替换标准输入文件描述符。它会读取文件并提取数据，就如同它是键盘上输入的。
```

`STDOUT`

```
STDOUT文件描述符代表shell的标准输出。在终端界面上，标准输出是终端显示器。shell的所有输出（包括shell中圆形的额程序和脚本）会被定向到标准输出中，也就是显示器。
通过输出重定向符号(>/>>)，会显示到显示器的额所有输出会被shell重定向到指定的重定向文件。
```

`STDERR`

```
shell通过特殊的STDERR文件描述符来处理错误消息。STDERR文件描述符代表shell的标准错误输出，shell或shell中圆形的程序和脚本出错时生成的错误消息都会发送到这个位置。
默认情况下，STDERR文件描述符和STDOUT文件描述符只想同样的大地方（尽管分配给他们的文件描述符值不同）。也即是，默认情况下，错误消息也会输出到显示器中输出。
```

示例

```shell
$ ls -al badfile > test
ls: cannot access badfile:No such file or directory
$ cat test
$

# 命令生成错误消息时，shell并未将错误消息重定向到输出重定向文件。shell创建了输出重定向文件，但错误信息却显示在了显示器屏幕上。
```

- 重定向错误

只重定向错误

```shell
# STDERR文件描述符被设为2，将文件描述符值放在重定向符号前可以实现只重定向错误
ls -al badfile 2> test
```

重定向错误和数据

```shell
# 使用2个重定向符号，可将错误和数据重定向到不同的输出文件
# 需要在符号前面放上待重定向数据所对应的文件描述符，然后指向用于保存数据的输出文件
ls -al badfile 2> test2 1> test2

# 可将错误和数据重定向到同一个输出文件
ls -al badfile &> test3
```

### 脚本中重定向

- 重定向输出

临时重定向

```shell
# 若有意在脚本中错误信息，可将单独的一行输出重定向到STDERR
echo "This is an error message" >&2

# 示例
# test.sh
echo "This is an error" >&2
echo "This is normal oouput"
# 终端执行
./test.sh
./test.sh 2>test2
cat test2
# STDOUT显示的文本显示在屏幕，发送给STDERR的重定向到输出到文件
```

永久重定向脚本中的所有命令

```shell
# 用exec命令告诉shell在脚本执行期间重定向某个特定文件描述符
exec 1>testout
exec 2>testerror
echo ...
# exec命令会启动一个新的sheell并将STDOUT/STDERR文件描述符重定向到文件
```

- 重定向输入

```shell
# exec命令允许将STDIN重定向到Linux系统上的文件
exec 0< testfile
count=1
while read line  # 当read试图从STDIN读入数据时，由于重定向，会从文件中读取而不是键盘
do
  echo "Line #$count: $line"
  count=$[ $count + 1 ]
done
```

### 自定义重定向

在脚本中重定向输入和输出时，并不局限于这3个默认的文件描述符，其他6个均可作为输入或输出重定向。

- 创建输出文件描述符

```shell
# exec可以给输出分配文件描述符。和标准的文件描述符一样，一旦将另一个文件描述符分配给一个文件，这个重定向就会一直有效，直到重新分配
exec 3>testout
echo "test"
echo "this should be stored in the file" >&3
```

- 重定向文件描述符

```shell
# 临时重定向输出，然后恢复默认输出位置
exec 3>&1  # 将文件描述符3重定向到1的STDOUT，发送给3的输出都将出现在显示器上
exec 1>testout  # 将STDOUT重定向到文件，但是3仍然指向STDOUT原来的位置即显示器
echo "hello, to file"
echo "this line end to file"
exec 1>&3  # 将STDOUT重定向到3（显示器），则STDOUT已经被重指向了原来的位置：显示器
echo "back to normal"
```

- 创建输入文件描述符

```shell
# 可以和重定向输出文件描述符同样的方法重定向输入文件描述符。
exec 6<&0
exec 0< testfile

count=1
while read line
do
  echo "Line #$count: $line"
  count=$[ $count + 1 ]
done
exec 0<&6
read -p "Are you done now? " answer
case $answer in
 Y|y) echo "bye";;
 N|n) echo "Sorry, this is the end.";;
esac
```

- 创建读写文件描述符

```shell
# 可以用同一个文件描述符对同一个文件进行读写
# 由于是对同一个文件进行数据读写，shell会维护一个内部指针，指明在文件中的当前位置。任何读或写都会从文件指针上次的位置开始。
# test.sh
exec 3<> testfile
read line <&3
echo "Read: $line"
echo "This is a test line" >&3
# testfile.sh
This is the first line.
This is the second line.
This is the third line.
```

- 关闭文件描述符

```shell
# 若是创建了新的输入或输出文件描述符，shell会在脚本退出时自动关闭他们。
# 若需要在脚本结束前手动关闭文件描述符，可以将它重定向得到特殊符号&-
exec 3<&-
```

### 列出打开的文件描述符

```shell
/usr/sbin/lsof
# 普通用户来运行，需要通过全路径名来引用

# 示例
/usr/sbin/lsof -a -p $$ -d 0,1,2
# -p允许指定进程PID，-d允许指定要显示的文件描述符编号，$$可以知道当前PID，-a用来对其他两个选项的结果执行布尔AND运算
```

`lsof`的默认输出

```shell
COMMAND		# 正在运行的命令名的前9个字符
PID			# 进程的PID
USER		# 进程属主的登录名
FD			# 文件描述符及访问类型
TYPE		# 文件的类型（CHR字符型，BLK块型，REG常规文件）
DEVICE		# 设备的设备号
SIZE		# 文件大小
NODE		# 本地文件的节点号
NAME		# 文件名
```

### 阻止命令输出

```shell
# 不想显示脚本的输出，尤其是将脚本作为后台进程运行时很常见
ls -al > /dev/null
```

### 创建临时文件

Linux系统有特殊的目录，专供临时文件使用。Linux使用`/tmp`目录来存放不需要永久保留的文件。大多数Linux发行版配置了系统在启动时自动删除`/tmp`目录的所有文件。

系统上任何用户账户都有权限在读写`/tmp`目录中的文件。这个特性提供了一种创建临时文件的简单方法，不用操心清理工作。

`mktemp`可以在`/tmp`目录中创建一个唯一的临时文件。shell会创建这个文件，但不用默认的umask值。它会将文件的读写权限分配给文件的属主，并将你设成文件的属主。一旦创建了文件，就在脚本中有了完整的读写权限，但其他人无法访问它（root除外）。

- 创建本地临时文件

默认情况下，`mktemp`会在本地目录中创建一个文件。要用`mktemp`命令在本地目录中创建一个临时文件，只要指定一个文件名模版即可。模版可以包含任意文本文件名，在文件名末尾加上6个X就可以了

```shell
mktemp testing.XXXXXX
# mktemp会用6个字符码替换X，从而保证文件名在目录中是唯一的。可以创建多个临时文件，它可以保证每个文件都是唯一的。
# 在脚本中使用
tempfile=$(mkteemp test.XXXXXX)
exec 3>$tempfile
echo "This is the first line" >&3
exec 3>&-
echo "Done creating temp file. The contents are:"
cat $tempfile
rm -f $tmpfile 2> /dev/null
```

- 在`/tmp`目录创建临时文件

```shell
mktemp -t test.XXXXXX
# -t选项会强制mktemp命令来在系统的临时目录来创建该文件。在用这个特性时，mktemp命令会返回用来创建临时文件的全路径，而不是只有文件名
# 在脚本中使用
tempfile=$(mkteemp -t test.XXXXXX)
echo "This is the first line" > $tempfile
echo "This is the second line" >> $tempfile
echo "The temp file is located at: $tempfile"
cat $tempfile
rm -f $tempfile
```

- 创建临时目录

```shell
mktemp -d test.XXXXXX
# -d选项告诉mktemp来创建一个临时目录而不是临时文件
tempdir=$(mktemp -d dir.XXXXXX)
cd $tempdir
tempfile1=$(mktemp temp.XXXXXX)
tempfile2=$(mktemp temp.XXXXXX)
exec 7> $tempfile1
exec 8> $tempfile2
echo "sending data to dir $tempdir"
echo "This is a test line of data for $tempfile1" >&7
echo "This is a test line of data for $tempfile2" >&8
```

### 记录消息

将输出同时发送到显示器和日志文件，不用将输出重定向两次，只要用特殊的`tee`命令就行.

`tee`命令相当于管道的一个T型接头，将从STDDIN过来的数据同时发往两处，一处是STDOUT，另一个是tee命令行所指定的文件名。

```shell
tee filename
# 由于tee会重定向来自STDIN的数据，可以配合管道命令来重定向命令输出
date | tee testfile  # 输出同时出现在了STDOUT（显示器）和指定的文件中
# 默认情况下，tee命令会在每次使用时覆盖输出文件内容，若想将数据追加到文件中，用-a选项
date ｜ tee -a testfile
# 脚本中
tempfile=test22file
echo "This is the start of the test" | tee $tempfile
echo "This is the seccond line of the test" | tee -a $tempfile
echo "This is the end of the test" | tee -a $tempfile
```

## 控制脚本

### 处理信号

Linux信号

| 信号 | 值      | 描述                           |
| ---- | ------- | ------------------------------ |
| 1    | SIGHUP  | 挂起进程                       |
| 2    | SIGINT  | 终止进程                       |
| 3    | SIGQUIT | 停止进程                       |
| 9    | SIGKILL | 无条件终止进程                 |
| 15   | SIGTERM | 尽可能终止进程                 |
| 17   | SIGSTOP | 无条件停止进程，但不是终止进程 |
| 18   | SIGTSTP | 停止或暂停进程，但不终止进程   |
| 19   | SIGCONT | 继续运行停止的进程             |

默认情况下，bash shell会忽略收到的任何STGQUIT(3)和SIGTERM(15)信号，但是会处理收到的SIGUP(1)和SIGINT(2)。

如果收到SIGHUP信号，比如要历来一个交互式shell，它就会退出。但是在退出之前，它会将SIGHUP信号传给所有由该shell所启动的进程（包括正在运行的shell脚本）

通过SIGINT信号，可以中断shell。Linux内核会停止为shell分配CPU处理时间。这种情况发生时，shell会将SIGNINT信号传给所有由它所启动的进程，一次告知出现的状况。

- 生成信号

bash shell允许用键盘上的组合键生成两种基本的Linux信号。

中断进程

```
Ctrl+C生成SIGINT信号，并将其发送给当前在shell中运行的所有进程，来停止当前运行的进程。
```

暂停进程

```
Ctrl+Z生成SIGTSTP信号，停止shell总运行的任何进程。停止进程会让程序继续保留在内存中，并能从上次停止的位置继续运行。
暂停可能会比较危险，比如：脚本打开了一个关键的系统文件的文件锁。
```

- 捕获信号

可以不忽略信号，在信号出现时捕获它们并执行其他命令。`trap`命令允许指定shell脚本要监看并从shell中拦截的Linux信号。如果脚本收到了`trap`命令中列出的信号，该信号不再由shell处理。而是交由本地处理。

```shell
trap commands signals
# 示例
trap "echo 'sorry, I have trapped Ctrl-C'" SIGINT
echo This is a test script
count=1
while [ $count -le 10]
do
  echo "Loop #$count"
  sleep 1
  count=$[ $count + 1 ]
done
echo "This is the end of the test script"
```

- 捕获脚本退出

 除了可以在shell脚本中捕获信号，也可以在shell脚本退出时进行捕获。

```shell
trap commandds exit
# 示例
trap "echo goodbye" exit  # 当脚本运行到正常退出位置时/提前退出脚本，会触发信息
echo This is a test script
count=1
while [ $count -le 5]
do
  echo "Loop #$count"
  sleep 1
  count=$[ $count + 1 ]
done
```

- 修改或移除捕获

若要在脚本不同位置进行不同的捕获处理，需重新使用带有新选项的`trap`命令

修改

```shell
trap "echo 'sorry, I have trapped Ctrl-C'" SIGINT
echo This is a test script
count=1
while [ $count -le 5]
do
  echo "Loop #$count"
  sleep 1
  count=$[ $count + 1 ]
done
trap "echo 'I modified the trap!'" SIGINT
count=1
while [ $count -le 5]
do
  echo "Second Loop #$count"
  sleep 1
  count=$[ $count + 1 ]
done
```

删除

```shell
trap "echo 'sorry, I have trapped Ctrl-C'" SIGINT
echo This is a test script
count=1
while [ $count -le 5]
do
  echo "Loop #$count"
  sleep 1
  count=$[ $count + 1 ]
done
trap -- SIGINT
echo "I just removed the trap"
count=1
while [ $count -le 5]
do
  echo "Second Loop #$count"
  sleep 1
  count=$[ $count + 1 ]
done
```

### 后台运行脚本

- 控制台运行脚本

在后台模式中，进程运行时不会和终端上点会话上的STDIN,STDOUT,STDERR关联。但是当后台进程运行时，仍然会使用终端显示器来显示STDOUT和STDERR消息。

在终端会话中使用后台进程时需注意：每一个后台进程都和终端会话终端关联，若终端会话退出，那么后台进程也会随之退出。

以后台模式运行shell脚本，在命令后加上`&`，将命令和bash shell分离开来，将命令作为系统中的一个独立的后台进程运行。

```shell
>>>./demo.sh &
[1] 3221
# 方括号中的数字是shell分配给后台进程的作业号。下一个数是Liunx系统分配给进程的进程ID(PID)。
# 当显示这些内容后，新的命令行界面提示符就出现了，可以回到shell，继续输入新的命令
# 当后台进程结束时，会在终端显示
[1] Done. ./demo.sh
```

- 在非控制台下运行脚本

使用`nohup`可以即使退出了终端会话，脚本也可以一直以后台模式运行到结束。当关闭会话时，脚本会忽略终端会话发过来的SIGHUP信号。

```shell
nohup ./test.sh &

nohup ./test.sh  > output.log 2>&1 &
```

由于`nohup`命令会解除终端与进程的关联，进程也就不再同STDOUT和STDERR联系在一起。为了保存该命令产生的输出，会自动将STOUT和STDERR的消息重定向到一个名为`nohup.out`的文件中。

注意：当运行位于同一个目录中的多个命令时要消息，因为所有的输出都会被发送到同一个`nohup.out`文件中。

### 作业控制

```shell
jobs
# 查看作业
-l 	# 列出进程到的PID以及作业号
-n	# 只列出上次shell发出的通知后改变了状态的作业
-p	# 只列出作业的PID
-r	# 只列出运行中的作业
-s	# 只列出已停止的作业

bg [作业号]
# 重启停止的作业
```

### 调整谦让度

在多任务操作系统中，内核负责将CPU时间分配给系统上运行的每个进程。调度优先级是内核分配给进程的CPU时间（相对于其他进程）。在Liunx系统中，由shell启动的所有进程的调度优先级默认都是相同的。

调度优先级是整个数值，从-20（最高优先级）到+19（最低优先级）。默认情况下，bash shell以优先级0来启动所有进程。

```shell
nice
# nice命令会让脚本以指定的优先级运行
nice -n 优先级数 启动命令
nice -优先级数 启动命令

renice
# renice可以改变系统上已经运行的命令的优先级
# 只能对属于你的进程进行，只能通过renice降低进程的优先级，root用户可以通过renice来任意调整进程的优先级
renice -n 优先级数 -p PID
```

### 定时运行作业

- at命令

`at`命令允许指定Linux系统何时运行脚本。`at`命令会将作业提交到队列中，指定shell何时运行该作业。`at`的守护进程`atd`会以后台模式运行，检查作业队列来运行作业。大多Linux发行版在启动时运行此守护进程。

`atd`守护进程会检查系统上的一个特殊目录(通常`/var/spool/at`)来获取用`at`命令提交的作业。默认情况下，`atd`守护进程会每60s检查一下这个目录。有作业时，`atd`守护进程会检查作业设置运行的额时间。如果时间跟当前时间匹配，`atd`守护进程就会运行此作业。

> at格式

```shell
at [-f filename] time
# 默认情况下，at命令会讲STDIN的输入放到队列中，可以用-f指定用于读取命令（脚本文件）的文件名
# time参数指定了何时运行该作业，若指定的时间已经错过，at命令会在第二天的那个时间运行指定的作业
# time时间格式
# 标准的小时+分钟：10:15
# AM/PM指示符：10:15 PM
# 特定可命名时间：now,noon,midnight,teatime
# 标准日期格式：MMDDYY，MM/DD/YY，DD.MM.YY
# 文本日期：2012 Jul 4, Dec 25
# 时间增量：now+25min,tomorow10:15PM,10:15+7days
```

该作业被提交到作业队列，针对优先级不同，存在26种不同的作业队列，通常用字母a~z和A～Z。字母排序越高，作业运行的优先级就越低。默认会被提交到a作业队列，可以使用`-q`参数指定不同的队列字母

> 获取作业的输出

当作业在linux系统上运行时，显示器并不会关联到该作业。Linux会将提交作业的用户的电子邮件地址作为STDOUT和STDERR。为了避免使用email，需要对STDOUT和STDERR进行重定向。

```shell
# 脚本test.sh
echo "This script ran at $(date + %B%d,%T)" > test.out
echo >> test.out
sleep 5
echo "This is the script end..." >> test.out
# 执行脚本
at -M -f test.sh now
# 查看输出
cat test.out
```

> 列出等待的作业

```shell
apq
```

> 删除作业

```shell
atrm [作业号]
```

- cron

[web设置](https://crontab.guru)

`cron`可以安排要定期执行的作业。`cron`程序会在后台运行并检查一个特殊的表（cron时间表），以获知已安排执行的作业。

> cron时间表

```shell
min hour dayofmonth month dayofweek command
# cron时间表允许用特定值、取值范围、通配符*来指定条目
minute: 区间为 0 – 59 
hour: 区间为0 – 23 
day-of-month: 区间为0 – 31 
month: 区间为1 – 12. 1 是1月. 12是12月. 
Day-of-week: 区间为0 – 7. 周日可以是0或7.
```

> 构建cron时间表

```shell
crontab -l  # 列出已有的cron时间表
crontab -e	# 为时间表添加条目
crontab -r 	# 删除 crontab 文件
```

示例

```shell
1 0 * * * /root/bin/backup.sh  # 在 12:01 a.m 运行，即每天凌晨过一分钟
10 13 1 * * /root/bin/backup.sh	# 每个月的第一天 1:10 p.m 运行
59 11 * * 1,2,3,4,5 /root/bin/backup.sh  # 每个工作日11:59 p.m运行
0 23 * * 1-5 /root/bin/incremental-backup.sh #每个工作日 11 p.m 运行。
*/5 * * * * /root/bin/check-status.sh  # 每5分钟运行一次
0 */1 * * * /root/bin/check-status.sh  # 每1小时运行一次
0 0 * * * /root/bin/check-status.sh  # 每天运行一次
0 0 * * 0 /root/bin/check-status.sh  # 每周运行一次
0 0 1 * * /root/bin/check-status.sh  # 每月运行一次
0 0 1 1 * /root/bin/check-status.sh  # 每年运行一次
```

> 浏览cron目录

若创建的脚本对精确到的执行时间要求不高，用预置的cron脚本目录会更方便。有四个基本目录：`hourly,daily,monthly,weekly`

```shell
ls /etc/cron.*ly
```

> anacron程序

cron程序的问题是假设Linux系统是7*24小时运行。若非，则Linux系统不运行时会错过程序。故许多linux系统包含了anacron程序。

anacron若知道某个作业错过了执行时间，会尽快运行该作业。此功能常用于日志维护的脚本。anacron可以保证系统每次启动时整理日志文件。

anacron程序只会处理位于cron目录的程序（不会运行`/etc/cron.hourly`的脚本，因为不会处理执行时间需求小于1天的脚本）。它用时间戳来决定作业是否在正确的计划间隔内运行了。每个cron目录都有个时间戳文件，该文件位于`/var/spool/anacron`

anacron时间表`/etc/anacrontab`

```shell
period delay identifier command
# anacron时间表的基本格式
# period定义了作业多久运行一次，以天为单位。anacron程序用此条目来检查作业的时间戳文件
# delay会指定系统启动后anacron程序需要等待等多少分钟再开始运行错过的脚本。
# command条目包含了run-parts程序和一个cron脚本目录名。run-parts程序负责运行目录中传给它的任何脚本
# identifier是一种特别的非空字符串，入cron-weekly。用于唯一标识日志消息和错误邮件中作业
```

