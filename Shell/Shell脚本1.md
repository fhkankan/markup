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
# 查看退出状态码
date
echo $?		# 展示上个命令的退出状态码

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

### 环境变量

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

### 用户变量

```shell
# 临时存储数据并在整个脚本中使用，可以是任何由字母数字或下划线组成的文本字符串，长度不超过20个
# 整体,中间无特殊字符
变量名=变量值
# 原样输出
变量名='变量值'
# 先解析，将结果与其他合成整体
变量名="变量值"
```

### 内置变量

```shell
$0	# 获取当前执行的shell脚本文件名，包括脚本路径
$n	# 获取当前执行的shell脚本的第n个参数值，0表示文件名，大于9就$(10)
$#	# 获取当前shell命令中参数的总个数
$?	# 获取执行上一个指令的返回值(0表示成功，非0为失败)
$$  # 当前shell的PID
```

### 变量操作

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

```
for 值 in 列表
do
	执行语句
done
```

while循环

```
while 条件
do
	执行语句
done
```

until循环

```
until 条件
do
	执行语句
done
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

使用多个命令输入，以分号隔开

```shell
date; who 
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

## 数据展示

### 数据展示

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

