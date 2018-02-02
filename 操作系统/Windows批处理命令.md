#定义

```
批处理（Batch）通常被认为是一种简化的脚本语言，它应用于DOS和Windows系统中，它是由DOS或者Windows系统内嵌的解释器（通常是COMMAND.COM或者CMD.EXE）解释运行。类似于Unix中的Shell脚本。批处理文件具有.bat或者.cmd的扩展名。在Windows下运用批出处理的主要目的是完成自动化操作（如无人值守安装程序、批量处理txt数据等或许这些操作需要调用外部程序）

Shell 在计算机科学中，是指“提供用户使用界面”的软件，通常指的是命令行界面的解析器。一般来说，这个词是指操作系统中，提供访问内核所提供之服务的程序。Shell也用于泛指所有为用户提供操作界面的程序，也就是程序和用户交互的层面。因此与之相对的是程序内核（英语：Core），内核不提供和用户的交互功能。

通常将Shell分为两类：命令行与图形界面。命令行壳层提供一个命令行界面（CLI）；而图形壳层提供一个图形用户界面（GUI）
```

# DOS下命令

## 屏幕显示控制

```
echo 		回显控制（在命令提示符或DOS下，每执行一条命令都会显示在屏幕上，这就叫回显）
echo on :: 打开回显
echo off :: 关闭回显
@echo off :: 关闭回显，且连这句都不显示，常用
echo hello world :: echo 还有打印的功能

cls 清屏命令

@ 符号		如某条命令前加@关闭该命令的回显
@cls
@echo hello World
```

## 文件控制

```
edit		文本文件的创建和编辑

del 		文件的删除
del [/P] [/F] [/S] [/Q] [/A[[:]attributes]] names

copy/move 	文件的复制和移动
copy/move 原文件路径/原文件名 新路径

ren 			文件重命名
REN [drive:][path]filename1 filename2.

attrib 			设置或更改文件属性
attrib/?		查询参数帮助
+ATTRIB [+R | -R] [+A | -A ] [+S | -S] [+H | -H] [+I | -I]     [drive:][path][filename] [/S [/D] [/L]]
  + 设置属性。
  - 清除属性。
  R 只读文件属性。
  A 存档文件属性。
  S 系统文件属性。
  H 隐藏文件属性。
  I 无内容索引文件属性。
  [drive:][path][filename]      指定 attrib 要处理的文件。
  /S 处理当前文件夹及其所有子文件夹中的匹配文件。
  /D 也处理文件夹。
  /L 处理符号链接和符号链接目标的属性。
```

## 文件夹控制

```
cd(chdir)	 	显示或改变当前目录名称
格式：cd [drive:]path
常用格式列举：
cd 
cd..     返回上级目录 
cd\     返回根目录 
cd fullPath

md(mkdir) 		创建文件夹
格式 md [drive:]path
可以嵌套创建文件夹，如 md a\b\c\d

rd(rmdir) 删除一个文件夹
格式：rd [/s] [/q] [drive:]path
/s 的意思删除指定目录下的所有文件以及子目录
/q 安静模式，不向用户询问是否删除
比如删除上面建立的文件夹
```

## 文件文件夹共同

```
dir 		显式目录中文件和子目录列表，详细参数dir/?

tree		显式目录结构
```

## 特殊符号

```
重定向符号  > 与 >>
重定向符号就是传递和覆盖的意思，它所起的作用是将运行的结果传递到后面的范围（后边可以是文件,也可以是默认的系统控制台，即命令提示符）。
例: tree /f > z:\result.txt :: 把当前目录的树形目录结构打印到 result.txt 文件中
> 是完全覆盖以前文件内容
>> 是在以前文件内容后面接着写

命令管道符  |
表示把在它之前的命令或语句的执行结果作为在它之后的命令或语句的处理对象，即，就是把它之前的输出作为它之后的输入。
:: 查找qq.exe进程 如找到就结束
tasklist | find /i "qq.exe" && taskkill /f /im qq.exe

组合命令  & 与 && 及 ||
& 顺序执行多条命令，前面命令执行失败了，不影响后边的命令执行
&& 顺序执行多条命令，当碰到执行错误的命令则停止执行，如无错则一直执行下去
|| 顺序执行多条命令，当碰到执行错误的命令才往后执行，如遇到执行正确的命令则停止

转义字符 ^
如 echo ^>

变量引用符 %
定义变量(后面批处理编程结构会提到)var 后，用两个%%包围变量的方式引用此变量 %var%

界定符 ””
当路径中有空格，需用英文状态下的双引号””包围路径

/? 		命令帮助

pause 	暂停批处理程序

type  	显式文本文件的内容

ver   	显式操作系统版本

rem		注释符号

:: 		注释符号

xcopy 	复制文件和目录

* 		通配符 表示任意多个字符

? 		通配符 表示一个字符

find 	搜索字符串
exit 	退出命令
```

# windows下命令

## 网络相关

```
ping		连接网络
常用参数 
-t 一直 ping 
-l 指定包的字节数，最大 65500 字节 
ping  [-l 65500 最大字节数] IP [-t] 

ipconfig 	显示当前电脑TCP/IP 配置 
常用参数： 
/all 显示所有信息 
/flushdns 刷新DNS信息 
/renew 更新 DHCP 配置

arp   		显式或修改ARP
arp –a :: 显式 ARP表
更多命令参见 arp /?

netstat 	TCP统计 
常用 
netstat -nao

racert 		路由追踪命令
例： tracert www.baidu.com
更多命令参见 tracert /?

pathping 	路程信息查看
pathping www.baidu.com
更多参数详见 pathping /?

telnet （推荐用 SecureCRT）
```

## 磁盘相关

```
chkdsk 		磁盘检测和修复（弱弱的修复）
详见 chkdsk /? 

subst 		将驱动器与路径关联(在第一家公司写代码需要指定统一路径时用过) 
创建 subst 虚拟磁盘: 物理磁盘路径 
删除 subst 虚拟磁盘: /d
```

## 系统相关

```
driverquery 	显示已安装驱动

systeminfo 		获取系统配置信息

regsvr32 		注册或卸载某个动态链接库 
regsvr32 /u或/i *.dll(*.ocx)
更多参数 详见 regsvr32 /?

tasklist 		显式进程列表
用法相见 tasklist /?

taskkill 		结束指定进程
taskkill /f /im 映像名
例如 关闭所有 IE 进程
taskkill /f /im iexplore.exe
结合 tasklist 和 taskkill 使用
tasklist | find /i "qq.exe" && taskkill /f /im qq.exe

shutdown 		关闭计算机
shutdown/s    关闭计算机 
shutdown/r    重启计算机 
shutdown/a    放弃关闭计算机 
更多参数详见shutdown/?

sc 				命令系列
sc create 创建服务
sc delete 删除服务
sc start 启动服务
sc stop 停止服务
更多参数详见 sc/?
```

# 批处理编程结构

## 变量

```
set			查看所有已知变量(环境变量、内置系统变量)

变量都是弱类型的,区分空格不区分大小写 
批处理变量命名：
set varA =311 

如果需要用局部变量则用 
setlocal 
set 语句 
endlocal

引用变量用 两个 %% 包围 如 %varA% 

数学运算 
+ 加、-减、*乘、/除、%求模 
set /a 数学表达式 
如 
::total 自加1 
set /a total+=1 
```

## 条件语句--if

```
格式： 
if cond ( 
  statement_1 
  ... 
  statement_n 
) 
[else( 
    statement_1 
  .. 
  statement_n 
)]

方括号“[]”的含义为可选 

例子 
if "%1"=="1" (echo is one) else ( echo is not one)

比较运算符 
==  判断相等 
equ 判断相等 
lss 小于 
leq 小于或等于 
gtr 大于 
geq 大于或等于
```

## 循环语句--for

```
基本格式: for iterator do (statements)

遍历一系列的值 
格式 ： 
for /L %%var in (start, step, end) do (statements) rem var 是单字母变量 如 %%i，如果是多字母变量如 %%aa 会报错 
eg:
@echo off
for /l %%B in (0,1,15) do echo %%B

对文件的遍历 
格式： 
for %%var in (fileSets) do (statements)  rem fileSets 文件的集合 
eg： 
::rem 打印 C盘下的txt文件
@echo off
for %%i in (C:\*.txt) do echo %%i
::rem 打印 C盘下的txt和 sys 文件
@echo off
for %%i in (C:\*.txt C:\*.sys) do echo %%i

对文件夹的遍历 
格式： 
for /d %%var in (directorySet) do (statements)  rem directorySet 目录的集合 
eg：
@echo off
for /d %%i in (Z:\) do echo %%i
:: 对 Z:\ 下目录的遍历 
@echo off
for /d %%i in (Z:\*) do echo %%i
::多个目录的例子
@echo off
for /d %%i in (%SystemRoot%\* Z:\*) do echo %%i

递归对文件遍历 
格式： 
for /r [path] %%var in (fileSet) do (statements) 
eg:
@echo off
for /r C:\ %%i in (*.txt) do echo %%i

/r 与 /d 结合
eg:
::输出 %SystemRoot% 下的所有目录及子目录
@echo off
for /r %SystemRoot% /d %%i in (*) do echo %%i
```

## 函数

```
1、不带参数的函数
@echo off
echo  调用前
pause

call :sub
::调用子过程

echo 调用后
pause
exit

:sub
echo 子过程调用中
pause
goto :eof

2、带参数对的函数
@echo off
set a=5
set b=4

call :add %a% %b%
::调用子过程

echo 两个数的和为%sum%
pause
exit

:add
set /a sum=%1+%2     
goto :eof
```

## 一键打开多个

```
# 文件夹
在.bat批处理文件中写入
explore.exe /n,path1
explore.exe /n,path2

# 文件
start path\test1.txt
start path\test2.txt

# 不同应用程序
start "" "path\name1.exe"
start "" "path\name2.exe"
或
cd /d path
start name1.exe
cd /d path
start name2.exe

# 一个程序多个窗口
C:/WindowsApplication2.exe是一个窗口应用程序
1.打开一个命令窗口，一次显示5个WindowsApplication2.exe：
for /L %%i in (1,1,5) do @start C:/WindowsApplication2.exe

2.打开一个命令窗口，一次显示一个WindowsApplication2.exe，一个关闭后显示下一WindowsApplication2.exe：
for /L %%i in (1,1,5) do C:/WindowsApplication2.exe

3.一次打开5个命令窗口和5个WindowsApplication2.exe
for /L %%i in (1,1,5) do @start call C:/WindowsApplication2.exe 

for /L %%i in (1,1,5) do call C:/WindowsApplication2.exe 效果等效于2

以背景线程方式打开并将输出写到文件中:
rem run program five times concurrently, and write info to log*.log files.
for /L %%i in (1,1,5) do @start /B WindowsApplication2.exe >log%%i.log

 
```

