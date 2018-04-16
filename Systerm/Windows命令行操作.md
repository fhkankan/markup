# 快捷方式

```
win+r+cmd	打开终端
↑			显示上次操作命令
↓ 			显示下一个已操作的命令
tab键 	   提示当前目录下的文件列表，自动补全功能 
cmd中按F7	  可以调出（之前所输入的）命令的历史列表
单击文件夹，shift+鼠标右键，选择'在此处打开命令窗口'
ctr+C		中断正在执行的命令
```

# 查看

```
ver				显示系统版本
dir				列举目录
chdir			显示当前目录的名称
type *.txt		显示当前目录所有.txt文件的内容
date 			显示当前日期，并提示输入新日期
date/t			只显示当前日期
time			显示当前时间，并提示输入新时间
time/t			只显示当前时间
tree d:			显示d盘的文件目录结构
```

# 查找

```
find "abc" c:test.txt		在 c:test.txt 文件里查找含 abc 字符串的行,如果找不到，将设 errorlevel 返回码为1
find /i “abc” c:test.txt	查找含 abc 的行，忽略大小写
find /c "abc" c:test.txt	显示含 abc 的行的行数
```

# 切换

```
# 避免盘符切换
在常用的地方建立文件夹为myFile, 把Windows的cmd.exe复制到myFile文件夹中，双击cmd.exe ---> ...\myFile>

# 切换盘符
盘符名			  进入新的盘符
c：				进入c盘
d：				进入d盘
e：				进入e盘

# 在一个盘符下
cd(或cd.) 	   进入当前目录
cd video		进入video目录
cd ..			进入上一级目录
cd\				退回根目录

# 在运行处直接进入指定目录
cmd /k d:		进入d:根目录
cmd /k cd/d d:\download		进入到指定盘符的指定目录中

:label			行首为:，表示改行为标签行，标签行不执行操作
goto label		跳转到指定的标签那一行

# 切换当前目录
@echo off
c: & cd & md mp3           #在 C: 建立 mp3 文件夹
md d:mp4                   #在 D: 建立 mp4 文件夹
cd /d d:mp4                #更改当前目录为 d:mp4
pushd c:mp3                #保存当前目录，并切换当前目录为 c:mp3
popd                        #恢复当前目录为刚才保存的 d:mp4
```

# 运行

```
切换至目录下，直接输入文件名，直接运行

&		顺序执行多条命令，而不管命令是否执行成功
&&		顺序执行多条命令，当碰到执行出错的命令后将不执行后面的命令
||		顺序执行多条命令，当碰到执行正确的命令后将不执行后面的命令
%0 %1 %2 %3 %4 %5 %6 %7 %8 %9 %*	命令行传递给批处理的参数
%0 批处理文件本身
%1 第一个参数
%9 第九个参数
%* 从第一个参数开始的所有参数

if 		判断命令
if "%1"=="/a" echo 第一个参数是/a
if /i "%1" equ "/a" echo 第一个参数是/a
/i 表示不区分大小写，equ 和 == 是一样的，其它运算符参见 if/?
if exist c:test.bat echo 存在c:test.bat文件
if not exist c:windows (
        echo 不存在c:windows文件夹
        )
if exist c:test.bat (
        echo 存在c:test.bat
        ) else (
        echo 不存在c:test.bat
        )
        
 for		
```



# 设置变量

```
引用变量可在变量名前后加 % ，即 %变量名%
set                        #显示目前所有可用的变量，包括系统变量和自定义的变量
set path				   #产看path下的环境变量	
echo %SystemDrive%         #显示系统盘盘符。系统变量可以直接引用
set p                      #显示所有以p开头的变量，要是一个也没有就设errorlevel=1
set p=aa1bb1aa2bb2         #设置变量p，并赋值为 = 后面的字符串，即aa1bb1aa2bb2
echo %p%                   #显示变量p代表的字符串，即aa1bb1aa2bb2
echo %p:~6%                #显示变量p中第6个字符以后的所有字符，即aa2bb2
echo %p:~6,3%              #显示第6个字符以后的3个字符，即aa2
echo %p:~0,3%              #显示前3个字符，即aa1
echo %p:~-2%               #显示最后面的2个字符，即b2
echo %p:~0,-2%             #显示除了最后2个字符以外的其它字符，即aa1bb1aa2b
echo %p:aa=c%              #用c替换变量p中所有的aa，即显示c1bb1c2bb2
echo %p:aa=%               #将变量p中的所有aa字符串置换为空，即显示1bb12bb2
echo %p:*bb=c%             #第一个bb及其之前的所有字符被替换为c，即显示c1aa2bb2
set p=%p:*bb=c%            #设置变量p，赋值为 %p:*bb=c% ，即c1aa2bb2
set /a p=39                #设置p为数值型变量，值为39
set /a p=39/10             #支持运算符，有小数时用去尾法，39/10=3.9，去尾得3，p=3
set /a p=p/10              #用 /a 参数时，在 = 后面的变量可以不加%直接引用
set /a p=”1&0″             #”与”运算，要加引号。其它支持的运算符参见set/?
set p=                     #取消p变量
set /p p=请输入
屏幕上显示”请输入”，并会将输入的字符串赋值给变量p
注意这条可以用来取代 choice 命令
注意变量在 if 和 for 的复合语句里是一次性全部替换的，如
@echo off
set p=aaa
if %p%==aaa (
        echo %p%
        set p=bbb
        echo %p%
        )
结果将显示
aaa
aaa
因为在读取 if 语句时已经将所有 %p% 替换为aaa
这里的"替换"，在 /? 帮助里就是指"扩充"、"环境变量扩充"
可以启用”延缓环境变量扩充”，用 ! 来引用变量，即 !变量名!
@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
set p=aaa
if %p%==aaa (
        echo %p%
        set p=bbb
        echo !p!
        )
ENDLOCAL
结果将显示
aaa
bbb
还有几个动态变量，运行 set 看不到
%CD%                      #代表当前目录的字符串
%DATE%                    #当前日期
%TIME%                    #当前时间
%RANDOM%                  #随机整数，介于0~32767
%ERRORLEVEL%              #当前 ERRORLEVEL 值
%CMDEXTVERSION%           #当前命令处理器扩展名版本号
%CMDCMDLINE%              #调用命令处理器的原始命令行
可以用echo命令查看每个变量值，如 echo %time%
注意 %time% 精确到毫秒，在批处理需要延时处理时可以用到
```

# 创建

```
echo> file.txt		 创建文件名为file.txt的文件
md video 		 在当前目录创建文件夹video
rem和::			注释行不执行操作

>		清除文件中原有的内容后再写入，若不存在则创建
>>		追加内容到文件末尾，而不会清除原有的内容
主要将本来显示在屏幕上的内容输出到指定文件中，指定文件如果不存在，则自动生成该文件
type c:test.txt >prn	屏幕上不显示文件内容，转向输出到打印机
echo hello world>con	在屏幕上显示hello world，实际上所有输出都是默认 >con 的
copy c:test.txt f: >nul	拷贝文件，并且不显示"文件复制成功"的提示信息，但如果f盘不存在，还是会显示出错信息
copy c:test.txt f: >nul 2>nul	不显示”文件复制成功”的提示信息，并且f盘不存在的话，也不显示错误提示信息
echo ^^W ^> ^W>c:test.txt	生成的文件内容为 ^W > W
^ 和 > 是控制命令，要把它们输出到文件，必须在前面加个 ^ 符号

<		从文件中获得输入信息，而不是从屏幕上
一般用于 date time label 等需要等待输入的命令
@echo off
echo 2005-05-01>temp.txt
date <temp.txt
del temp.txt
这样就可以不等待输入直接修改当前日期
```

# 编辑

```
ren d:temp tmp	对文件夹重命名 
ATTRIB			显示或更改文件属性
```

# 复制

```
# copy
copy c:test.txt d:test.bak		复制c:test.txt文件到d:并重命名为 test.bak
copy con test.txt				从屏幕上等待输入，按 Ctrl+Z 结束输入，输入内容存为test.txt文件,con代表屏幕，prn代表打印机，nul代表空设备
copy 1.txt + 2.txt 3.txt		合并 1.txt 和 2.txt 的内容，保存为 3.txt 文件,如果不指定 3.txt ，则保存到 1.txt
copy test.txt +					复制文件到自己，实际上是修改了文件日期

# xcopy(外部命令)
xcopy d:mp3 e:mp3 /s/e/i/y		复制 d:mp3 文件夹、所有子文件夹和文件到 e: ，覆盖已有文件
加 /i 表示如果 e: 没有 mp3 文件夹就自动新建一个，否则会有询问

```



# 显示

```
cls 				清屏
title 新标题		  设置命令窗口的标题
pause 				暂停命令
more c:test.txt		逐屏显示c:test.txt的文件内容
|		管道命令
dir *.* /s/a | find /c ".exe"	表示先执行 dir 命令，对其输出的结果执行后面的 find 命令，该命令行结果：输出当前文件夹及所有子文件夹里的.exe文件的个数
type c:test.txt|more	这个和 more c:test.txt 的效果是一样的
```

# 打开

```
# 打开应用程序
在环境变量中，添加路径后，再命令行直接输入文件名，直接打开应用程序

# start
start 文件夹路径		在命令行打开文件夹
批处理中调用外部程序的命令，否则等外部程序完成后才继续执行剩下的指令

# call
批处理中调用另外一个批处理的命令，否则剩下的批处理指令将不会被执行
有时有的应用程序用start调用出错的，也可以call调用

# choice
让用户输入一个字符，从而选择运行不同的命令，返回码errorlevel为1234……
win98里是choice.com
win2000pro里没有，可以从win98里拷过来
win2003里是choice.exe
choice /N /C y /T 5 /D y>nul
延时5秒
```

# 服务

```
sc create servicename binpath= "pathname -service " start= ? depend= Tcpip		创建服务
start参数值包括AUTO(自动),DEMAND（手动） ,DISABLED（禁用）
sc delete servicename 		删除该服务
net/sc start servicename	启动该服务
net/sc stop servicename		停止该服务
注：net和sc区别 net用于启动未禁止的服务,sc可以启动禁止的服务
```

# 删除

```
rd video			删除当前路径下的vido空文件夹
rd /s dir			删除目录dir及其子目录下所有文件
del dir				删除dir目录下的所有文件而不是目录，若有子目录，则不删除子目录文件
del file.txt		删除文件file.txt
del *.txt			删除所有txt文件
del *				删除所有文件
```

# 帮助

```
help				命令列表
help 命令名		  对命令的描述
```

# 开关

```
logoff						注销命令
exit  						退出命令行窗口
shutdown /p （或shutdown -h） 关闭本地计算机      
具体参数命令行键入 shutdown /?查看
shutdown -r					重启

shutdown -s -t 600
Note: -s表示关机,-t是时间,时间单位是秒。
在某个时间点关机
比如说凌晨一点钟,只需要输入 at 1:00 shutdown -s
取消自动关机命令
输入运行窗口shutdown -a ,确定后会有取消的提示。
```

# 常用

```
1、cleanmgr:打开磁盘清理工具
2、compmgmt.msc:计算机管理
3、conf:启动系统配置实用程序
4、charmap:启动字符映射表
5、calc:启动计算器
6、chkdsk.exe:Chkdsk磁盘检查
7、cmd.exe:CMD命令提示符
8、certmgr.msc:证书管理实用程序
9、Clipbrd:剪贴板查看器
10、dvdplay:DVD播放器
11、diskmgmt.msc:磁盘管理实用程序
12、dfrg.msc:磁盘碎片整理程序
13、devmgmt.msc:设备管理器
14、dxdiag:检查DirectX信息
15、dcomcnfg:打开系统组件服务
16、explorer:打开资源管理器
17、eventvwr:事件查看器
18、eudcedit:造字程序
19、fsmgmt.msc:共享文件夹管理器
20、gpedit.msc:组策略
21、iexpress:工具，系统自带
22、logoff:注销命令
23、lusrmgr.msc:本机用户和组
24、MdSched:来启动Windows内存诊断程序
25、mstsc:远程桌面连接
26、Msconfig.exe:系统配置实用程序
28、mspaint:画图板
29、magnify:放大镜实用程序
30、mmc:打开控制台
31、mobsync:同步命令
32、notepad:打开记事本
33、nslookup:网络管理的工具向导
34、narrator:屏幕“讲述人”
35、netstat:an(TC)命令检查接口
36、OptionalFeatures：打开“打开或关闭Windows功能”对话框
37、osk:打开屏幕键盘
38、perfmon.msc:计算机性能监测程序
39、regedt32:注册表编辑器
40、rsop.msc:组策略结果集
41、regedit.exe:注册表
42、services.msc:本地服务设置
44、sigverif:文件签名验证程序
45、shrpubw:创建共享文件夹
46.sfc /scannow-----启动系统文件检查器
47.route print------查看路由表
48.taskmgr 显示任务管理器
49.control userpasswords2-----User Account 权限设置
```

