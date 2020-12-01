[TOC]

# 系统配置

```
# 更改系统默认引导
# 1.打开配置文件
cd /boot/grub/
vim grub.conf
# 2.修改信息
default = 0  # 表示设定使用哪行菜单进行系统启动的引导
timeout = 5  # 在无动作的情况下，默认5秒启动

# 不同登录界面
# 1.默认进入图形界面，若更改进入文本系统界面
vim /etc/inittab
id:5:initdefault:  # 将其更改为id:3:initdefault:即可在开机后进入文本系统界面
# 2.界面调换
# 文本回图形
终端中输入startx并enter
或终端中init 5并enter
# 图形回文本
使用了startx的直接注销
或终端中init 3
```

# 快捷键

```python
# 打开终端窗口
ctrl+alt+t
crt+shift+t
# 放大、减小终端窗口的字体显示
ctrl+/-
# 自动补全
Tab
# 中断执行
ctr+c
# 在使用过的命令之间来回切换
上/下光标键
# 查看隐藏文件
ctr+h
# 跳转到命令行末尾
ctr+e
# 跳转到命令行首部
ctr+a

# 清屏
ctrl+l
# 删除行首至光标所在处的所有字符
ctrl+u
# 删除光标所在处至行尾的所有字符
ctrl+k
```

windows下

```
ctrl + win + 向左    	窗口显示左半屏
ctrl + win + 向右    	窗口显示右半屏
ctrl + win + 向上    	窗口最大化
ctrl + win + T       	在当前窗口打开一个终端
ctrl + win + n		   	打开一个新的终端窗口
```

# 快捷命令

```python
# pycharm
vim ~/.bashrc
alias pycharm = "bash pycharm.sh所在的目录路径"
source ~/.bashrc
```

# 查看信息

网络连接

```python
# 查看ip
ifconfig
sudo ifconfig ens 33 新的ip地址：修改ip
# 查看网络连接
ping ip地址
# 查看ip+端口连接
telnet ip port
```

日期时间

```shell
# 日历
cal
# 显示全年的日历
cal -y
# 时间
date
# 时间格式化
date "+%y-%m-%d"
```

历史帮助

```shell
# 查看使用历史命令
history
# 执行历史命令
！历史命令索引
# 查看命令帮助
ls --help
# 使用手册查看命令
man ls
# 使用man时的操作键：
空格键 ---》 显示手册页的下一屏
Enter键---》 一次滚动手册页的一行
b      ---》 回滚一屏
f      ---》 前滚一屏
q      ---》 退出
/word  ---》 搜索word字符串

# 清楚终端上的显示
clear
```

查看程序执行时间

```
time -p python ./demo.py

# 结果
real 0.04	# 执行脚本总时间
user 0.03	# 执行脚本消耗CPU时间
sys 0.00	# 执行内核函数消耗时间
```

# 进程管理

## 查看进程

```python
ps
# 显示终端上的所有进程，包括其他用户的进程
-a  显示现行终端机下的所有程序,包括其他用户的程序
-u	显示进程的详细状态
-x	显示没有控制终端的进程
-w	显示加宽，以便显示更多的信息
-r	只显示正在运行的进程

pstree(选项)
# 树状显示进程信息
-a：显示每个程序的完整指令，包含路径，参数或是常驻服务的标示；
-c：不使用精简标示法；
-G：使用VT100终端机的列绘图字符；
-h：列出树状图时，特别标明现在执行的程序；
-H<程序识别码>：此参数的效果和指定"-h"参数类似，但特别标明指定的程序；
-l：采用长列格式显示树状图；
-n：用程序识别码排序。预设是以程序名称来排序；
-p：显示程序识别码；
-u：显示用户名称；
-U：使用UTF-8列绘图字符；
-V：显示版本信息。

pgrep(选项)(进程名)
# 以名称为依据从运行进程队列中查找进程，并显示查找到的进程id
# 每一个进程ID以一个十进制数表示，通过一个分割字符串和下一个ID分开，默认的分割字符串是一个新行。对于每个属性选项，用户可以在命令行上指定一个以逗号分割的可能值的集合。
-o：仅显示找到的最小（起始）进程号
-n：仅显示找到的最大（结束）进程号
-l：显示进程名称
-P：指定父进程号
-g：指定进程组
-t：指定开启进程的终端
-u：指定进程的有效用户ID


top 
# 动态显示进程
M	根据内存使用量来排序
P	根据cpu占有率来排序
T	根据进程时间的长短来排序
U	可以根据后面输入的用户名来筛选进程
K	可以根据后面面输入的PID来杀死进程
q	退出
h	获得帮助
htop

# 显示系统中名为xxx的进程
ps -aux|grep xxx

ps -elf|g

# 查使用内存最多的K个进程
ps -aux | sort -k4nr | head -K
或 top M

# 查使用CPU最多的K个进程
ps -aux | sort -k3nr | head -K
或 top P
```

## 停止进程

```shell
# 杀死单个进程
kill 进程编号，杀死指定进程， kill -9 进程编号，强制杀死进程

# 批量杀死进程
ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -15

管道符“|”用来隔开两个命令，管道符左边命令的输出会作为管道符右边命令的输入。 
“ps aux”是linux 里查看所有进程的命令。这时检索出的进程将作为下一条命令“grep python”的输入。 
“grep python”的输出结果是，所有含有关键字“python”的进程，这是python程序
“grep -v grep”是在列出的进程中去除含有关键字“grep”的进程。 
“cut -c 9-15”是截取输入行的第9个字符到第15个字符，而这正好是进程号PID。 
“xargs kill -15”中的xargs命令是用来把前面命令的输出结果（PID）作为“kill -15”命令的参数，并执行该令。 
“kill -15”会正常退出指定进程，-9强行杀掉
```

# 端口管理

| 端口类型 | 范围         | 说明                                                         |
| -------- | ------------ | ------------------------------------------------------------ |
| 公认端口 | 0～1023      | 与一些常见服务绑定                                           |
| 注册端口 | 1024～49151  | 它们松散地绑定于一些服务                                     |
| 私有端口 | 49152～65535 | 可用于任意软件与任何其他的软件通信的端口数<br/>使用因特网的传输控制协议，或用户传输协议 |

## 端口范围

查看系统端口范围

```shell
# 本地TCP/UDP的端口范围
cat /proc/sys/net/ipv4/ip_local_port_range
```

修改端口范围

```shell
# 方法一：
echo 1024 65535 > /proc/sys/net/ipv4/ip_local_port_range
# 方法二：
vim /etc/sysctl.conf
net.ipv4.ip_local_port_range = 1024 65000
```

## 查看端口

> 查看本机

nmap

```shell
# 查看本机开放的所有端口
nmap 127.0.0.1
```

netstat

```shell
netstat -anlp | grep 端口号
netstat -tunlp|grep 端口号
# 查看所有端口状态
netstat -ntlup  
```

lsof

```
sudo lsof -i:端口号
sudo lsof -Pti:端口号
lsof -i:端口号 |grep "(LISTEN)"
```

> 查看远程

telnet

```
telnet ip port
```

netcat

```
nc -vv ip port
```

## 开放端口

> iptables

```shell
# 安装
sudo apt-get install iptables
# 添加规则，开放5600端口接收广播
sudo iptables -A INPUT -p udp -d 0/0 -s 0/0 --dport 5600 -j ACCEPT
# 开放8000端口接收tcp
sudo iptables -A INPUT -p tcp -d 0/0 -s 0/0 --dport  8000 -j ACCEPT
# 保存规则，服务器重启则规则失效
sudo iptables-save
# 持续化规则
sudo apt-get install iptables-persistent
sudo netfilter-persistent save
sudo netfilter-persistent reload
# 设置防火墙规则
sudo ufw allow  5600
sudo ufw allow  8000
```

# 防火墙

ufw

```shell
# 安装
apt install ufw
# 启用
sudo ufw enable
# 关闭
sudo ufw disable
# 查看状态
sudo ufw status
# 允许端口访问
sudo ufw allow port
```

# 终端显示

```python
more
# 分屏显示文件内容
空白键 (space)：代表向下翻一页；
Enter         ：代表向下翻『一行』；
/字串         ：代表在这个显示的内容当中，向下搜寻『字串』这个关键字；
:f            ：立刻显示出档名以及目前显示的行数；
q             ：代表立刻离开 more ，不再显示该文件内容。
b 或 [ctrl]-b ：代表往回翻页，不过这动作只对文件有用，对管线无用。

less
# 与more类似，但是可以往前翻页
空白键    ：向下翻动一页；
[pagedown]：向下翻动一页；
[pageup]  ：向上翻动一页；
/字串     ：向下搜寻『字串』的功能；
?字串     ：向上搜寻『字串』的功能；
n         ：重复前一个搜寻 (与 / 或 ? 有关！)
N         ：反向的重复前一个搜寻 (与 / 或 ? 有关！)
q         ：离开 less 这个程序；

head [-n number] 文件名
# 只看头几行

tail [-n number] 文件名
# 只看尾巴几行

|
管道：一个命令的输出可以通过管道作为另一个命令的输入
# 对于容量长的内容可以不用写入文件直接通过|在终端分屏输出
ls -ih | more



tac
# 从最后一行开始显示

nl [-bnw] 文件
# 显示的时候，输出行号
-b ：指定行号指定的方式，主要有两种：
-b a ：表示不论是否为空行，也同样列出行号(类似 cat -n)；
-b t ：如果有空行，空的那一行不要列出行号(默认值)；
-n ：列出行号表示的方法，主要有三种：
-n ln ：行号在荧幕的最左方显示；
-n rn ：行号在自己栏位的最右方显示，且不加 0 ；
-n rz ：行号在自己栏位的最右方显示，且加 0 ；
-w ：行号栏位的占用的位数。
```

# 目录管理

## 查看目录

```python
# 当前路径下的文件与文件夹
ls
# 查看路径下的额文件与文件夹
ls 路径
# 显示指定目录下所有子目录与文件，包括隐藏文件
ls -a
# 以列表方式显示文件的详细信息
ls -l
# 以人性化的方式显示文件大小
ls -lh
ls -lht 
# 当前的文件路径
pwd
# 以树状形式显示目录结构
tree
# 查看可执行命令位置
which command
```

## 切换目录

```shell
/home/python
# 读法：/根目录下/home文件夹下/python文件夹里面
# 切换目录
cd[目录名]
# 切换至根目录
cd /
# 切换到当前目录 
cd .
# 切换至上一级目录
cd ..
# 切换至上一级目录的上一级目录
cd ../..
# 切换至上一历史目录
cd -
# 快速切换至家目录下
cd ~
```

## 创建目录

```shell
mkdir
# 创建文件夹
mkdir[目录名]
# 当前路径下创建可嵌套的文件夹
mkdir 目录名 -p
# 显示创建进度信息
mkdir 目录名 -v
```

# 文件管理

## 查找文件

```shell
正则表达式
# 搜寻以a开头的行
^a
# 搜寻以a结尾的行
a$
# 匹配[]里面一系列字符中的任一个
[Ll]
# 匹配一个非换行符的字符
.
# 匹配任意一个字符，可以为空(find)
*
# 匹配任意一个非空字符（find）
?


grep
# 在文件中查找特定的内容
# 注意：
1. 在查看某个文件的内容的时候，是需要有<文件名>
2. grep命令个在结合|(管道符)使用的情况下，后面的<文件名>是没有的
3. 可以通过 grep --help查看grep的帮助信息
grep [-选项] '搜索内容串' 文件名 
# 参数：
-c	只能输出匹配行的计数
-n	显示匹配行及行号
-v	显示不包含匹配文本的所有行
# 显示filename文件中不含有’s'字符的所有行
grep  -v 's' filename.txt
# 显示filename文件中含有’s'字符的行数
grep  -c 's' filename.txt
# 显示filename文件中含有’s'字符的行数和内容
grep  -n 's' filename.txt
# 忽略大小写，显示filename文件中含有’s'字符的内 
grep  -i 's' filename.txt
# 精确定位错误代码
grep -nr [错误关键字] *


find
# 查找符合条件文件并将查找结果输出
find [路径] [参数] [关键字]
# 参数
-name	按照文件名查找文件
-perm	按照文件权限来查找文件
-user	按照文件属性来查找文件
-group	按照文件所属的组来查找文件
-type	查找某一类型的文件
	b	块设备文件
	d	目录
	c	子符设备文件
	p	管道文件
	l	符号链接文件
	f	普通文件
-size n:[c] 查找文件长度为n块的文件，带有c时表示文件长度以字节计
-depth	在查找文件时，首相查找当前目录中文件，然后再在其子目录中查找
-mindepth n	查找文件时，查找当前目录中的第n层目录的文件，然后再在其子目录中查找
```

## 查看输出

```shell
cat
# 将文件内容连接后传送到标准输出或重定向到文件
cat [option][file]...
-A ：相当於 -vET 的整合选项，可列出一些特殊字符而不是空白而已；
-b ：列出行号，仅针对非空白行做行号显示，空白行不标行号！
-E ：将结尾的断行字节 $ 显示出来；
-n ：列印出行号，连同空白行也会有行号，与 -b 的选项不同；
-T ：将 [tab] 按键以 ^I 显示出来；
-v ：列出一些看不出来的特殊字符
-s : 将连续的两个空白行合并为一行
# eg:
cat 1.txt 2.txt > 3.txt
```

## 创建文件

```shell
touch
# 在当前路径下创建一个新的空文件
touch[文件名]
# 在路径下创建文件
touch 路径/文件名

/>>
# 将执行命令结果重定向到一个文件，把本应显示在终端上的内容保存到指定文件中
# 若test.txt不存在则创建，存在则覆盖
ls > test.txt
# 若test.txt不存在则创建，存在则追加
ls >> test.txt
```

## 权限更改

```shell
chmod
# 更改文件的访问权限
chmod u/g/o/a+/-/=rwx 文件名
u:文件所有者
g:同一用户组
o:其他人
a:这三者皆是
+：增加权限
-：撤销权限
=：设定权限
# 权限：
r：4,读取，可以通过ls查到目录的内容
w：2,写入，可以在目录下创建新文件
x：1,执行，可以通过cd进入
-: 0,不具有任何权限

chown
# 更改文件的所有者
chown [option]...[owner][:[group]]file
chown [option]...--reference=rfile file
-c:显示文件所有者更改后的信息
-f:忽略错误消息的输出
-R:以递归的方式更改目录及子目录的所有者
```

## 链接

```shell
ln
# 在文件之间创建链接，硬链接和软链接
ln [option]...[-T]target link_name
# 参数
-f:在链接时先将同名文件删除
-d:允许系统管理员硬链接自己的目录
-i:在删除同名文件时先询问
-n:在进行软链接时，将dist视为一般的档案
-s:创建软连链接
# 创建硬链接，占用相同的空间，无法创建文件夹的硬链接，会使硬链接数+1
ln 源文件 链接文件
# 创建软链接，不占用空间，若不在一个目录，源文件要使用绝对路径
ln -s 源文件 链接文件
# 修改软链接指向
ln -fs source target  # 文件
ln -fns source target  # 文件夹
# 删除软链接
rm -rf symbolic_name
```

## 删除

```shell
rm
# 删除文件和目录
rm [option]...file...
# 参数
-i:删除前逐一询问确认
-f:强行删除，无需逐一确认
-r:以地柜的方式将目录及子目录和文件逐一删除
```

## 计算字数

```shell
wc
# 计算文件或标准输出设备的字节数、字数或是列数
wc [option]...[file]...
# 参数
-c:只显示字节数
-l:只显示行数
-w:只显示字数
```

## 文件分割

```shell
split
# 将文件分割成制定大小的子文件
split [option][input][prefix]
# 参数
-a:指定用于构成输出名称文件后缀部分的字母数
-b:用于指定子文件的字节数
-l:指定每个输出文件的行数(默认1000行)
```

# 文件传输

```python
scp	将要传输的文件		要放置的位置

# 将本地文件推送到远程主机
scp python.tar.gz root@192.168.8.15:/root/
# 将远程主机上的文件拉取到本地
scp root@192.168.8.15:/root/python.tar.gz ./
    
# 远端主机文件放置位置的表示形式
远程连接的用户@远程主机:远程主机的目录路径
# 远端主机文件位置的表示形式
远程连接的用户@远程主机:远程主机的文件路径
```

# 文件的备份

```python
# 文件的备份要有一定的标志符号，使用时间戳
date  [option]
# 参数
%F	显示当前日期格式， %Y-%m-%d
%T	显示当前日期格式， %H:%M:%S

# 指定命令显示的格式
年月日	date + %Y%m%d
时分秒	date + %H%M%S

# 指定时间戳格式
年月日时分秒	date + %Y%m%d%H%M%S

# 备份命令效果格式
# 复制备份
cp nihao nihao-$(date + %Y%m%d%H%M%S)
# 移动备份
mv nihao nihao-$(date + %Y%m%d%H%M%S)
```

# 复制移动

```python
cp
# 拷贝文件到指定路径
文件名 ../新文件名：复制文件至路径下新文件
文件名 路径：复制文件至新路径
# 复制目录时使用，保留链接、文件属性，并递归地复制文件夹及文件
cp 文件夹名 目标文件夹名 -a
# 已存在的目标文件而不提示
cp 文件名 目标文件名 -f
# 交互式复制，在覆盖目标文件之前将给出提示要求用户确认
cp 文件名 目标文件名 -i
# 若给出的源文件是目录文件，则递归复制该目录下的所有子目录和文件
cp 文件夹名 目标文件夹名 -r
# 显示拷贝进度和路径
cp 文件名 目标文件名 -v

mv 
# 移动文件到指定路径
文件(夹)名 文件(夹)名: 同一目录下为重命名文件夹
文件名 ../新文件名：剪切文件至路径下新文件
文件名 路径：剪切文件至文件夹
# 禁止交互式操作，如有覆盖也不会给出提示
mv 文件名 文件名 -f
# 确认交互方式操作，如果mv将导致目标文件的覆盖，系统会询问是否重写，要求用户回答以免覆盖文件
mv 文件名 文件名 -i
# 显示移动进度和剪切的路劲
mv 文件名 文件名 -v
# 打开文件，编辑内容 
gedit
```

# 编辑修改

```python
sed
# 行文件编辑工具
sed [参数] '<匹配条件> [动作]' [文件名]

# 注意：
1. 可以通过sed --help查看sed的帮助信息
# 参数
参数为空	表示sed的操作效果，实际上不对文件进行编辑
-i		表示对文件进行编辑
		注意：mac中需在后面单独加上：-i ''
# 匹配条件
数字行号
关进字		---> '/关键字/'
			注意：隔离符号/可更换为@#！等			
# 动作详解(参数为i)
-a		在匹配到的内容下一行增加内容
-i		在匹配到的内容上一行增加内容
-d		删除匹配到的内容
-s		替换匹配到的内容

# 替换命令
# 替换每行首个匹配内容
sed -i 's#原内容#替换后内容#' 文件名
# 替换全部匹配内容
sed -i 's#原内容#替换后内容#g' 文件名
# 指定行号替换首个匹配内容
sed -i '行号s#原内容#替换后内容#' 文件名
# 首行指定列号替换匹配内容
sed -i 's#原内容#替换后内容#列号' 文件名
# 指定行号列号匹配内容
sed -i '行号s#原内容#替换后内容#列号' 文件名

# 增加操作
# 在指定行号的下一行增加内容
sed -i '行号a\增加的内容' 文件名
# 增加多行
sed -i '1,3a\增加内容' 文件名
# 在指定行号的当行增加内容
sed -i '行号i\增加的内容' 文件名
# 增加多行
sed -i '1,3i\增加内容' 文件名

# 删除操作
# 指定行号删除
sed -i '行号d' 文件名
# 删除多行
sed -i '1,3d' 文件名


awk
# 文档编辑工具，不仅能以行尾单位还能以列为单位处理文件

awk [参数] '[ 动作 ]' [ 文件名 ]

# 参数
-F		指定行的分隔符

# 动作
print	显示内容
	$0	显示文档所有内容
	$n	显示文档的第n列内容，若存在多个$n,他们之间使用逗号隔开

# 内置变量
FILENAME	当前输入文件的文件名，该变量是只读的
NR			指定显示行的行号
NF			输出最后一列的内容
OFS			输出格式的列分隔符，缺省是空格
FS			输入文件的类分隔符，缺省是连续的空格和Tab

# eg
# 打印指定列的内容
awk '动作' 文件名
# 打印指定行和列的内容
awk 'NR==行号 {动作}' 文件名
# 指定分割符查看内容
awk -F '列分隔符' '动作' 文件名
# 设置显示分割符，显示内容
awk 'BEGIN{OFS='列分割符'}{动作}' 文件名
```

# 压缩解压

```python
tar
# 建立、还原备份文件，本身无压缩功能，但支持归档式压缩
tar [options][archive_file_name][source_file]
# 参数
-c 生成档案文件，创建打包文件
-d:查找归档文件与文件系统的差异
-u:仅增加归档文件中没有的文件
-v 列出归档解档的详细过程，显示进度
-f 指定档案文件名称，f后面一定是.tar文件，必须放在选项的最后
-t 列出档案中包含的文件
-x 解开档案文件
-z 压缩或解压缩归档文件
# eg
tar zcvf 压缩后的文件名 将要压缩的文件  # 文件的压缩
tar xf 压缩后的文件名  # 文件的解压
zcat  压缩文件  # 查看压缩文件内容
# 分步打包、压缩、解压、解包
打包：tar -cvf xxx.tar *.txt
压缩：gzip xxx.tar  ---->  xxx.tar.gz
解压：gzip -d xxx.tar.gz ----> gzip xxx.tar
解包：tar -xvf xxx.tar
# 合并打包压缩、解压解包
打包压缩：tar -zcvf xxx.tar.gz *.txt
解压解包：tar -zxvf xxx.tar.gz 
          tar -zxvf xxx.tar.gz -C ../ 解压后放置的路径

zip
# 压缩程序，压缩后文件以“.zip”为后缀
zip[options][file1 file2...]
# 参数
-c:给压缩文件加上注释
-d:删除压缩文件内指定的文件
-g:将文件压缩后附加已有压缩文件
-j: 只保存文件名称及其内容
-m:删除被压缩文件的原文件
-o:将压缩文件的时间设置为与最新文件的时间相同
-q: 不显示命令执行的过程
-r:以递归方式处理指定目录下的文件   


bzip2
# 采用新的压缩演算法，压缩效果较好，但不能压缩目录
bzip2[options][filenames...]
# 参数
-c:将压缩与解压缩的结果发送到标准输出
-d:执行解压缩
-f:在压缩或解压缩过程中强行覆盖同名文件
-k:在压缩或解压缩过程中保留源文件
-t:测试压缩文件的完整性
-z:强制执行压缩
# 组合
打包及压缩:tar -jcvf xxx.bz2 *.txt
解压并解包:tar -jxvf xxx.bz2
	         tar -jxvf xxx.bz2 -C ../ 解压后放置的路径  
        
gzip
# 压缩后的文件以.gz为文件后缀名
gzip[options][name...]
# 参数
-a：使用ASCII格式模式
-f:强行压缩文件
-l:列出压缩文件的相关信息
-n:当压缩文件时不保存原来的文件名及时间戳，与-N选项功能相反
-q:忽略警告信息
    
    
    
unzip
# 解压缩.zip命令的压缩文件
unzip[options][filename]
# 参数
-c:将解压缩的结果系那是到屏幕，并对字符做适当的转换
-f:更新现有的文件
-l:显示压缩文件包含的文件
-a:对文本文件进行必要的字符转换
-C:当压缩文件时忽略文件名大小写
-n: 当解压缩时不覆盖原有的文件，与-o选项作用相反
    

bunzip2
# 解压缩.bz2格式的压缩包
bunzip2[-fkvsVL][filenames...]
# 参数
-f:当解压缩时强行覆盖同名文件
-k:当解压缩时保留原文件
-s:在执行命令时减少内存的使用
-v:显示解压缩过程的详细信息


gunzip
# 全名gun unzip，用于解压缩.gz格式的压缩文件
gunzip[options][-S suffix][name...]
# 参数
-l:显示压缩文件的相关信息
-L:显示版本及相关信息
-N:当解压缩时将含有原文件名称及时间戳的文件保存到解压缩文件中
-r:以递归方式将指定目录的所有文件及子目录一起处理
-S:更改压缩后缀字符串
```

# 程序管理

运行

```
切换至目录下，输入./文件名, 直接执行

其他见作业管理
```

执行过程

```
1.读取由键盘输入的命令行
2.分析命令，以命令名作为文件名，并将其他参数改造为系统调用execve()完成内部处理所要求的形式
3.终端进程调用fork()创建一个子进程
4.终端进程本身用系统调用wait4()来等待子进程完成(若是后台命令，则不等待)。当子进程运行时调用execve()，子进程根据文件名到目录中查找有关文件(命令解释程序构成的文件)，将它调入内存，执行这个程序(解释这条命令)
5.若命令末尾有&，则终端进程不需要系统调用wait4()等待，立即显示提示符，让用户输入下一条命令，跳转到步骤(1).若命令末尾没&,则终端进程要一直等待，当子进程完成处理后终止，向父进程(终端进程)报告，此时唤醒终端进程，子啊做必要的判断等工作后，终端进程显示提示符，让用户输入新的命令，重复上述处理过程

nohup xxx &
```

作业管理

```python
# 将“当前”作业放到后台“暂停”
ctrl+z
# 让命令在后台运行,"./myjob&"
command&
# 提交作业不再作业列表中
(./myjob &)
# 忽略hangup信号，防止shell关闭时程序停掉
nohup ./myjob &
# 忽略HUP信号
disown -hmyjob
# 观察当前后台作业状态
jobs
参数：
-l	除了列出作业号之外同时列出PID
-r	列出仅在后台运行(run)的作业
-s	仅列出暂停的作业
# 将后台作业调到前台来继续运行
fg %jobnumber(%可有可无)
# 将一个在后台暂停的命令，继续执行
bg %jobnumbern
# 终止后台作业
kill -signal %jobnumber
参数：
-l 列出当前kill能够使用的信号
signal：表示给后台的作业什么指示，用man 7 signal可知
-1（数字）：重新读取一次参数的设置文件（类似reload）
-2：表示与由键盘输入ctrl-c同样的动作
-9：立刻强制删除一个作业
-15：以正常方式终止一项作业。与-9不一样。
# 终止前台作业
ctr + c
```
# 磁盘管理

## 块复制

```shell
dd
# 将指定大小的块复制到一个文件
dd[operand]...
dd option
# 参数
if=file:输入文件名称，默认是标准输入
of=file:输出文件名称，默认是标准输出
bs=bytes:同时设置输入/输出的块大小，单位是字节
count=blocks:指定要复制的块数
cbs=bytes:每次转储的字节数，即指定转储缓冲区的大小
obs=bytes:每次输出的字节数，即指定块的大小
```

## 交换分区

```shell
mkswap
# 设置linux系统的交换分区
mkswap[option][-l label]device[size]
# 参数
-c:创建交换分区前先检查是否有损坏的区块
-v0:创建旧交换分区，此为预设值
-v1:创建新式交换分区
```

## 磁盘分区

```shell
fdisk
# 磁盘分区表操作工具
fdisk[-u][-b sectorsize][options] device
fdisk -l [-u][device...]
fdisk -s partition...
# 参数
-b:指定的磁盘分区的大小
-H:指定磁盘头数
-l:输出后面接的装置所有的分区内容。若仅有 fdisk -l 时， 则系统将会把整个系统内能够搜寻到的装置的分区均列出来。
-v:显示版本信息

mkfs [-t 文件系统格式] 装置文件名
# 磁盘格式化
-t ：可以接文件系统格式，例如 ext3, ext2, vfat 等(系统有支持才会生效)

fsck [-t 文件系统] [-ACay] 装置名称
# 磁盘检验
-t : 给定档案系统的型式，若在 /etc/fstab 中已有定义或 kernel 本身已支援的则不需加上此参数
-s : 依序一个一个地执行 fsck 的指令来检查
-A : 对/etc/fstab 中所有列出来的 分区（partition）做检查
-C : 显示完整的检查进度
-d : 打印出 e2fsck 的 debug 结果
-p : 同时有 -A 条件时，同时有多个 fsck 的检查一起执行
-R : 同时有 -A 条件时，省略 / 不检查
-V : 详细显示模式
-a : 如果检查有错则自动修复
-r : 如果检查有错则由使用者回答是否修复
-y : 选项指定检测每个文件是自动输入yes，在不确定那些是不正常的时候，可以执行 # fsck -y 全部检查修复。
```

## 磁盘空间

```shell
df [options] [目录或文件名]
# 查看整个电脑存储情况
-a ：列出所有的文件系统，包括系统特有的 /proc 等文件系统；
-k ：以 KBytes 的容量显示各文件系统；
-m ：以 MBytes 的容量显示各文件系统；
-h ：以人们较易阅读的 GBytes, MBytes, KBytes 等格式自行显示；
-H ：以 M=1000K 取代 M=1024K 的进位方式；
-T ：显示文件系统类型, 连同该 partition 的 filesystem 名称 (例如 ext3) 也列出；
-i ：不用硬盘容量，而以 inode 的数量来显示

dudu [-ahskm] 文件或目录名称 		
# 检测目录所占磁盘空间
-a ：列出所有的文件与目录容量，因为默认仅统计目录底下的文件量而已。
-h ：以人们较易读的容量格式 (G/M) 显示；
-s ：列出总量而已，而不列出每个各别的目录占用容量；
-S ：不包括子目录下的总计，与 -s 有点差别。
-k ：以 KBytes 列出容量显示；
-m ：以 MBytes 列出容量显示；
```

## 挂载文件系统

```shell
mount
# 将某个磁盘分区的内容解读成文件系统，然后将其挂载到目录的某个位置之下
mount [-lhV]
mount -a [-fFnrsvw][-t vfstype][-O optlist]
mount [-fnrsvw][-o options[,...]] device | dir

mount [-t 文件系统] [-L Label名] [-o 额外选项] [-n]  装置文件名  挂载点
umount [-fn] 装置文件名或挂载点
# 参数
-a:将/etc/fstab文件中定义的所有文件系统挂载
-f:模拟整个文件系统挂载的过程
-n:挂载未写入/etc/mtab文件的文件系统
-L:将含有特定标签的硬盘分割挂载
-U uuid:将指定标识符的分区挂载
-o ro:用只读模式挂载
-o rw:用可读写模式挂载
```

# 账户

```python
# 远程登陆
ssh 用户名@ip地址
# 退出登录
exit
# 进入root用户
sudo -s
# 当前登录用户名
whoami
# 当前哪些用户在登录
who
# 切换用户
su 用户名

# 创建超级用户密码
sudo passwd

# 用户口令的管理
passwd 选项 用户名
-l 锁定口令，即禁用账号。
-u 口令解锁。
-d 使账号无口令。
-f 强迫用户下次登录时修改口令。
当前用户下，使用passwd,为修改当前用户密码

# 添加用户
sudo useradd 选项 用户名
-c comment 指定一段注释性描述。
-d 目录 指定用户主目录，如果此目录不存在，则同时使用-m选项，可以创建主目录。
-g 用户组 指定用户所属的用户组。
-G 用户组，用户组 指定用户所属的附加组。
-s Shell文件 指定用户的登录Shell。
-u 用户号 指定用户的用户号，如果同时有-o选项，则可以重复使用其他用户的标识号。
useradd -d /user/sam -m sam
# 此命令创建了一个用户sam，其中-d和-m选项用来为登录名sam产生一个主目录/usr/sam（/usr为默认的用户主目录所在的父目录）
useradd -s /bin/sh -g group –G adm,root gem
# 此命令新建了一个用户gem，该用户的登录Shell是 /bin/sh，它属于group用户组，同时又属于adm和root用户组，其中group用户组是其主组。 

# 修改用户
usermod 选项 用户名
# 用户组新创建的用户，默认不能sudo,需进行以下操作
sudo usermod -a -G adm 用户名
sudo usermod -a -G sudo 用户名
usermod -s /bin/ksh -d /home/z –g developer sam
# 此命令将用户sam的登录Shell修改为ksh，主目录改为/home/z，用户组改为developer

# 删除用户
sudo userdel -r 用户名
```


# 包管理器

```shell
sudo apt-cache search package             --->搜索软件包
sudo apt-cache show package               --->获取包的相关信息，如说明、大小、版本等
sudo apt-cache depends package            --->了解使用该包依赖哪些包
sudo apt-cache rdepends package           --->查看该包被哪些包依赖
sudo apt-get check                        --->检查是否有损坏的依赖
sudo apt-get update                       --->更新源
sudo apt-get upgrade                      --->更新已安装的包
sudo apt-get dist-upgrade                 --->升级系统
sudo apt-get source package               --->下载该包的源代码
sudo apt-get install package              --->安装包
sudo apt-get install package --reinstall  --->重新安装包
sudo apt-get -f install                   --->修复安装
sudo apt-get build-dep package            --->安装相关的编译环境
sudo apt-get remove package               --->删除包
sudo apt-get remove package --purge       --->删除包，包括配置文件等
sudo apt-get clean && sudo apt-get autoclean--->清理无用的包


# 安装离线deb包
第一种：使用Ubuntu软件中心安装，即直接双击软件包就可以了；
第二种：使用dpkg命令方法安装：sudo dpkg -i package.deb；
第三种：使用apt命令方法安装：sudo apt-get install package.deb；
第四中：使用gbebi:sudo apt-get install gdebi,sudo gbedi package.deb



# 安装二进制包(.bin)
chmod u+x 软件名  # 添加可执行权限
./软件名  # 执行要安装的二进制文件

安装python第三方库
pip install  libname
python setup.py install
```

> deb包

```bash
# 方法一
安装deb软件包 dpkg -i xxx.deb
删除软件包 dpkg -r xxx.deb
连同配置文件一起删除 dpkg -r --purge xxx.deb
查看软件包信息 dpkg -info xxx.deb
查看文件拷贝详情 dpkg -L xxx.deb
查看系统中已安装软件包信息 dpkg -l
重新配置软件包 dpkg-reconfigure xxx

# 方法二

安装gbebi: sudo apt-get install gdebi
安装软件 sudo gbedi package.deb
```

> 源码

```
# 安装源码包
tar -zvxf 源码压缩包  # 解压源码包
cd 目录中  # 进入安装包目录
./configure  # 执行安装前的配置
make  # 编译
make test  # 测试
make install  # 安装

# 卸载源代码编译的的软件：
cd 源代码目录
make clean
./configure
（make）
make uninstall
rm -rf 目录
```



# 系统管理

## 终止进程

```shell
kill
# 终止执行中的程序
kill [-s signal | -p][-a][--] pid...
kill -l [signal]
# 参数
-l:显示信号的信息
-s:指定要发送的信号
# eg
kill -l  # 显示所有信号的信息
kill -9 pid  # 强行终止进程号为pid的进程

skill(选项)
# 用于向选定的进程发送信号，冻结进程
-f：快速模式；
-i：交互模式，每一步操作都需要确认；
-v：冗余模式；
-w：激活模式；
-V：显示版本号；
-t：指定开启进程的终端号；
-u：指定开启进程的用户；
-p：指定进程的id号；
-c：指定开启进程的指令名称。

killall(选项)(进程名)
# 使用进程的名称来杀死进程，使用此指令可以杀死一组同名进程
-e：对长名称进行精确匹配；
-l：忽略大小写的不同；
-p：杀死进程所属的进程组；
-i：交互式杀死进程，杀死进程前需要进行确认；
-l：打印所有已知信号列表；
-q：如果没有进程被杀死。则不输出任何信息；
-r：使用正规表达式匹配要杀死的进程名称；
-s：用指定的进程号代替默认信号“SIGTERM”；
-u：杀死指定用户的进程。

pkill(选项)(进程名)
# 按照进程名杀死进程
-o：仅向找到的最小（起始）进程号发送信号；
-n：仅向找到的最大（结束）进程号发送信号；
-P：指定父进程号发送信号；
-g：指定进程组；
-t：指定开启进程的终端。
```

## 用户信息

```shell
last
# 显示系统开机以来记录用户登录、系统重启等信息的列表清单
last[options][name...] [tty...]
# 参数
-a:在最后一行显示主机名
-R:忽略显示主机名
-o:以旧格式读取wtmp文件
-x:显示系统条目和运行级别的变化
```

## 系统内存

```shell
free
# 显示内存信息，包括物理内存、虚拟的交换文件内存、共享内存区段及系统主要使用的缓冲等
free [options][-s delay][option]
# 参数
-b:以字节为单位显示内存使用情况
-k:以kb为单位显示内存使用情况
-m:以mb为单位显示内存使用情况
-o:不显示缓冲区调节列
-s:持续观察内存使用情况
-t:以kb为单位显示内存值，并统计每列值得总和
```

## 系统信息

```shell
uname
# 显示系统信息
uname [option]...
# 参数
-a:显示系统概要信息
-m:显示系统主机类型
-n:显示系统的计算机主机名
-r:显示系统发行版的内核编号
```

## 日期时间

```shell
date
# 可以不同格式显示或设置当前的系统时钟值
date [option]...[+format]
# 参数
-d:以string格式来显示时间
-r:显示文件最后的修改时间
-s:将系统时间设为datestr中描述的格式
-u:显示或设置通用时间值

hwclock
# 设置系统硬件时钟
hwclock[options][--ser--date=<date and time>]
# 参数
--adjust:估算硬件时钟的偏差，并用来校正硬件时钟
--drectisa:直接以IO指令来存取硬件时钟
--hctosys:将系统时钟值调整为与目前硬件时钟值一致
--show:显示硬件时钟的时间与日期
--systohc:将硬件时钟值调整为与目前的系统时钟值一致
```

# 关机重启

```
sync			将数据由内存同步到硬盘中
reboot	        重新启动操作系统
halt			关闭系统
init 0 			关机
shutdown –r now	重新启动操作系统，shutdown会给别的用户提示
shutdown -h now	立刻关机，其中now相当于时间为0的状态
shutdown -h 20:25	系统在今天的20:25 会关机
shutdown -h +10	系统再过十分钟后自动关机
```

# 







