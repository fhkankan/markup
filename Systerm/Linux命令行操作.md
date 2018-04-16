[TOC]



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
# 清理
ctrl+l
# 查看隐藏文件
ctr+h
# 跳转到命令行末尾
ctr+e
# 跳转到命令行首部
ctr+a

```

# 运行

```
切换至目录下，输入./文件名, 直接执行
```
# 查看

## 文件目录

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
ls -l -h
# 当前的文件路径
pwd
# 以树状形式显示目录结构
tree
# 查看可执行命令位置
which command
```

## 查看进程
```python
# 显示终端上的所有进程，包括其他用户的进程
ps
-a  显示现行终端机下的所有程序,包括其他用户的程序
-u	显示进程的详细状态
-x	显示没有控制终端的进程
-w	显示加宽，以便显示更多的信息
-r	只显示正在运行的进程

# 动态显示进程
top 
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

# 根据端口号查pid进程
sudo lsof -i:端口号
sudo lsof -Pti:端口号

# 终止进程
kill [-signal] pid 
[0~15],其中9为绝对终止，可以处理一般无法终止的进程

# 查看端口被哪个程序占用
lsof -i:端口号 |grep "(LISTEN)"
netstat -tunlp|grep 端口号
```

## 查看磁盘

```python
df [-ahikHTm] [目录或文件名]
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

fdisk [-l] 装置名称
# 磁盘分区表操作工具
-l ：输出后面接的装置所有的分区内容。若仅有 fdisk -l 时， 则系统将会把整个系统内能够搜寻到的装置的分区均列出来。

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
    
mount [-t 文件系统] [-L Label名] [-o 额外选项] [-n]  装置文件名  挂载点
umount [-fn] 装置文件名或挂载点
# 磁盘挂载与卸除  
-f ：强制卸除！可用在类似网络文件系统 (NFS) 无法读取到的情况下；
-n ：不升级 /etc/mtab 情况下卸除。
  
```

## 查看网络

```python
# 查看网卡
ifconfig
# 修改ip
sudo ifconfig ens 33 新的ip地址
# 测试网络
ping (ip)
```

## 查看时间

```python
# 日历
cal
# 显示全年的日历
cal -y
# 时间
date
# 时间格式化
date "+%y-%m-%d"
```
##查看其它
```python
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

# 显示
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

head [-n number] 文件
# 只看头几行

tail [-n number] 文件
# 只看尾巴几行

|
管道：一个命令的输出可以通过管道作为另一个命令的输入
# 对于容量长的内容可以不用写入文件直接通过|在终端分屏输出
ls -ih | more

cat [-AbEnTv]
# 查看或者合并文件内容，从第一行开始
cat 1.txt 2.txt > 3.txt
-A ：相当於 -vET 的整合选项，可列出一些特殊字符而不是空白而已；
-b ：列出行号，仅针对非空白行做行号显示，空白行不标行号！
-E ：将结尾的断行字节 $ 显示出来；
-n ：列印出行号，连同空白行也会有行号，与 -b 的选项不同；
-T ：将 [tab] 按键以 ^I 显示出来；
-v ：列出一些看不出来的特殊字符

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

# 切换

```python
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

# 搜索

```python
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
# 在路径下搜索名字上带有'搜索内容'的文件或文件夹
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

find 路径 -name '搜索内容'
# 在路径下搜索容量为...的文件或文件夹
find 路径 -size [空/+/-]文件大小
空表示等于文件大小
+表示大于文件大小
-表示小于文件大小
# 查找路径下权限为...的文件或目录 
find 路径 -perm 权限数字
```

# 创建
```python
touch
# 在当前路径下创建一个新的空文件
touch[文件名]
# 在路径下创建文件
touch 路径/文件名

mkdir
# 创建文件夹
mkdir[目录名]
# 当前路径下创建可嵌套的文件夹
mkdir 目录名 -p
# 显示创建进度信息
mkdir 目录名 -v

/>>
# 将执行命令结果重定向到一个文件，把本应显示在终端上的内容保存到指定文件中
# 若test.txt不存在则创建，存在则覆盖
ls > test.txt
# 若test.txt不存在则创建，存在则追加
ls >> test.txt

ln
# 硬链接，占用相同的空间，无法创建文件夹的硬链接，会使硬链接数+1
ln 源文件 链接文件
# 软链接，不占用空间，若不在一个目录，源文件要使用绝对路径
ln -s 源文件 链接文件
```

# 删除
```python
# 删除当前空目录
rmdir
# 删除当前文件和目录
rm 文件名
# 以进行交互方式执行
rm 文件名 -i
# 强制删除，忽略不存在的文件，无需提示
rm 文件名 -f
# 以递归的方式把文件里面的所有文件删除
rm 文件夹名 -r
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
tar [参数] 打包文件名 文件
-z 指定压缩文件的格式为 tar.gz
-c 生成档案文件，创建打包文件
-v 列出归档解档的详细过程，显示进度
-f 指定档案文件名称，f后面一定是.tar文件，必须放在选项的最后
-t 列出档案中包含的文件
-x 解开档案文件

gzip [选项] 被压缩文件
-d 解压
-r 压缩所有子目录

# 文件的压缩
tar zcvf 压缩后的文件名 将要压缩的文件

# 文件的解压
tar xf 压缩后的文件名

# 查看压缩文件内容
zcat  压缩文件

# 分步打包、压缩、解压、解包
打包：tar -cvf xxx.tar *.txt
压缩：gzip xxx.tar  ---->  xxx.tar.gz
解压：gzip -d xxx.tar.gz ----> gzip xxx.tar
解包：tar -xvf xxx.tar

# 合并打包压缩、解压解包
打包压缩：tar -zcvf xxx.tar.gz *.txt
解压解包：tar -zxvf xxx.tar.gz 
          tar -zxvf xxx.tar.gz -C ../ 解压后放置的路径

# bz2文件
压缩：bzip2 xxx.tar  ---->  xxx.tar.bz2
解压：bzip2 -d xxx.tar.bz2 ----> gzip xxx.tar

# 组合
打包及压缩:tar -jcvf xxx.bz2 *.txt
解压并解包:tar -jxvf xxx.bz2
	         tar -jxvf xxx.bz2 -C ../ 解压后放置的路径

zip文件
压缩：zip 目标文件 源文件
解压：unzip -d 目录文件 压缩文件
  	unzip -d ../ xxx.zip 解压后放置的路径
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

# 改权限

```python
chmod u/g/o/a+/-/=rwx 文件名
u:文件所有者
g:同一用户组
o:其他人
a:这三者皆是
+：增加权限
-：撤销权限
=：设定权限

权限：
r：4,读取，可以通过ls查到目录的内容
w：2,写入，可以在目录下创建新文件
x：1,执行，可以通过cd进入
-: 0,不具有任何权限
```

# 作业管理

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
# 关机重启
```
sync			将数据由内存同步到硬盘中
reboot	        重新启动操作系统
halt			关闭系统
shutdown –r now	重新启动操作系统，shutdown会给别的用户提示
shutdown -h now	立刻关机，其中now相当于时间为0的状态
shutdown -h 20:25	系统在今天的20:25 会关机
shutdown -h +10	系统再过十分钟后自动关机
```
# 网络
```
ifconfig :查询计算机ip信息
ping ip地址 :检查网络连接状态
netstat - an ：检查端口使用状况
lsof -i [tcp/udp]:端口号，查看指定端口那个运行起来的程序在使用
kill 进程编号，杀死指定进程， kill -9 进程编号，强制杀死进程
```
# 软件操作
```
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


安装离线deb包
第一种：使用Ubuntu软件中心安装，即直接双击软件包就可以了；
第二种：使用dpkg命令方法安装：sudo dpkg -i package.deb；
第三种：使用apt命令方法安装：sudo apt-get install package.deb；

安装python第三方库
pip install  libname
python setup.py install
```

# 快捷命令

```python
# pycharm
vim ~/.bashrc
alias pycharm = "bash pycharm.sh所在的目录路径"
source ~/.bashrc
```











