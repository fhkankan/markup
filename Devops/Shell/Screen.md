# Screen

开启Screen后，只要Screen进程没有终止，其内部运行的会话都可以恢复，即使网络连接中断，也可以重新进入已开启的screen中，对中断的会话进行控制，包括恢复或删除。

## 介绍

screen是Linux系统下的一个非常 useful 的终端复用工具，主要功能和用法如下

1. 多窗口操作：通过 Screen 命令，你可以在同一个终端窗口中创建多个窗口，并在这些窗口中同时运行不同的应用程序，而不需要打开多个终端窗口。
2. 多任务操作：在一个窗口中使用 Screen 命令可以实现多任务操作，比如同时运行多个进程或命令等。
3. 断线恢复：如果你在使用远程连接时突然断开连接，那么在 Screen 命令下运行的任务仍然可以继续执行，并且在重新连接后可以通过 Screen 命令重新打开之前的会话，恢复之前的工作状态。
4. 后台运行：Screen 命令可以将一个命令或脚本放到后台运行，而不需要打开一个新的终端窗口或使用 nohup 命令。
5. 共享会话：使用 Screen 命令可以与其他用户共享一个会话，这对于协同工作或者远程技术支持非常有用。

screen创建的虚拟终端状态：

***Attached***：表示当前screen正在作为主终端使用，为活跃状态。

***Detached***：表示当前screen正在后台使用，为非激发状态

## 安装

- 有root权限

```
sudo apt install screen
```

- 无root权限

下载源码包进行安装，[地址](https://ftp.gnu.org/gnu/screen/)

```
# 解压
tar -zxvf screen...
# 安装
cd screen...
./configure --prefix=/home/whq/APP/Screen4.9.0/Path/  # 自己的安装路径
```

检查是否安装成功

```
screent -ls
```

## 使用

查看/进入

```shell
# 查看已经创建过的视窗
screen -ls
# 退回到某视窗
screen -r xx

# 无法进入原先视窗的解决方法
screen -d xx
screen -r xx

# 显示版本信息
screen -v xx
# 检查并删除无法使用的screen作业
screen -wipe xx
```

创建/编辑

```shell
# 创建新视窗
screen -S xx
# 即使已经有screen作业在运行，仍强制建立新的Screen作业
screen -m xx
# 先尝试回复离线的作业，若找不到则建立新的screen作业
screen -R xx

# 将所有视窗调整为当前终端的大小
screen -A xx
# 指定视窗的缓冲区行数
screen -h xx

# 当前视窗新建窗口
Ctrl+a，然后按c键创建一个新的虚拟终端窗口

# 窗口内命名
ctrl+a，然后按a键来为当前窗口重命名
```

退出/删除

```shell
# 窗口外部关闭某视窗
screen -S xx -X quit

# 将指定的screen进程离线
screen -d xx

# 完全退出screen会话并关闭所有窗口，在当前窗口
exit
# 关闭当前窗口
Ctrl+d
Ctrl+a，然后按k键来关闭
# 临时退出当前视窗
Ctrl+a，然后按d来退出screen，此时程序仍在后台执行
# 暂时禁用会话
Ctrl+a，然后按z键暂停screen会话，而不是完全分离它。恢复可以用fg命令

# 切换窗口
Ctrl+a，案后按n(下一个)或p(上一个)在多窗口间切换
```

