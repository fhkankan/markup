# Screen

开启Screen后，只要Screen进程没有终止，其内部运行的会话都可以恢复，即使网络连接中断，也可以重新进入已开启的screen中，对中断的会话进行控制，包括恢复或删除。

## 介绍

screen是Linux系统下的一个非常 useful 的终端复用工具，主要功能和用法如下：

- 会话管理

可以在一个screen会话内同时运行多个终端，并在多个终端之间自由切换。

- 会话恢复

screen会话被切断后可以随时恢复，保持原样运行的程序不会被中断。

- 远程操作

可以对一个screen会话进行远程连接，从不同机器访问同一个screen。

- 多视窗

一个screen可以创建和管理多个视窗，用于运行不同的程序。

- 视窗及shell管理

支持视窗重命名、编号、切换；支持shell的后台、前台切换。

- 复制粘贴

支持屏幕滚动回滚，可以复制屏幕内容到粘贴板。

- 访问控制

可以通过密码保护一个screen，避免未经授权的访问。

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

```shell
# 创建新视窗
screen -S xx
# 查看已经创建过的视窗
screen -ls
# 退回到某视窗
screen -r xx
# 无法进入原先视窗的解决方法
screen -d xx
screen -r xx
# 关闭某视窗
screen -S xx -X quit
# 退出当前视窗
按下Ctrl+a，然后按下d来推出screen，此时程序仍在后台执行
# 将所有视窗调整为当前终端的大小
screen -A xx
# 将指定的screen进程离线
screen -d xx
# 指定视窗的缓冲区行数
screen -h xx
# 即使已经有screen作业在运行，仍强制建立新的Screen作业
screen -m xx
# 先尝试回复离线的作业，若找不到则建立新的screen作业
screen -R xx
# 指定建立新视窗时要执行的shell
screen -s xx
# 显示版本信息
screen -v xx
# 检查并删除无法使用的screen作业
screen -wipe xx
```

