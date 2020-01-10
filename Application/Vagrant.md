# Vagrant

[参考](https://www.cnblogs.com/hafiz/p/9175484.html)

Vagrant是一个基于Ruby的工具，用于创建和部署虚拟化开发环境。它 使用Oracle的开源[VirtualBox](https://baike.baidu.com/item/VirtualBox)虚拟化系统，使用 Chef创建自动化虚拟环境。我们可以使用它来干如下这些事：

- 建立和删除虚拟机
- 配置虚拟机运行参数
- 管理虚拟机运行状态
- 自动配置和安装开发环境
- 打包和分发虚拟机运行环境

Vagrant的运行，需要**依赖**某项具体的**虚拟化技术**，最常见的有VirtualBox以及VMWare两款，早期，Vagrant只支持VirtualBox，后来才加入了VMWare的支持。

 为什么我们要选择Vagrant呢？因为它有**跨平台**、**可移动**、**自动化部署无需人工参与**等优点。

## 安装

- 安装virtualbox

下载地址：https://www.virtualbox.org/wiki/Downloads

- 安装vagrant

下载地址：https://www.vagrantup.com/downloads.html

> 注意
>
> 下载的时候，virtualbox和vagrant的版本要搭配，建议都下载最新版的。还有就是要根据自己的操作系统版本进行选择32位或者64位下载。在windows系统中，可能还需要配置环境变量以及一定要开启VT-x/AMD-V硬件加速。

## 基本命令

- vagrant box

```shell
# 列出本地环境中所有的box
vagrant box list

# 添加box到本地vagrant环境
vagrant box add box-name(box-url)

# 更新本地环境中指定的box
vagrant box update box-name

# 删除本地环境中指定的box
vagrant box remove box-name

# 重新打包本地环境中指定的box
vagrant box repackage box-name
```
在线查找需要的box
官方网址：https://app.vagrantup.com/boxes/search

- vagrant

```shell
# 在空文件夹初始化虚拟机
vagrant init [box-name]

# 在初始化完的文件夹内启动虚拟机
vagrant up

# ssh登录启动的虚拟机
vagrant ssh

# 挂起启动的虚拟机
vagrant suspend

# 重启虚拟机
vagrant reload

# 关闭虚拟机
vagrant halt

# 查找虚拟机的运行状态
vagrant status

# 销毁当前虚拟机
vagrant destroy
```

