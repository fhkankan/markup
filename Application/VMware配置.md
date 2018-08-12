# 安装

- 安装VMware
- 安装操作系统
- 安装VMware Tool

```
实现主机和虚拟机之间的文件传输
```

# 网络配置

- 桥接模式

桥接模式的网络相当于虚拟机的网卡和主机的物理网卡均连接到由虚拟机软件提供的VMnet0虚拟交换机上，虚拟机和主机是平等的，相当于一个网络中的两台计算机，当设置虚拟机的IP与主机在同一网段时，即可实现主机和虚拟机之间的通信。在这种模式下，VMware虚拟出来的操作系统就像局域网中的一台独立主机，可以访问局域网内的任何一台计算机

> 配置虚拟机IP为静态IP

```
# 进入虚拟机终端
vi /etc/network/interfaces
# 编辑为如下内容
auto lo
iface lo inet loopback
auto etch0
iface eth0 inet static
address 192.168.0.4
netmask 255.255.255.0
gateway 192.168.0.1
# 重启网络
/etc/int.d/networking restart
# 测试
ifconfig/ping
```

- NAT模式

可使虚拟系统借助NAT(网络地址转换)功能通过主机所在的网络访问公网，就是主机再构建一个局域网，在该局域网中只有一台计算机(虚拟机)。虚拟机可以和主机通信，但不能和主机同级别的其他计算机通信。优势是不需要进行IP地址、子网掩码、网关灯配置即可实现虚拟机上网。

# 配置Samba共享

由于主机和虚拟机之间的文件系统可能不兼容，故不建议使用共享文件夹的方式实现大文件的的共享。采用Samba服务可解决这个问题

> 虚拟机中

```
# 下载安装samba
sudo apt-get install samba
# 在ubuntu中创建共享目录
mkdir /share_folder
chmod 777 /share_folder
# 修改配置文件
vim /etc/samba/smb.conf
# 在末尾添加如下
[share]
	path = /share_folder
	public = yes
	writable = yes
	browseable = yes
	avaliable = yes
	create mask = 0777
	directory mask = 0777
# 重启samba服务
sudo /etc/init.d/samba restart
```

> 主机中

```
# 在主机中添加虚拟机Samba共享目录的磁盘映射
在win10系统中，在主机"计算机"地址栏中输入\\\192.168.0.4(虚拟机IP地址),可看到名为Share的虚拟机共享文件夹，右键点击Share文件夹选择"映射网络驱动器",在弹出的界面中选择想要设置的盘符即可将虚拟机中的share_folder文件夹映射为主机中的网络磁盘
```

> 使用

主机的映射磁盘和虚拟机的share_folder目录等价，在此目录下即可实现大文件的共享

