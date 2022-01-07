# NFS挂载

[参考1](https://note.youdao.com/ynoteshare/index.html?id=da360a46ee23b3cd1fe94df586d5692d&type=note&_time=1637914489063)	[参考2](https://www.cnblogs.com/merely/p/10793877.html)

NFS 是Network File System的缩写，中文意思是网络文件系统。它的主要功能是通过网络（一般是局域网）让不同的主机系统之间可以共享文件或目录。NFS客户端（一般为应用服务器，例如web）可以通过挂载（mount）的方式将NFS服务器端共享的数据目录挂载到NFS客户端本地系统中（就是某一个挂载点下）。从客户端本地看，NFS服务器端共享的目录就好像是客户端自己的磁盘分区或者目录一样，而实际上却是远端的NFS服务器的目录。

## 安装

ubuntu 安装nfs

https://blog.csdn.net/csdn_duomaomao/article/details/77822883

centos 安装nfs

https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-centos-6

## 配置

> 注意：以root身份配置

服务器

```
Master: 12.34.56.789
Client: 12.33.44.555
```

### Server

- 安装依赖

```shell
# 检查
rpm -qa|grep nfs
rpm -qa|grep rpcbind

# 安装
sudo yum install nfs-utils nfs-utils-lib

# 设置开机启动
sudo chkconfig nfs on
sudo chkconfig rpcbind on 
# 开启
sudo service rpcbind start 
sudo service nfs start

sudo systemctl enable nfs.service
sudo systemctl enable rpcbind.service
sudo systemctl start nfs.service
sudo systemctl start rpcbind.service
```

- 配置分享目录

```shell
# 设置分享目录
sudo mkdir /opt/data
sudo chmod -R 777 /opt/data

# 配置exports文件
sudo vim  /etc/exports
# 文件末尾添加信息
/opt/data     12.33.44.555(rw,sync,no_root_squash,no_subtree_check)
# 参数含义
- rw: # 此选项允许客户端服务器在共享目录中读取和写入
- sync: # 只有在提交更改后，同步才会确认对共享目录的请求。 
- no_subtree_check: # 此选项可防止子树检查。当共享目录是较大文件系统的子目录时，nfs 会对其上方的每个目录进行扫描，以验证其权限和详细信息。禁用子树检查可能会增加 NFS 的可靠性，但会降低安全性。
- no_root_squash: # 这句话允许root连接到指定目录

# 刷新配置立即生效
sudo exportfs -a
```

- 查看分享目录

```shell
# 可用showmount -e 服务端ip来查看可mount目录
sudo showmount -e  192.168.1.1
```

###  Client

- 安装依赖

```shell
sudo yum install nfs-utils nfs-utils-lib
```

- 挂载目录

```shell
# 创建包含nfs分享的文件的目录
sudo mkdir /opt/data

# 挂载目录
sudo mount 12.34.56.789:/opt/data /opt/data

# 检查挂载
df -h  # 查看挂载的文件目录
mount  # 查看挂载信息
```

- 卸载

```shell
# 移除挂载
sudo umount /opt/data
```

- 自动化

您可以通过将目录添加到客户端上的 fstab 文件来确保挂载始终处于活动状态。这将确保在服务器重新启动后挂载启动。

```shell
sudo vim /etc/fstab
# 填写信息
12.34.56.789:/opt/data  /opt/data   nfs      auto,noatime,nolock,bg,nfsvers=3,intr,tcp,actimeo=1800 0 0
```

### 测试

客户端创建文件检查服务端

```shell
# 客户端创建
cd /opt/data
touch demo.txt
# 服务端检查
cd /opt/data
ls
```

### 其他

```shell
man nfs  		# 查看fstab更多选项
mount -a		# 在任何后续服务器重新启动后，使用单个命令挂载 fstab 文件中指定的目录

df -h
mount			# 检查已安装的目录
```