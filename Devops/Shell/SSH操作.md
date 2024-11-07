# SSH使用
## 登录访问

**密码登录**

```bash
# 在客户端登录到远端计算机
ssh user_name@远端计算机IP
# 输入密码
*****
```

非22端口

```shell
ssh -p 8022 user_name@远端计算机ip
```

保持长链接

```shell
# 编辑ssh_config文件
sudo vi /etc/ssh/ssh_config
# 在Host *下面加入如下命令，命令含义：每隔60s客户端向服务器发送一个空包
ServerAliveInterval 60
```

- 别名访问

```
# 配置
sudo vi ~/.ssh/config  # 创建打开文件

Host master1  # 指定别名为 master1
hostname 192.168.56.11  # 指定目标 ip
user root  # 指定登录用户名

# 效果
ssh root@192.168.56.11 ==> ssh master1
```

- 免密登陆

```shell
# 首先使用生成密钥
ssh-keygen -t rsa
# 将id_rsa.pub中的内容复制到远端计算机的.ssh/authorized_keys文件中，就可无密码访问远端计算机
ssh-copy-id user@host
# 无密码访问
ssh user_name@远端计算机IP
```

4台机器互相免密

```shell
# 4台机器分别执行密钥生成
ssh-keygen  -t  dsa  -P  ''  -f  ~/.ssh/id_dsa
# 分别将4台机器的密钥追加到authorized_keys中
cat ~/.ssh/id_dsa.pub >> ~/.ssh/authorized_keys
# 将汇总了4台机器密钥的authorized_keys文件分发给4台机器
scp ~/.ssh/authorized_keys 其他服务器的.ssh地址
```

## 隧道建立

[参考](https://www.cnblogs.com/fbwfbi/p/3702896.html)

<img src="/Users/henry/Markup/Application/images/011534166739962.jpg" alt="011534166739962" style="zoom:50%;" />

基础条件：A-B-C由于防火墙无实现A对C的访问，A-B-D，D-B-C是可以正常访问，D服务器有独立的ip，运行OpenSSH服务器（`GatewayPorts为yes`）

### 本地隧道

条件

```
1. A-B-C由于防火墙无实现A对C的访问，A-B-D，D-B-C是可以正常访问，D服务器有独立的ip，运行OpenSSH服务器
2. 
中间服务器d的IP地址:123.123.123.123
目标服务器c的IP地址:234.234.234.234
目标服务器c的端口:21
```

目的

```
A机器的端口2121实现对C机器的端口21的访问
```

建立隧道并访问

```shell
# 建立隧道
# 在A机器上执行
ssh -N -f -L 2121:234.234.234.234:21 123.123.123.123
# 参数含义
-N 告诉SSH客户端，这个连接不需要执行任何命令。仅仅做端口转发
-f 告诉SSH客户端在后台运行
-L 做本地端口映射，X:Y:Z的含义是，将IP为Y的机器的Z端口通过中间服务器映射到本地机器的X端口

# 实现访问
# 在A机器访问本地2121端口，实现连接234.234.234.234的21端口
ftp localhost:2121 
```

其他

```shell
# 建立本地ssh隧道，可以指定本地主机的地址
ssh -Nf -L 192.168.0.100:2121:234.234.234.234:21 123.123.123.123
# 那么本地局域网的任何机器访问192.168.0.100:2121都会自动被映射到234.234.234.234:21


# 从某主机的80端口开启到本地主机2001端口的隧道
ssh -N -L 2001:localhost:80 somemachine
# 可以在浏览器中输入http://localhost:2001访问这个网站。


ssh -N -f -L 6000:<内网服务器ip>:22 -p <跳板机端口> username@<跳板机ip> -o TCPKeepAlive=yes
# 参数含义
-N 告诉SSH客户端，这个连接不需要执行任何命令。仅仅做端口转发
-f 告诉SSH客户端在后台运行
-L 做本地映射端口
# 登录本地的6000端口就相当于登录内网服务器
ssh -p 6000 服务器用户名@localhost
```

实例

```shell
# 通过跳板机隧道访问其他内网服务器
# 配置条件
1.外网不可访问服务器3的数据库
1.可通过外网访问服务器server1
2.服务器server1可访问服务器server2
3.服务器server2可访问服务器server3数据库

# 实现目的
外网通过访问server1来访问服务器3的数据库

# 配置步骤
1.在server1上通过ssh建立server1的端口映射到server3
ssh -o TCPKeepAlive=yes -NCPf user@server2 -L 0.0.0.0:33661:server3:3306
2.通过ssh连接到server1,访问其33661端口，即是访问server3的数据库
ssh user@server1
mysql -uroot -P 33661 -p
```

### 远程隧道

条件

```
1.不能通过互联网直接访问A（通过路由器接入互联网，无独立ip），但是可以通过A-B-D来完成D-B-A连接
2. 
需要访问内部机器A的远程机器D的IP地址:123.123.123.123
需要让远程机器能访问的内部机器的IP地址(这里因为是想把本机映射出去，因此IP是127.0.0.1)
需要让远程机器能访问的内部机器的端口号(端口:22)
```

目的

```
D机器的222端口访问A机器22端口
```

建立隧道和访问

```shell
# 建立隧道
# 在ip为192.168.0.100的A主机上执行下面的命令
ssh -Nf -R 2222:127.0.0.1:22 123.123.123.123
# 参数
-R 做远程端口映射，X:Y:Z 就是把我们内部的Y机器的Z端口映射到远程机器的X端口上。

# 实现访问
# 在ip为123.123.123.123的D机器上登陆ip是192.168.0.100的A机器
ssh -p 2222 localhost
```

其他

```shell
# 建立远程的ssh隧道时，可以指定公网的主机地址，不过一般情况是要访问内网的主机，所以这条命令应该在任何一台内网主机上执行，比如在192.168.0.102的主机上运行
ssh -Nf -R 123.123.123.123:2222:192.168.0.100:22 123.123.123.123

# 只要在局域网里192.168.0.102可以直接连接内网主机192.168.0.100，且192.168.0.102可以直接与公网主机123.123.123.123建立ssh连接。那么任何外网主机通过访问公网主机123.123.123.123:2222就会被连接到192.168.0.100:22，从而可以完成外网穿越NAT到内网的访问，而不需要在内网网关和路由器上做任何操作。
```

### socks服务器

如果我们需要借助一台中间服务器访问很多资源，一个个映射显然不是高明的办法。幸好，SSH客户端为我们提供了通过SSH隧道建立SOCKS服务器的功能。

通过下面的命令我们可以建立一个通过123.123.123.123的SOCKS服务器。

```shell
ssh -N -f -D 1080 123.123.123 # 将端口绑定在127.0.0.1上
ssh -N -f -D 0.0.0.0:1080 123.123.123.123 # 将端口绑定在0.0.0.0上
```

通过SSH建立的SOCKS服务器使用的是SOCKS5协议，在为应用程序设置SOCKS代理的时候要特别注意

## 其他连接

**通过中间主机建立SSH连接**

Unreachable_host表示从本地网络无法直接访问的主机，但可以从reachable_host所在网络访问，这个命令通过到reachable_host的“隐藏”连接，创建起到unreachable_host的连接。

```
ssh -t reachable_host ssh unreachable_host
```

**创建到目标主机的持久化连接**

```
ssh -MNf <user>@<host>
在后台创建到目标主机的持久化连接，将这个命令和你~/.ssh/config中的配置结合使用：

Host host
ControlPath ~/.ssh/master-%r@%h:%p
ControlMaster no
所有到目标主机的SSH连接都将使用持久化SSH套接字，如果你使用SSH定期同步文件（使用rsync/sftp/cvs/svn），这个命令将非常有用，因为每次打开一个SSH连接时不会创建新的套接字。
```

**保持SSH会话永久打开**

```
auto ssh -M50000 -t server.example.com ‘screen -raAd mysession’

打开一个SSH会话后，让其保持永久打开，对于使用笔记本电脑的用户，如果需要在Wi-Fi热点之间切换，可以保证切换后不会丢失连接。
```

**通过SSH连接屏幕**

```
ssh -t remote_host screen –r

直接连接到远程屏幕会话（节省了无用的父bash进程）。
```

**通过SSH运行复杂的shell命令**

```
ssh host -l user $(<cmd.txt)
更具移植性的版本：

ssh host -l user “`cat cmd.txt`”
```

**如果建立一个可以重新连接的远程GNU screen**

```
ssh -t user@some.domain.com /usr/bin/screen –xRR

人们总是喜欢在一个文本终端中打开许多shell，如果会话突然中断，或你按下了“Ctrl-a d”，远程主机上的shell不会受到丝毫影响，你可以重新连接，其它有用的screen命令有“Ctrl-a c”（打开新的shell）和“Ctrl-a a”（在shell之间来回切换），请访问http://aperiodic.net/screen/quick_reference阅读更多关于screen命令的快速参考。
```

**声音传输**

```
# 将你的麦克风输出到远程计算机的扬声器
dd if=/dev/dsp | ssh -c arcfour -C username@host dd of=/dev/dsp

这样来自你麦克风端口的声音将在SSH目标计算机的扬声器端口输出，但遗憾的是，声音质量很差，你会听到很多嘶嘶声。
```

## 检测分析

**端口测试**

```
knock <host> 3000 4000 5000 && ssh -p <port> user@host && knock <host> 5000 4000 3000
在一个端口上敲一下打开某个服务的端口（如SSH），再敲一下关闭该端口，需要先安装knockd，下面是一个配置文件示例。

[options]
logfile = /var/log/knockd.log
[openSSH]
sequence = 3000,4000,5000
seq_timeout = 5
command = /sbin/iptables -A INPUT -i eth0 -s %IP% -p tcp –dport 22 -j ACCEPT
tcpflags = syn
[closeSSH]
sequence = 5000,4000,3000
seq_timeout = 5
command = /sbin/iptables -D INPUT -i eth0 -s %IP% -p tcp –dport 22 -j ACCEPT
tcpflags = syn
```

**实时SSH网络吞吐量测试**

```
yes | pv | ssh $host “cat > /dev/null”
通过SSH连接到主机，显示实时的传输速度，将所有传输数据指向/dev/null，需要先安装pv。

如果是Debian：

apt-get install pv
如果是Fedora：

yum install pv
（可能需要启用额外的软件仓库）。
```

**通过SSH W/ WIRESHARK分析流量**

```
ssh root@server.com ‘tshark -f “port !22″ -w -' | wireshark -k -i –

使用tshark捕捉远程主机上的网络通信，通过SSH连接发送原始pcap数据，并在wireshark中显示，按下Ctrl+C将停止捕捉，但也会关闭wireshark窗口，可以传递一个“-c #”参数给tshark，让它只捕捉“#”指定的数据包类型，或通过命名管道重定向数据，而不是直接通过SSH传输给wireshark，我建议你过滤数据包，以节约带宽，tshark可以使用tcpdump替代：

ssh root@example.com tcpdump -w – ‘port !22′ | wireshark -k -i –
```

**更稳定，更快，更强的SSH客户端**

```
ssh -4 -C -c blowfish-cbc
强制使用IPv4，压缩数据流，使用Blowfish加密。
```

**使用cstream控制带宽**

```
tar -cj /backup | cstream -t 777k | ssh host ‘tar -xj -C /backup’
使用bzip压缩文件夹，然后以777k bit/s速率向远程主机传输。Cstream还有更多的功能，请访问http://www.cons.org/cracauer/cstream.html#usage了解详情，例如：

echo w00t, i’m 733+ | cstream -b1 -t2
```

## 文件处理

### 对比

**比较远程和本地文件 **

```
ssh user@host cat /path/to/remotefile | diff /path/to/localfile –
```

### 复制
**复制文档**

```bash
# 同cp命令一样，复制目录时可以加上-r选项。
scp -r 文件 账户@远端计算机IP:目录名	# 从本地到服务器
scp -r 账户@远端计算机IP:文件 本地目录	# 从服务器到本地
# 非22端口
scp -P 5002 文件 账户@远端计算机IP:目录名
scp -P 5002 账户@远端计算机IP:文件  本地目录

# 继续SCP大文件
rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file
它可以恢复失败的rsync命令，当你通过VPN传输大文件，如备份的数据库时这个命令非常有用，需要在两边的主机上安装rsync。

rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file local -> remote
或
rsync –partial –progress –rsh=ssh $user@$host:$remote_file $destination_file remote -> local
```
**上传下载 **

```bash
# 上传文件到远端计算机命令：
put 本地文件 远端计算机目录

# 下载文件到本地计算机命令：
get 远端文件 本地目录
```
**通过SSH将MySQL数据库复制到新服务器**

```
mysqldump –add-drop-table –extended-insert –force –log-error=error.log -uUSER -pPASS OLD_DB_NAME | ssh -C user@newhost “mysql -uUSER -pPASS NEW_DB_NAME”

通过压缩的SSH隧道Dump一个MySQL数据库，将其作为输入传递给mysql命令，我认为这是迁移数据库到新服务器最快最好的方法。
```

**从一台没有SSH-COPY-ID命令的主机将你的SSH公钥复制到服务器**

```
cat ~/.ssh/id_rsa.pub | ssh user@machine “mkdir ~/.ssh; cat >> ~/.ssh/authorized_keys”

如果你使用Mac OS X或其它没有ssh-copy-id命令的*nix变种，这个命令可以将你的公钥复制到远程主机，因此你照样可以实现无密码SSH登录。
```

**一步将SSH公钥传输到另一台机器**

```
ssh-keygen; ssh-copy-id user@host; ssh user@host

这个命令组合允许你无密码SSH登录，注意，如果在本地机器的~/.ssh目录下已经有一个SSH密钥对，ssh-keygen命令生成的新密钥可能会覆盖它们，ssh-copy-id将密钥复制到远程主机，并追加到远程账号的~/.ssh/authorized_keys文件中，使用SSH连接时，如果你没有使用密钥口令，调用ssh user@host后不久就会显示远程shell。
```

**将标准输入（stdin）复制到你的X11缓冲区**

```
ssh user@host cat /path/to/some/file | xclip

你是否使用scp将文件复制到工作用电脑上，以便复制其内容到电子邮件中？xclip可以帮到你，它可以将标准输入复制到X11缓冲区，你需要做的就是点击鼠标中键粘贴缓冲区中的内容。
```

### 删除

```
# 删除文本文件中的一行内容，有用的修复
在这种情况下，最好使用专业的工具。
ssh-keygen -R <the_offending_host>

# 删除文本文件中的一行，修复“SSH主机密钥更改”的警告
sed -i 8d ~/.ssh/known_hosts
```

## 远端挂载

从<http://fuse.sourceforge.net/sshfs.html>下载sshfs，它允许你跨网络安全挂载一个目录。

```
sshfs name@server:/path/to/folder /path/to/mount/point
```



