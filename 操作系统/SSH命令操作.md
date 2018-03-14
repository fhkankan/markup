#登录访问

**密码登录**

```
# 在客户端登录到远端计算机
ssh user_name@远端计算机IP
# 输入密码
*****
```

**无密码登录**

```
# 首先使用生成密钥
ssh-keygen -t rsa

# 将id_rsa.pub中的内容复制到远端计算机的.ssh/authorized_keys文件中，

# 无密码访问远端计算机了
ssh-copy-id user@host
```

# 其他连接

**通过中间主机建立SSH连接**

Unreachable_host表示从本地网络无法直接访问的主机，但可以从reachable_host所在网络访问，这个命令通过到reachable_host的“隐藏”连接，创建起到unreachable_host的连接。

```
ssh -t reachable_host ssh unreachable_host
```

**直接连接到只能通过主机B连接的主机A**

```
ssh -t hostA ssh hostB
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
autossh -M50000 -t server.example.com ‘screen -raAd mysession’

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

**端口隧道**

```
# 从某主机的80端口开启到本地主机2001端口的隧道
ssh -N -L2001:localhost:80 somemachine

现在你可以直接在浏览器中输入http://localhost:2001访问这个网站。
```

**声音传输**

```
# 将你的麦克风输出到远程计算机的扬声器
dd if=/dev/dsp | ssh -c arcfour -C username@host dd of=/dev/dsp

这样来自你麦克风端口的声音将在SSH目标计算机的扬声器端口输出，但遗憾的是，声音质量很差，你会听到很多嘶嘶声。
```

# 检测分析

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



#文件处理

##对比

**比较远程和本地文件 **

```
ssh user@host cat /path/to/remotefile | diff /path/to/localfile –
```

##复制

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

**上传下载 **

```
# 上传文件到远端计算机命令：

put 本地文件 远端计算机目录

# 下载文件到本地计算机命令：

get 远端文件 本地目录
```

**复制**

```
# 同cp命令一样，复制目录时可以加上-r选项。
scp 文件 账户@远端计算机IP:目录名

scp 账户 @远端计算机IP:文件 本地目录

# 继续SCP大文件
rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file
它可以恢复失败的rsync命令，当你通过VPN传输大文件，如备份的数据库时这个命令非常有用，需要在两边的主机上安装rsync。

rsync –partial –progress –rsh=ssh $file_source $user@$host:$destination_file local -> remote
或

rsync –partial –progress –rsh=ssh $user@$host:$remote_file $destination_file remote -> local
```

**将标准输入（stdin）复制到你的X11缓冲区**

```
ssh user@host cat /path/to/some/file | xclip

你是否使用scp将文件复制到工作用电脑上，以便复制其内容到电子邮件中？xclip可以帮到你，它可以将标准输入复制到X11缓冲区，你需要做的就是点击鼠标中键粘贴缓冲区中的内容。
```

## 删除

```
# 删除文本文件中的一行内容，有用的修复
在这种情况下，最好使用专业的工具。
ssh-keygen -R <the_offending_host>

# 删除文本文件中的一行，修复“SSH主机密钥更改”的警告
sed -i 8d ~/.ssh/known_hosts


```

#远端挂载

从<http://fuse.sourceforge.net/sshfs.html>下载sshfs，它允许你跨网络安全挂载一个目录。

```
sshfs name@server:/path/to/folder /path/to/mount/point
```



