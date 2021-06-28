# paramiko

[官网](http://www.paramiko.org/index.html)

Paramiko 是 SSHv2 协议 [1] 的 Python (2.7, 3.4) 实现，提供客户端和服务器功能。虽然它利用 Python C 扩展进行低级加密（密码学），但 Paramiko 本身是一个围绕 SSH 网络概念的纯 Python 接口。

- 安装

```
pip install paramiko
```

## 核心

paramiko包含两个核心组件：SSHClient和SFTPClient。

- SSHClient的作用类似于Linux的ssh命令，是对SSH会话的封装，该类封装了传输(Transport)，通道(Channel)及SFTPClient建立的方法(open_sftp)，通常用于执行远程命令。
- SFTPClient的作用类似与Linux的sftp命令，是对SFTP客户端的封装，用以实现远程文件操作，如文件上传、下载、修改文件权限等操作。

基础名词

```
1、Channel：是一种类Socket，一种安全的SSH传输通道；
2、Transport：是一种加密的会话，使用时会同步创建了一个加密的Tunnels(通道)，这个Tunnels叫做Channel；
3、Session：是client与Server保持连接的对象，用connect()/start_client()/start_server()开始会话。
```

## 使用

### SSHClient

- 常用方法

`connect()`：实现远程服务器的连接与认证，对于该方法只有hostname是必传参数。

```
hostname 连接的目标主机
port=SSH_PORT 指定端口
username=None 验证的用户名
password=None 验证的用户密码
pkey=None 私钥方式用于身份验证
key_filename=None 一个文件名或文件列表，指定私钥文件
timeout=None 可选的tcp连接超时时间
allow_agent=True, 是否允许连接到ssh代理，默认为True 允许
look_for_keys=True 是否在~/.ssh中搜索私钥文件，默认为True 允许
compress=False, 是否打开压缩
```

`set_missing_host_key_policy()`：设置远程服务器没有在know_hosts文件中记录时的应对策略。目前支持三种策略：

```python
# 设置连接的远程主机没有本地主机密钥或HostKeys对象时的策略，目前支持三种：
 
AutoAddPolicy # 自动添加主机名及主机密钥到本地HostKeys对象，不依赖load_system_host_key的配置。即新建立ssh连接时不需要再输入yes或no进行确认
WarningPolicy # 用于记录一个未知的主机密钥的python警告。并接受，功能上和AutoAddPolicy类似，但是会提示是新连接
RejectPolicy # 自动拒绝未知的主机名和密钥，依赖load_system_host_key的配置。此为默认选项
```

`exec_command()`：在远程服务器执行Linux命令的方法。

`open_sftp()`：在当前ssh会话的基础上创建一个sftp会话。该方法会返回一个SFTPClient对象。

```python
# 利用SSHClient对象的open_sftp()方法，可以直接返回一个基于当前连接的sftp对象，可以进行文件的上传等操作.
 
sftp = client.open_sftp()
sftp.put('test.txt','text.txt')
```

- 使用示例

连接方式

```python
#实例化SSHClient
client = paramiko.SSHClient()
 
#自动添加策略，保存服务器的主机名和密钥信息，如果不添加，那么不再本地know_hosts文件中记录的主机将无法连接
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# 方法一：密钥连接
private = paramiko.RSAKey.from_private_key_file('/Users/ch/.ssh/id_rsa')  # 配置私人密钥文件位置
client.connect(hostname='10.0.0.1',port=22,username='root',pkey=private) 

# 方法二：账号连接
client.connect(hostname='192.168.1.105', port=22, username='root', password='123456')  # 连接SSH服务端，以用户名和密码进行认证
```

SSHClient封装Transport

```python
import paramiko
 
# 创建一个通道
transport = paramiko.Transport(('hostname', 22))
transport.connect(username='root', password='123')
 
ssh = paramiko.SSHClient()
ssh._transport = transport
 
stdin, stdout, stderr = ssh.exec_command('df -h')
print(stdout.read().decode('utf-8'))
 
transport.close()
```

远程执行命令，获取结果/结果码

```python
import paramiko
 
# 实例化SSHClient
client = paramiko.SSHClient()
 
# 自动添加策略，保存服务器的主机名和密钥信息，如果不添加，那么不再本地know_hosts文件中记录的主机将无法连接
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 
# 连接SSH服务端，以用户名和密码进行认证
client.connect(hostname='192.168.1.105', port=22, username='root', password='123456')
 
# 打开一个Channel并执行命令
stdin, stdout, stderr = client.exec_command('df -h ')  # stdout 为正确输出，stderr为错误输出，同时是有1个变量有值
 
# 打印执行结果
print(stdout.read().decode('utf-8'))

# 获取执行命令的结果码
channel = stdout.channel
ret = channel.recv_exit_status()
if ret == 0:
  print(stdout.read().decode("utf-8"))
else:
  print(stderr.read().decode("utf-8"))
 
# 关闭SSHClient
client.close()
```

远程执行命令，生成子进行并交互执行命令

```python
import paramiko,time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname='192.168.0.1', port=22, username='root', password='Aa12345!')
interact = conn.invoke_shell()
stdin, stdout, stderr = ssh.exec_command('df -k')
interact = conn.invoke_shell()
interact.send("sed -i '/192.168.0.2/d' /root/.ssh/known_hosts" + '\n')
time.sleep(1)
interact.send('ssh root@172.16.128.2' + '\n')
time.sleep(2)
interact.send('yes' + '\n')
time.sleep(2)
interact.send('Aa12345!' + '\n')
time.sleep(2)
interact.send('df -k!' + '\n')
result = interact.recv(65535)
print result
```

### SFTPClient

- 常用方法

```python
SFTPCLient作为一个sftp的客户端对象，根据ssh传输协议的sftp会话，实现远程文件操作，如上传、下载、权限、状态
 
from_transport(cls,t)  # 创建一个已连通的SFTP客户端通道
put(localpath, remotepath, callback=None, confirm=True)  # 将本地文件上传到服务器 参数confirm：是否调用stat()方法检查文件状态，返回ls -l的结果
get(remotepath, localpath, callback=None)  # 从服务器下载文件到本地
mkdir() # 在服务器上创建目录
remove() # 在服务器上删除目录
rename() # 在服务器上重命名目录
stat()  # 查看服务器文件状态
listdir()  # 列出服务器目录下的文件
```

- 使用

密钥连接方式

```python
# 配置私人密钥文件位置
private = paramiko.RSAKey.from_private_key_file('/Users/ch/.ssh/id_rsa')
 
#实例化SSHClient
client = paramiko.SSHClient()
 
#自动添加策略，保存服务器的主机名和密钥信息，如果不添加，那么不再本地know_hosts文件中记录的主机将无法连接
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 
#连接SSH服务端，以用户名和密码进行认证
client.connect(hostname='10.0.0.1',port=22,username='root',pkey=private)
```

SSHClient封装Transport

```python
import paramiko
 
# 创建一个通道
transport = paramiko.Transport(('hostname', 22))
transport.connect(username='root', password='123')
 
ssh = paramiko.SSHClient()
ssh._transport = transport
 
stdin, stdout, stderr = ssh.exec_command('df -h')
print(stdout.read().decode('utf-8'))
 
transport.close()
```

上传/下载文件

```python
import paramiko
 
# 获取Transport实例
tran = paramiko.Transport(('10.0.0.3', 22))
 
# 连接SSH服务端，使用password
tran.connect(username="root", password='123456')
# 或使用
# 配置私人密钥文件位置
private = paramiko.RSAKey.from_private_key_file('/Users/root/.ssh/id_rsa')
# 连接SSH服务端，使用pkey指定私钥
tran.connect(username="root", pkey=private)
 
# 获取SFTP实例
sftp = paramiko.SFTPClient.from_transport(tran)
 
# 设置上传的本地/远程文件路径
localpath = "/Users/root/Downloads/1.txt"
remotepath = "/tmp/1.txt"
 
# 执行上传动作
sftp.put(localpath, remotepath)
# 执行下载动作
sftp.get(remotepath, localpath)
 
tran.close()
```

### 综合类

```python
import paramiko
import logging
import time
import os

file_path1 = "aaaaaaaaaaa"
file_path2 = "aaaaaaaaaa"

target_path1 = "xxxxxxxxxxxx"
target_path2 = "ccccccccccccccccc"

file_list = [xxxxxxxxxxxxxx]


class SSHConnection:
    def __init__(self, host="xxxxxxxx", port=22, username="xxx", password="xxxxxxxxx"):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def connect(self):
        transport = paramiko.Transport((self.host, self.port))
        transport.connect(username=self.username, password=self.password)
        self.__transport = transport

    def close(self):
        self.__transport.close()
        
    def run_cmd(self, command):
        """
        执行shell命令,返回字典
        return {'color': 'red','res':error}或
        return {'color': 'green', 'res':res}
        """
        ssh = paramiko.SSHClient()
        # 允许连接不在know_hosts文件中的主机
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh._transport = self.__transport
        # ssh.connect(hostname=self.host,port=self.port,username=self.username,password=self.password)
        # 执行命令
        stdin, stdout, stderr = ssh.exec_command(command)
        # 获取命令结果
        res = SSHConnection.to_str(stdout.read())
        # 获取错误信息
        error = SSHConnection.to_str(stderr.read())
        # 如果有错误信息，返回error
        # 否则返回res
        if error.strip():
            return {'color':'red','res':error}
        else:
            return {'color': 'green', 'res':res}    

    def upload(self,local_path, target_path):
        # 连接，上传
        sftp = paramiko.SFTPClient.from_transport(self.__transport)
        # 将location.py 上传至服务器 /tmp/test.py
        sftp.put(local_path, target_path, confirm=True)
        # print(os.stat(local_path).st_mode)
        # 增加权限
        # sftp.chmod(target_path, os.stat(local_path).st_mode)
        sftp.chmod(target_path, 0o755)  # 注意这里的权限是八进制的，八进制需要使用0o作为前缀
 
    def download(self,target_path, local_path):
        # 连接，下载
        sftp = paramiko.SFTPClient.from_transport(self.__transport)
        # 将location.py 下载至服务器 /tmp/test.py
        sftp.get(target_path, local_path)
 
    # 销毁
    def __del__(self):
        self.close()
       
   	@staticmethod
    def to_str(bytes_or_str):
    	"""
    	把byte类型转换为str
    	:param bytes_or_str:
    	:return:
    	"""
    	if isinstance(bytes_or_str, bytes):
    	    value = bytes_or_str.decode('utf-8')
    	else:
    	    value = bytes_or_str
    	return value

```

