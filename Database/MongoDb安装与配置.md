

# 安装与启动

## 安装

- 下载mongodb的版本，两点注意
  1. 根据业界规则，偶数为稳定版，如3.2.X；奇数为开发版，如3.3.X
  2. 32bit的mongodb最大只能存放2G的数据，64bit就没有限制
- MongoDB官网安装包下载地址：<http://www.mongodb.org/downloads>
- MongoDB安装文档：<https://docs.mongodb.com/getting-started/shell/installation/>

> Ubuntu：

```
sudo apt-get install mongodb
```

> mac安装:

```
# 预先安装xcode
brew install MongDB

# 后台及登录启动
brew services start mongodb

# 临时启动
redis-server /usr/local/etc/mongodb.conf
```

## 客户端

- 客户端命令为 **mongo**，可以通过help查看所有参数。
- 这个shell即是mongodb的客户端，用来对MongoDB进行操作和管理的交互式环境。

```
python@ubuntu:~$ mongo --help
```

- 终端退出连接

```
> exit

(或Ctrl+C)
```

## 服务端

MongoDB 默认的存储数据目录为 /data/db，默认端口27017

- 服务的命令为mongod，可以通过help查看所有参数

```
python@ubuntu:~$ mongod --help
```

- 相关文件存放路径：默认各个文件存放路径如下所示：

> - 可执行文件存放路径：`/usr/bin/mongod` 和 `/usr/bin/mongo`
> - 数据库文件存放路径：`/data/db`
> - 日志文件存放路径：`/var/log/mongodb/mongod.log`
> - 配置文件存放路径：`/etc/mongod.conf`

- 启动注意事项

1. 首次启动

出错：表示默认的存储数据目录 /data/db 不存在：

`[initandlisten] exception in initAndListen: 29 Data directory /data/db not found., terminating`

创建 /data目录和 /data/db 目录，并指定 读/写/执行 权限

```
python@ubuntu:~$ sudo mkdir -p /data/db
python@ubuntu:~$ sudo chmod 777 /data/db

```

2. 再次启动

启动成功，但是可能会有如下警告：

```
#### 此乃 Warning 1：
[initandlisten] ** WARNING: /sys/kernel/mm/transparent_hugepage/enabled is 'always'.
[initandlisten] **        We suggest setting it to 'never'
[initandlisten] 
[initandlisten] ** WARNING: /sys/kernel/mm/transparent_hugepage/defrag is 'always'.
[initandlisten] **        We suggest setting it to 'never'

#### 此乃 Warning 2：
[initandlisten] ** WARNING: soft rlimits too low. rlimits set to 1024 processes, 64000 files. Number of processes should be at least 32000 : 0.5 times number of files.

#### 此乃 Warning 3：
[initandlisten] ** WARNING: You are running this process as the root user, which is not recommended.

```

**warning1**:

Linux的内存分配默认由内核动态分配，而不是由程序自行管理。而MongoDB对内存占用有那么点...严重，所以为了防止MongoDB占用内存过大而被内核"管理"，官方推荐关闭动态分配。

默认"always"表示允许动态分配，对应的"never"就是不允许，所以我们将这两个文件内容修改为"naver"后就没有warning了。

```
# Ctrl + c 退出 MongoDB 数据库服务
# 然后进入 root 用户下，执行修改命令

python@ubuntu:~$ sudo su
[sudo] python 的密码： 

root@ubuntu:~# sudo echo "never" > /sys/kernel/mm/transparent_hugepage/enabled
root@ubuntu:~# sudo echo "never" >  /sys/kernel/mm/transparent_hugepage/defrag

```

实际上，除非网站DBA对数据库性能有极限要求，在通常情况下系统动态分配的内存页大小足够我们正常使用，而且更能优化整个系统，所以一般不必理会这个warning。而且这样只是临时修改Linux内核的设置，在Linux服务器重启后则会失效。

**waring2**

这个WARNING（如果有的话）含义为： 表示默认分配给MongoDB的进程和文件数量限制过低，需要重新分配值：

- mongodb当前限制：1024 processes, 64000 files
- mongodb建议要求：processes = 0.5*files=32000（至少）

```
# 打开 相关配置文件：
root@ubuntu:~# vi /etc/security/limits.conf

# 在打开的 文件最下方，添加，然后保存退出
mongod  soft  nofile  64000
mongod  hard  nofile  64000
mongod  soft  nproc  32000
mongod  hard  nproc  32000

```

**warining3**

意思是我们在用root权限做这些事，理论上是不安全的。我们可以通过附加`--auth`参数，来使用用户认证来处理这个情况，这个后面会讲到。

### 三种启动方式：

- 命令行方式直接启动

MongoDB默认的存储数据目录为`/data/db`（需要事先创建），默认端口27017，也可以修改成不同目录：

```
# 直接启动mongod，默认数据存储目在 /data/db
python@ubuntu:~$ sudo mongod

# 启动mongod，并指定数据存储目录（目录必须存在，且有读写权限）
python@ubuntu:~$ sudo mongod --dbpath=/xxxxx/xxxxx

```

- 配置文件方式启动

启动时加上`-f`参数，并指向配置文件即可，默认配置文件为`/etc/mongodb.conf`，也可以自行编写配置文件并指定。

```
# 启动mongod，并按指定配置文件执行
python@ubuntu:~$ sudo mongod -f /etc/mongodb.conf
```

- 守护进程方式启动

**启动**

MongoDB提供了一种后台程序方式启动的选择，只需要加上—fork参数即可。但是注意：如果用到了`--fork`参数，就必须启用`--logpath`参数来指定log文件，这是强制的。

```
python@ubuntu:~$ sudo mongod --logpath=/data/db/mongodb.log --fork

about to fork child process, waiting until server is ready for connections.
forked process: xxxxx
child process started successfully, parent exiting

```

**关闭 **

如果使用`--fork`在后台运行mongdb服务，那么就要通过本机admin数据库向服务器发送shutdownServer()消息来关闭。

```
python@ubuntu:~$ mongo
MongoDB shell version: 3.2.8
connecting to: test

> use admin
switched to db admin

> db.shutdownServer()
server should be down...
2017-05-16T22:34:22.923+0800 I NETWORK  [thread1] trying reconnect to 127.0.0.1:27017 (127.0.0.1) failed
2017-05-16T22:34:22.923+0800 W NETWORK  [thread1] Failed to connect to 127.0.0.1:27017, reason: errno:111 Connection refused
2017-05-16T22:34:22.923+0800 I NETWORK  [thread1] reconnect 127.0.0.1:27017 (127.0.0.1) failed failed 

```

- 启用用户认证方式启动

如果之前未定义过用户，所以mongod将允许本地直接访问操作数据库将使用本地root权限，如果使用`--auth`参数启动，将启用MongoDB授权认证，即启用不同的用户对不同的数据库的操作权限。

> 也可以在配置文件`mongod.conf`中加入`auth = true`按第二种启动方式启动。

[参考阅读](http://www.mongoing.com/docs/tutorial/enable-authentication.html)

```
# 启动mongod，并启用用户认证
python@ubuntu:~$ sudo mongod --auth

# 启动mongo shell
python@ubuntu:~$ mongo

# 1. 切换admin数据库下
> use admin

# 2. 创建一个拥有root权限的超级用户，拥有所有数据库的所有权限
#      用户名：python，密码：chuanzhi，角色权限：root（最高权限）
> db.createUser({user : "python", pwd : "chuanzhi", roles : ["root"]})

# 3. 如果 MongoDB 开启了权限模式，并且某一个数据库没有任何用户时，可以不用认证权限并创建一个用户，但是当继续创建第二个用户时，会返回错误，若想继续创建用户则必须认证登录。
> db.createUser({user : "bigcat", pwd : "bigcat", roles : [{role : "read", db : "db_01"}, {role : "readWrite", db : "db_02"}]})

# 4. 认证登录到python用户（第一次创建的用户）
> db.auth("python","chuanzhi")

# 5. 查看当前认证登录的用户信息
> show users

# 6. 认证登录成功，可以继续创建第二个用户
# 用户名：bigcat，密码：bigcat，角色权限：[对db_01 拥有读权限，对db_02拥有读/写权限]
> db.createUser({user : "bigcat", pwd : "bigcat", roles : [{role : "read", db : "db_01"}, {role : "readWrite", db : "db_02"}]})


# 7. 查看当前数据库下所有的用户信息.
> db.system.users.find()

# 8. 认证登录到 bigcat 用户
> db.auth("bigcat", "bigcat")

# 9. 切换到 数据库db_01，读操作没有问题
> use db_01

> show collections

# 10. 切换到 数据库db_02，读操作没有问题
> use db_02

> show collections

# 11. 切换到 数据库db_03，读操作出现错误，bigcat用户在db_03数据库下没有相关权限
> use db_03

> show collections
2017-05-17T00:26:56.143+0800 E QUERY    [thread1] Error: listCollections failed: {
    "ok" : 0,
    "errmsg" : "not authorized on db_03 to execute command { listCollections: 1.0, filter: {} }",
    "code" : 13
} :
_getErrorWithCode@src/mongo/shell/utils.js:25:13
DB.prototype._getCollectionInfosCommand@src/mongo/shell/db.js:773:1
DB.prototype.getCollectionInfos@src/mongo/shell/db.js:785:19
DB.prototype.getCollectionNames@src/mongo/shell/db.js:796:16
shellHelper.show@src/mongo/shell/utils.js:754:9
shellHelper@src/mongo/shell/utils.js:651:15
@(shellhelp2):1:1

>
# 12. 认证登录到python用户下
> db.auth("python", "chuanzhi")
1
>
# 13. 删除bigcat用户
> db.dropUser("bigcat")
true
>
# 14. 尝试认证登录bigcat失败
> db.auth("bigcat", "bigcat")
Error: Authentication failed.
0
>
# 15. 退出mongo shell
> exit
bye
python@ubuntu:~$

```

### mongod部分参数说明

在源代码中，mongod的参数分为一般参数，windows参数，replication参数，replica set参数以及隐含参数。下面列举的是一般参数。

mongod的参数中，没有设置内存大小的相关参数，因为MongoDB使用os mmap机制来缓存数据文件数据，自身目前不提供缓存机制。mmap在数据量不超过内存时效率很高，但是数据超过内存后，写入的性能不太稳定。

```
dbpath：数据文件存放路径。每个数据库会在其中创建一个子目录，防止同一个实例多次运行的mongod.lock也保存在次目录中。

logpath：错误日志文件

auth：用户认证

logappend：错误日志采用追加模式(默认覆写模式)

bind_ip：对外服务的绑定ip，一般设置为空，及绑定在本机所有可用ip上。如有需要可以单独绑定。

port：对外服务端口。Web管理端口在这个port的基础上+1000。

fork：以后台Daemon形式运行服务。

journal：开启日志功能，通过保存操作日志来降低单机故障的恢复时间。

syncdelay：系统同步刷新磁盘的时间，单位为秒，默认时60秒。

directoryperdb：每个db存放在单独的目录中，建议设置该参数。

repairpath：执行repair时的临时目录。如果没有开启journal，异常down机后重启，必须执行repair操作。

```

### MongoDB 统计信息

要获得关于MongoDB的服务器统计，需要在MongoDB客户端键入命令`db.stats()`。这将显示数据库名称，收集和数据库中的文档信息

输出信息的参数如下：

```
"db" : "test" ,表示当前是针对"test"这个数据库的描述。想要查看其他数据库，可以先运行$ use datbasename
"collections" : 3,表示当前数据库有多少个collections.可以通过运行show collections查看当前数据库具体有哪些collection.
"objects" : 267,表示当前数据库所有collection总共有多少行数据。显示的数据是一个估计值，并不是非常精确。
"avgObjSize" : 623.2322097378277,表示每行数据是大小，也是估计值，单位是bytes
"dataSize" : 16640,表示当前数据库所有数据的总大小，不是指占有磁盘大小。单位是bytes
"storageSize" : 110592,表示当前数据库占有磁盘大小，单位是bytes,因为mongodb有预分配空间机制，为了防止当有大量数据插入时对磁盘的压力,因此会事先多分配磁盘空间。
"numExtents" : 0,没有什么真实意义
"indexes" : 2 ,表示system.indexes表数据行数。
"indexSize" : 53248,表示索引占有磁盘大小。单位是bytes
"ok" : 1,表示服务器正常
```

# 集群部署

