[TOC]

# Docker安装使用

## 概述

Docker是一个开源的容器引擎，它基于LXC容器技术，使用Go语言开发。
源代码托管在Github上，并遵从Apache2.0协议。
Docker采用C/S架构，其可以轻松的为任何应用创建一个轻量级的、可移植的、自给自足的容器。
简单来说:Docker就是一种快速解决生产问题的一种技术手段。

优点

```
多:  适用场景多
快:  环境部署快、更新快
好:  好多人在用，东西好
省:  省钱省力省人工
```

缺点

```
依赖操作系统
依赖网络
银行U盾等场景不能用
```

## 安装

> ubuntu

https://docs.docker.com/install/linux/docker-ce/ubuntu/#set-up-the-repository

检查宿主机环境

```
$ uname -a
$ ls -l /sys/class/misc/device-mapper
```

删除旧版本

```
$ sudo apt-get remove docker docker-engine docker.io
```

安装

```
# 安装基本软件
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
# 使用官方源
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
# 或者使用阿里云的源
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
# 安装docker
$ sudo apt-get update
$ sudo apt-get install docker-ce
# 安装指定版本
$ sudo apt-get install docker-ce=<VERSION>
# 查看支持的docker版本
apt-cache madison docker-ce
# 测试安装是否ok
$ sudo docker run hello-world
```

配置加速器

```
# 访问daocloud.io,登录daocloud账户
# 点击右上角"加速器"
# 复制“配置Docker加速器”中Linux的内容并在终端执行
curl -sSL https://get.daocloud.io/daotools/set_mirror.sh | sh -s http://74f21445.m.daocloud.io
# 修改daemon.json文件，增加如下内容
{"registry-mirrors": ["http://74f21445.m.daocloud.io"], "insecure-registries": []}
# 重启docker
systemctl restart docker
```

删除docker

```
apt-get purge docker-ce -y
# 删除docker的应用目录
rm -rf /var/lib/docker
# 删除docker的认证目录
rm -rf /etc/docker
```

修改docker镜像源

```
# 1.打开配置文件
sudo vim /etc/docker/daemon.json
# 2.添加信息
{"registry-mirrors" : ["https://docker.mirrors.ustc.edu.cn"]}
# 3.重启docker
sudo service docker restart
```

> mac/windows

安装Docker桌面系统

修改docker镜像源

```
# 1.打开配置文件
vim ~/.ssh/daemon.json
# 2.添加信息
{"registry-mirrors" : ["https://docker.mirrors.ustc.edu.cn"]}
# 3.重启docker desktop
```

## 使用

### 基本命令

```
systemctl [参数] docker
参数详解
start		开启服务
stop		关闭
restart		重启
status		状态
```

登陆

```
docker login --username=xxxx
```

### 镜像管理

镜像是一个只读文件，是一个能被docker运行起来的一个程序。通过运行这个程序完成各种应用的部署。

> 搜索、查看、获取

```
# 搜索镜像
docker search [image_name]
# 获取镜像
docker pull [image_name]  # 下载的镜像在/var/lib/docker目录下
# 查看镜像
docker images <image_name>
docker images -a  # 列出所有本地的images(包括已删除的镜像记录)
```

> 重命名、删除

```
# 重命名
docker tag [old_image]:[old_version] [new_image]:[new_version]
# 删除
docker rmi [image_id/image_name:image_version]
```

> 导入、导出

```
# 将已经下载好的镜像，导出到本地，以备后用
docker save -o [导出后本地的镜像名称] [源镜像名称]
# 导入镜像(两种方式)
docker load < [image.tar.gz]
docker load --input [image.tar.gz]
```

> 历史、创建

```
# 查看历史
docker history [image_name]
# 根据模板创建镜像
cat 模板文件名 | docker import - [自定义镜像名]
# 基于容器创建镜像
docker commit -m '改动信息' -a "作者信息" [container_id] [new_image:tag] # 方式一
docker export ae63ab299a84 > gaoji.tar  # 方式二
```

### 容器管理

容器类似一个操作系统，这个操作系统启动了某些服务，这里的容器指运行起来的一个Docker镜像

> 查看

```
# 查看容器详情
docker ps [OPTIONS]

OPTIONS说明：
-a :显示所有的容器，包括未运行的。
-f :根据条件过滤显示的内容。
--format :指定返回值的模板文件。
-l :显示最近创建的容器。
-n :列出最近创建的n个容器。
--no-trunc :不截断输出。
-q :静默模式，只显示容器编号。
-s :显示总的文件大小。

# 查看容器ip
docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_id
```


>启动

基于镜像创建新容器并启动

```
docker run <参数，可选> [docker_image] [执行的命令]
eg:docker run nginx /bin/echo "hello docker"
参数选项：
-a stdin: 指定标准输入输出内容类型，可选 STDIN/STDOUT/STDERR 三项；
-d: 后台运行容器，并返回容器ID；
-i: 以交互模式运行容器，通常与 -t 同时使用；
-p: 端口映射，格式为：主机(宿主)端口:容器端口
-v: 挂载一个数据卷，格式为：宿主机文件:容器文件
-t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用；
-m :设置容器使用内存最大值；
-h "mars": 指定容器的hostname；
-e username="ritchie": 设置环境变量；
--name="nginx-lb": 为容器指定一个名称；
--dns 8.8.8.8: 指定容器使用的DNS服务器，默认和宿主一致；
--dns-search example.com: 指定容器DNS搜索域名，默认和宿主一致；
--env-file=[]: 从指定文件读入环境变量；
--cpuset="0-2" or --cpuset="0,1,2": 绑定容器到指定CPU运行；
--net="bridge": 指定容器的网络连接类型，支持 bridge/host/none/container: 四种类型；
--link=[]: 添加链接到另一个容器；
--expose=[]: 开放一个端口或一组端口；
```
启动已终止的容器
```
docker start [container_id]
```
在后台以守护进程方式启动
```
docker run -d [image_name] command ...
```

> 关闭、删除

```
# 关闭容器
docker stop [container_id]

# 删除容器
# 正常删除(删除已关闭的)
docker rm [container_id]
# 强制删除(删除正在运行的)
docker rm -f [container_id]
# 批量删除相关容器
docker rm -f $(docker ps -a -q)
```

> 进入、退出

进入

```
# 方式一：创建并进入容器
docker run --name [container_name] -it [docker_image] /bin/bash
# 参数
--name:给容器定义一个名称
-i:则让容器的标准输入保持打开。
-t:让docker分配一个伪终端,并绑定到容器的标准输入上
/bin/bash:执行一个命令

# 方式二：手工方式进入
docker exec -it container_id /bin/bash

# 方式三：生产中进入容器,会用脚本,
# 脚本docker_in.sh内容
#!/bin/bash
# 定义进入仓库函数
docker_in(){
	NAME_ID=$1
	PID=$(docker inspect -f "{{ .State.Pid }}" $NAME_ID)
	nsenter -t $PID -m -u -i -n -p
}
docker_in $1
# 赋权执行
chmod +x docker_in.sh
# 进入指定容器，并测试
./docker_in.sh [container_id]
```

退出

```
# 方式一：
exit
# 方式二：
Ctrl + D
```

> 基于容器创建镜像

```
# 方式一
docker commit -m '改动信息' -a "作者信息" [container_id] [new_image:tag]

# 方式二
docker export [容器id] > 模板文件名.tar
```

> 日志、信息

```
# 查看容器运行日志
docker logs [container_id]
# 查看容器详细信息
docker inspect [container_id]  # 查看全部信息
 docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' [container_id]	# 查看容器网络信息
# 查看容器端口信息
docker port [container_id]
```

> 迁移

备份

```
docker ps -a  # 查看容器
docker commit -p 30b8f18f20b4 container-backup  # 创建快照
docker images  # 查看镜像
# 方式一：云备份
docker login 172.16.101.192  # 登录Docker注册中心，
docker tag a25ddfec4d2a arunpyasi/container-backup:test # 打标签
docker push arunpyasi/container-backup  # 推送镜像
# 方式二：本地备份
docker save -o ~/container-backup.tar container-backup  # 本地保存镜像
```

恢复

```
# 方式一：云拉取
docker pull arunpyasi/container-backup:test  # 拉取镜像
# 方式二：本地加载
docker load -i ~/container-backup.tar  # 本地加载
docker images  # 查看镜像
docker run -d -p 80:80 container-backup # 运行镜像
```

### 仓库管理

仓库指的是Docker镜像存储的地方。
Docker的仓库有两大类：
公有仓库： Docker hub、 Docker cloud、 等
私有仓库： registry、 harbor、 等

> 仓库相关命令

```
# 登录仓库
docker login [仓库名称]
# 拉取镜像
docker pull [镜像名称]
# 推送镜像
docker push [镜像名称]
# 查找镜像
docker search [镜像名称]
```

> 私有仓库部署

创建流程

```
1、 根据registry镜像创建容器
2、 配置仓库权限
3、 提交镜像到私有仓库
4、 测试
```

实施过程

```bash
# 下载registry镜像
docker pull registry
# 启动仓库容器
docker run -d -p 5000:5000 registry
# 检查容器效果
curl 127.0.0.1:5000/v2/_catalog
# 配置容器权限
vim /etc/docker/daemon.json
{"registry-mirrors": ["http://74f21445.m.daocloud.io"], "insecure-registries": ["192.168.8.14:5000"]}
# 注意：私有仓库的ip地址是宿主机的ip， 而且ip两侧有双引号
# 重启docker服务
systemctl restart docker
systemctl status docker
# 启动容器
docker start 315b5422c699
# 标记镜像
docker tag ubuntu-mini 192.168.8.14:5000/ubuntu-14.04-mini
# 提交镜像
docker push 192.168.8.14:5000/ubuntu-14.04-mini
# 下载镜像
docker pull 192.168.8.14:5000/ubuntu-14.04-mini
```

### 数据管理

docker的镜像是只读的， 虽然依据镜像创建的容器可以进行操作， 但是我们不能将数据保存到容器中， 因为容器会随时关闭和开启， 而是使用数据卷和数据卷容器保存数据

数据卷就是将宿主机的某个目录， 映射到容器中， 作为数据存储的目录， 我们就可以在宿主机对数据进行存储

命令

```
docker run --help
...
-v, --volume list
# 挂载一个数据卷，默认为空
```

使用命令 docker run 用来创建容器， 可以在使用docker run 命令时添加 -v 参数， 就可以创建并挂载**一个到多个**数据卷到当前运行的容器中。
-v 参数的作用是将宿主机的一个目录作为容器的数据卷挂载到docker容器中， 使宿主机和容器之间可以共享一个目录， 如果本地路径不存在， Docker也会自动创建。
-v 宿主机文件:容器文件

> 目录

```bash
docker run -itd --name [容器名字] -v [宿主机目录]:[容器目录] [镜像名称] [命令(可选)]
```

eg

```bash
# 创建测试文件
echo "file1" > /tmp/file1.txt
# 启动一个容器，挂载数据卷
docker run -itd --name test1 -v /tmp:/test1 nginx
# 测试效果
~# docker exec -it a53c61c77 /bin/bash
root@a53c61c77bde:/# cat /test1/file1.txt
file1
```

> 文件

```bash
docker run -itd --name [容器名字] -v [宿主机文件]:[容器文件] [镜像名称] [命令(可选)]
```

eg

```bash
# 创建测试文件
echo "file1" > /tmp/file1.txt
# 启动一个容器，挂载数据卷
docker run -itd --name test2 -v /tmp/file1.txt:/nihao/nihao.sh nginx
# 测试效果
~# docker exec -it 84c37743 /bin/bash
root@84c37743d339:/# cat /nihao/nihao.sh
file1
```

### 网络管理

> 端口映射

默认情况下， 容器和宿主机之间网络是隔离的， 我们可以通过端口映射的方式， 将容器中的端口， 映射到宿主机的某个端口上。这样我们就可以通过宿主机的ip+port的方式来访问容器里的内容

有两种方式：

```
随机映射	-P(大写)
指定映射	-p 宿主机端口:容器端口
```

随机映射

```bash
# 默认随机映射
docker run -d -P [镜像名称]

# -P自动绑定所有对外提供服务的容器端口映射的端口将会从没有使用的端口池中自动随机选择，但是如果连续启动多个容器的话，则下一个容器的端口默认是当前容器占用端口号+1
```

eg

```
# 启动一个nginx镜像
docker run -d -P nginx
# 网络访问
http://docker容器宿主机的ip:容器映射的端口
```

指定映射

```bash
# tcp协议
# 若不指定宿主机ip，则默认0.0.0.0
docker run -d -p [宿主机ip]:[宿主机端口]:[容器端口] --name [容器名字] [镜像名称]

# udp协议
# 一般使用在dns业务
docker run -d -p [宿主机ip]:[宿主机端口]:[容器端口]/udp --name [容器名字] [镜像名称]
```

eg

```bash
# 在启动容器的时候， 给容器指定一个访问的端口 1199
docker run -d -p 192.168.8.14:1199:80 --name nginx-1 nginx

# 查看新容器ip
docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 0ad3acfbfb76

# 浏览器访问
http://192.168.8.14:1199/
```

> 网络管理命令

查看网络命令帮助

```
docker network help
```

查看当前主机网络

```
docker network ls
```

查看bridge网络信息

```
docker network inspect container_id
```

> 网络模式

bridge

```
默认模式
它会在docker容器启动时候，自动配置好自己的网络信息，同一宿主机的所有容器都在一个网络下，彼此间可以通信。 类似于我们vmware虚拟机的nat模式

宿主机ip:198.x.x.x
容器ip:172.17.0.x
```

host

```
容器使用宿主机的ip地址进行通信

宿主机ip:198.x.x.x
容器ip:
```

container

```
新创建的容器间使用使用已创建的容器网络， 类似一个局域网。

宿主机ip:198.x.x.x
容器ip:172.x.x.x
容器ip:172.x.x.x
```
overlay

```
容器彼此不在同一网络， 而且能互相通行

宿主机ip:198.x.x.x
容器ip:172.x.x.x
容器ip:202.x.x.x
```
none

```
不做任何网络配置，最大限度定制化

宿主机ip:198.x.x.x
容器ip:x.x.x.x
```

### 日志查看

```
# 查看容器运行日志
docker logs [container_id]
# 查看容器详细信息
docker inspect [container_id]  # 查看全部信息
 docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' [container_id]	# 查看容器网络信息
# 查看容器端口信息
docker port [container_id]
```

- 日志命令

```
docker logs -f -t --since="2017-05-31" --tail=10 edu_web_1


docker logs [OPTIONS] CONTAINER
Options:
        --details        显示更多的信息
    -f, --follow         跟踪日志输出，最后一行为当前时间戳的日志
        --since string   显示自具体某个时间或时间段的日志
        --tail string    从日志末尾显示多少行日志， 默认是all
    -t, --timestamps     显示时间戳
    	--until string   截止时间
```

输出形式

```
stdout 标准输出

stderr 标准错误

以json格式存放在容器对于到日志文件中
```

内容类型

```
docker自身运行时Daemon的日志内容

docker容器的日志内容
```

实现原理

```
Docker Daemon是Docker架构中一个常驻在后台的系统进程，它在后台启动了一个Server，Server负责接受Docker Client发送的请求；接受请求后，Server通过路由与分发调度，找到相应的Handler来执行请求

当我们输入docker logs的时候会转化为Docker Client向Docker Daemon发起请求,Docker Daemon 在运行容器时会去创建一个协程(goroutine)，绑定了整个容器内所有进程的标准输出文件描述符。因此容器内应用的所有只要是标准输出日志，都会被 goroutine 接收，Docker Daemon会根据容器id和日志类型读取日志内容，最终会输出到用户终端上并且通过json格式存放在/var/lib/docker/containers目录下。
```

声明周期

```
docker logs是跟随容器而产生的，如果删除了某个容器，相应的日志文件也会随着被删除
```

### 清理命令

杀死所有正在运行的容器

```
docker kill $(docker ps -a -q)
```

删除所有已经停止的容器

```
docker rm $(docker ps -a -q)
```

删除所有未打 dangling 标签的镜像

```
docker rmi $(docker images -q -f dangling=true)
```

通过镜像的id来删除指定镜像

```
docker rmi <image id>
```

删除所有镜像

```
docker rmi $(docker images -q)
```

为这些命令创建别名

```
# ~/.bash_aliases

# 杀死所有正在运行的容器.
alias dockerkill='docker kill $(docker ps -a -q)'

# 删除所有已经停止的容器.
alias dockercleanc='docker rm $(docker ps -a -q)'

# 删除所有未打标签的镜像.
alias dockercleani='docker rmi $(docker images -q -f dangling=true)'

# 删除所有已经停止的容器和未打标签的镜像.
alias dockerclean='dockercleanc || true && dockercleani'
```

# Dockerfile

Dockerfile类似于我们学习过的脚本， 将我们在上面学到的docker镜像， 使用自动化的方式实现出来

作用

```
找一个镜像： 	ubuntu
创建一个容器： docker run ubuntu
进入容器： 	docker exec -it 容器 命令
操作： 		各种应用配置
构造新镜像： 	docker commit
```

准则

```
大: 首字母必须大写D
空: 尽量将Dockerfile放在空目录中。 
单: 每个容器尽量只有一个功能。 
少: 执行的命令越少越好。
```

命令

```shell
# 构建镜像命令格
docker build -t [镜像名]:[版本号] [Dockerfile所在目录]

# 构建样例
docker build
-t nginx:v0.2 /opt/dockerfile/nginx/ 

# 参数详解:
-t 	# 指定构建后的镜像信息，
/opt/dockerfile/nginx/ # 则代表Dockerfile存放位置，如果是当前目录，则用 .(点)表示
```

## 样例

- 创建一个定制化的镜像:nginx

创建Dockerfile专用目录

```
mkdir /docker/images/nginx -p
cd /docker.images.nginx
```

创建Dockerfile文件

```shell
# 构建一个基于ubuntu的docker定制镜像
# 基础景象
FROM ubuntu
# 镜像作者
MAINTAINER President.Wang 000000@qq.com
# 执行命令
RUN apt-get update
RUN apt-get install nginx -y  # -y交互时默认输入y
# 对外端口
EXPOSE 80
```

构建镜像

```
docker build -t nginx:v01 .
```

修改Dockerfile文件

```
# 合并RUN命令
RUN apt-get update  && apt-get install nginx -y
```

第二次构建镜像

```
docker build -t nginx:v02 .
```

## 构建过程

**Dockerfile**构建过程:
```
从基础镜像1运行一个容器A 遇到一条Dockerfile指令，都对容器A做一次修改操作 执行完毕一条命令，提交生成一个新镜像2 再基于新的镜像2运行一个容器B 遇到一条Dockerfile指令，都对容器B做一次修改操作 执行完毕一条命令，提交生成一个新镜像3
 ...
```

构建过程镜像介绍 
```
构建过程中，创建了很多镜像，这些中间镜像，我们可以直接使用来启动容器，通过查看容器效果，从侧面能看到我们每次构建的效果。 
提供了镜像调试的能力
```
构建缓存 

```
我们第一次构建很慢，之后的构建都会很快，因为他们用到了构建的缓存。

不适用构建缓存方法常见两种:
  全部不同缓存: docker build --no-cache -t [镜像名]:[镜像版本] [Dockerfile位置] 
  部分使用缓存: ENV REFRESH_DATE 2018-01-12，只要构建的缓存时间不变，那么就用缓存，如果时间一旦改变，就不用缓存了
  
样例:
# 构建一个基于 ubuntu 16.04 的 docker 定制镜像 
# 基础镜像
FROM ubuntu-16.04
# 镜像作者
MAINTAINER President.Wang 000000@qq.com
# 创建构建刷新时间
ENV REFRESH_DATE 2018-01-12
# 执行命令
RUN apt-get update
RUN apt-get install nginx -y
# 对外端口 EXPOSE 80
```
构建历史
```
# 查看构建过程查看
docker history
```
## 指令

### 基础

FROM

```
# 格式
FROM <image>
FROM <image>:<tag>。

# 解释
FROM 是 Dockerfile 里的第一条而且只能是除了首行注释之外的第一条指令
可以有多个 FROM 语句，来创建多个 image
FROM 后面是有效的镜像名称，如果该镜像没有在你的本地仓库，那么就会从远程仓库 Pull 取，如果远程也没有，就报 错失败
下面所有的 系统可执行指令 在 FROM 的镜像中执行。
```

MAINTAINER

```
# 格式
MAINTAINER <name>

# 解释
指定该dockerfile文件的维护者信息。类似我们在 docker commit 时候使用-a 参数指定的信息
```

RUN

```
# 格式
RUN <command> (shell 模式) 
RUN["executable", "param1", "param2"]。 (exec 模式)

# 解释
表示当前镜像构建时候运行的命令，如果有确认输入的话，一定要在命令中添加 -y 如果命令较长，那么可以在命令结尾使用 \ 来换行 生产中，推荐使用上面数组的格式

# 注释
shell模式:类似于 /bin/bash-ccommand
举例: RUN echo hello
exec 模式:类似于 RUN ["/bin/bash", "-c", "command"]
举例: RUN ["echo", "hello"]
```

EXPOSE

```
# 格式
EXPOSE <port> [<port>...]

# 解释
设置 Docker 容器对外暴露的端口号，Docker 为了安全，不会自动对外打开端口，如果需要外部提供访问，还需要启动容
器时增加-p 或者-P 参数对容器的端口进行分配。
```

### 运行时

CMD
```
# 格式
CMD ["executable","param1","param2"]  (exec模式)推荐
CMD command param1 param2  (shell模式)
CMD ["param1","param2"]  提供给 ENTRYPOINT 的默认参数

# 解释
CMD 指定容器启动时默认执行的命令
每个Dockerfile只运行一条 CMD 命令，如果指定了多条，只有最后一条会被执行 如果你在启动容器的时候使用 docker run 指定的运行命令，那么会覆盖 CMD 命令。 举例: CMD ["/usr/sbin/nginx","-g","daemon off"]
```
ENTRYPOINT

```
# 格式
ENTRYPOINT ["executable", "param1","param2"] (exec 模式) ENTRYPOINT command param1 param2 (shell 模式)

# 解释
和 CMD 类似都是配置容器启动后执行的命令，并且不会被 docker run 提供的参数覆盖。 每个 Dockerfile 中只能有一个 ENTRYPOINT，当指定多个时，只有最后一个起效。 生产中我们可以同时使用 ENTRYPOINT 和 CMD，
想要在dockerrun时被覆盖，可以使用 "dockerrun--entrypoint"
```
- 实践

CMD实践

```shell
# 1.修改Dockerfile文件内容
# 在上一个 Dockerfile 文件内容基础上，末尾增加下面一句话
CMD ["/usr/sbin/nginx","-g","daemon off;"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.3 .
# 3.根据镜像创建容器，创建时，不添加执行命令
docker run -p 80 --name nginx-3 -d ubuntu-nginx:v0.3 docker ps
# 3.根据镜像创建容器,创建时候，添加执行命令/bin/bash
docker run -p 80 --name nginx-4 -d ubuntu-nginx:v0.3 /bin/bash
docker ps
```

ENTRYPOINT实践

```shell
# 1.修改Dockerfile文件内容
# 在上一个 Dockerfile 文件内容基础上，修改末尾的CMD为ENTRYPOINT
ENTRYPOINT ["/usr/sbin/nginx","-g","daemon off;"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.4 .
# 3.根据镜像创建容器，创建时，不添加执行命令
docker run -p 80 --name nginx-5 -d ubuntu-nginx:v0.4 docker ps
# 3.根据镜像创建容器,创建时候，添加执行命令/bin/bash
docker run -p 80 --name nginx-6 -d ubuntu-nginx:v0.4 /bin/bash
docker ps
# 4.根据镜像创建容器,创建时候，使用--entrypoint参数
docker run -p 80 --entrypoint "/bin/bash" --name nginx-7 -d ubuntu-nginx:v0.4 
docker ps
```

CMD&ENTRYPOINT实践

```shell
# 1.修改Dockerfile文件内容
# 在上一个 Dockerfile 文件内容基础上，修改末尾的的ENTRYPOINT
ENTRYPOINT ["/usr/sbin/nginx"]
CMD["-g"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.5 .
# 3.根据镜像创建容器，创建时，不添加执行命令
docker run -p 80 --name nginx-8 -d ubuntu-nginx:v0.5
docker ps
# 4.根据镜像创建容器,创建时候，不添加执行命令，覆盖cmd的参数-g "daemon off;"
docker run -p 80 --name nginx-9 -d ubuntu-nginx:v0.5 -g "daemon off;"
docker ps

# 注意
任何docker run设置的命令参数或者CMD指令的命令，都将作为ENTRYPOINT 指令的命令参数，追加到ENTRYPOINT指令之后
```

### 文件编辑

ADD

```
# 格式
ADD <src>... <dest>
ADD ["<src>",... "<dest>"]

# 解释
将指定的 <src> 文件复制到容器文件系统中的 <dest>
src 指的是宿主机，dest 指的是容器
淆。

所有拷贝到 container 中的文件和文件夹权限为 0755,uid 和 gid 为 0
如果文件是可识别的压缩格式，则 docker 会帮忙解压缩

# 注意
1、如果源路径是个文件，且目标路径是以 / 结尾， 则 docker 会把目标路径当作一个目录，会把源文件拷贝到该目录下;如果目标路径不存在，则会自动创建目标路径。 
2、如果源路径是个文件，且目标路径是不是以 / 结尾，则 docker 会把目标路径当作一个文件。
如果目标路径不存在，会以目标路径为名创建一个文件，内容同源文件;
如果目标文件是个存在的文件，会用源文件覆盖它，当然只是内容覆盖，文件名还是目标文件名。
如果目标文件实际是个存在的目录，则会源文件拷贝到该目录下。 注意，这种情况下，最好显示的以 / 结尾，以避免混
3、如果源路径是个目录，且目标路径不存在，则docker会自动以目标路径创建一个目录，把源路径目录下的文件拷贝进来。 如果目标路径是个已经存在的目录，则 docker 会把源路径目录下的文件拷贝到该目录下。
4、如果源文件是个压缩文件，则docker会自动帮解压到指定的容器目录中。
```

COPY

```
# 格式
COPY <src>... <dest>
COPY ["<src>",... "<dest>"]

# 解释
COPY 指令和 ADD 指令功能和使用方式类似。只是 COPY 指令不会做自动解压工作。
单纯复制文件场景，Docker 推荐使用 COPY
```

VOLUME

```
# 格式
VOLUME ["/data"]

# 解释
VOLUME 指令可以在镜像中创建挂载点，这样只要通过该镜像创建的容器都有了挂载点
通过 VOLUME 指令创建的挂载点，无法指定主机上对应的目录，是自动生成的。 
举例:VOLUME ["/var/lib/tomcat7/webapps/"]
```

- 实践

ADD实践

```shell
# 1. Dockerfile文件内容
...
# 执行命令
...
# 增加普通文件
ADD ["sources.list", "etc/apt/sources.list"] 
# 增加压缩文件
ADD ["linshi.tar.gz", "/nihao/"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.6 .
# 3.根据镜像创建容器，创建时候，不添加执行命令
docker run -p 80 --name nginx-10 -d ubuntu-nginx:v0.6
docker ps
```

COPY实践

```shell
# 1.修改Dockerfile文件
...
# 执行命令
...
# 增加普通文件
COPY index.html /var/www/html/
...
# 2.构建镜像
docker build -t ubuntu-nginx:v0.8 .
# 3.根据镜像创建容器，创建时候，不添加执行命令
docker run -p 80 --name nginx-12 -d ubuntu-nginx:v0.8
docker ps
```

VOLUME实践

```shell
# 1.修改Dockerfile文件
# 在上一个Dockerfile文件内容基础上，在COPY下面增加一个VOLUME
VOLUME ["/data/"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.9 .
# 3.创建容器
docker run -itd --name nginx-13 ubuntu-nginx:v0.9
# 4.查看镜像
docker inspect nginx-13
# 5.验证操作
docker run -itd --name vc-nginx-1 --volumes-from nginx-11 nginx
docker run -itd --name vc-nginx-2 --volumes-from nginx-11 nginx
# 6.进入容器
# 进入容器1
docker exec -it vc-nginx-1 /bin/bash
echo 'nihao itcast'>data/nihao.txt
# 进入容器2
docker exec -it vc-nginx-2 /bin/bash
cat data/nihao.txt
```

### 环境

USER

```
# 格式:
USER daemon

# 解释:
指定运行容器时的用户名和 UID，后续的 RUN 指令也会使用这里指定的用户。 
如果不输入任何信息，表示默认使用 root 用户
```

ENV

````
# 格式:
ENV <key> <value>
ENV <key>=<value> ...

# 解释:
设置环境变量，可以在 RUN 之前使用，然后 RUN 命令时调用，容器启动时这些环境变量都会被指定
````

WORKDIR

```
# 格式
WORKDIR /path/to/workdir (shell 模式)
 
# 解释
切换目录，为后续的 RUN、CMD、ENTRYPOINT 指令配置工作目录。 相当于 cd 可以多次切换(相当于 cd 命令)，
也可以使用多个 WORKDIR 指令，后续命令如果参数是相对路径，则会基于之前命令指定的路径。
例如 举例:
WORKDIR /a 
WORKDIR b
WORKDIR c
RUN pwd 
则最终路径为 /a/b/c。
```

ARG

```
# 格式
ARG <name>[=<default value>]

# 解释
ARG 指定了一个变量在 docker build 的时候使用，可以使用--build-arg <varname>=<value>来指定参数的值，不过如果构建的时候不指定就会报错。
```

- 实践

ENV实践

```shell
# 1.修改Dockerfile文件
# 在上一个Dockerfile文件内容基础上，在RUN下面增加一个ENV
ENV NNIHAO=helloword
# 2.构建镜像
docker build -t ubuntu-nginx:v0.10 .
# 3. 根据镜像创建容器，创建的时候，不添加执行命令
docker run -p 80 --name nginx-13 -d ubuntu-nginx:v0.10
docker exec -it 54f86 /bin/bash
echo $NIHAO
```

WORKDIR实践

```shell
# 1.修改Dockerfile文件内容
# 在上一个Dockerfile文件内容基础上，在RUN下面增加一个WORKDIR
WORKDIR /nihao/itcast/
RUN ["touch", "itcast.txt"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.11 .
# 3.根据镜像创建容器，创建的时候，不添加执行命令
docker run -p 80 --name nginx-14 -d ubuntu-nginx:v0.11
docker exec -it 54f75 /bin/bash
ls
```

### 触发器

ONBUILD

```
# 格式
ONBUILD [command]
# 解释
当一个镜像A被作为其他镜像B的基础镜像时，这个触发器才会被执行， 新镜像B在构建的时候，会插入触发器中的指令。
```

- 实践

```shell
# 1.修改Dockerfile文件内容：在COPY前面加ONBUILD
# 构建一个基于ubuntu的docker定制镜像
# 基础镜像
FROM ubuntu
# 镜像作者
MAINTAINER President.Wang 0000@qq.com
# 执行命令
RUN apt-get update
RUN apt-get install nginx -y
# 增加文件
ONBUILD COPY index.html /vzr/www/html/
# 对外端口
EXPOSE 80
# 执行命令ENTRYPOINT
ENTRYPOINT ["/usr/sbin/nginx", "-g", "daemon off;"]
# 2.构建镜像
docker build -t ubuntu-nginx:v0.12 .
# 3.根据镜像创建容器
docker run -p 80 --name nginx-15 -d ubuntu-nginx:v0.12
docker ps
# 4.访问容器页面，是否被更改


# 1.构建子镜像Dockerfile文件
FROM ubuntu-nginx:v0.12
MAINTAINER President.Wang 0000@qq.com
EXPOSE 80
ENTRYPOINT ["/usr/sbin/nginx", "-g", "daemo off;"]
# 2.构建子镜像
docker build -t ubuntu-nginx-sub:v0.1 .
# 3.根据镜像创建容器
docker run -p 80 --name nginx-16 -d ubuntu-nginx-sub:v0.1
docker ps
```
