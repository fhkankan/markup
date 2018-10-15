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
rm -rf /etc/dpcker
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

### 镜像管理

镜像是一个只读文件，是一个能被docker运行起来的一个程序。通过运行这个程序完成各种应用的部署。

> 搜索、查看、获取

```
# 搜索镜像
docker search [image_name]
# 获取镜像
docker pull [image_name]
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
docker save -o [导出的镜像名称] [镜像]
# 导入镜像(两种方式)
docker load < [image.tar_name]
docker load --input [image.tar_name]
```

> 历史、创建

```
# 查看历史
docker history [image_name]
# 根据模板创建镜像
cat 模板文件名 | docker import - [自定义镜像名]
```

### 容器管理

容器类似一个操作系统，这个操作系统启动了某些服务，这里的容器指运行起来的一个Docker镜像

> 查看、启动

```
# 查看容器详情
docker ps
# 查看容器ip
docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_id

# 启动
# 方式一：基于镜像创建新容器并启动
docker run <参数，可选> [docker_image] [执行的命令]
eg:docker run nginx /bin/echo "hello docker"
参数选项：
-a stdin: 指定标准输入输出内容类型，可选 STDIN/STDOUT/STDERR 三项；
-d: 后台运行容器，并返回容器ID；
-i: 以交互模式运行容器，通常与 -t 同时使用；
-p: 端口映射，格式为：主机(宿主)端口:容器端口
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

# 方式二：启动已终止的容器
docker start [container_id]
# 方式三：在后台以守护进程方式启动
docker rn -d [image_name] command ...
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
docker inspect [container_id]
# 查看容器端口信息
docker port [container_id]
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

使用命令 docker run 用来创建容器， 可以在使用docker run 命令时添加 -v 参数， 就可以创建并挂载一个到多个数据卷到当前运行的容器中。
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



