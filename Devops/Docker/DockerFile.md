[TOC]
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

# 将当前目录作为docker目录
docker build -t nginx:v0.3 -f ./docker/Dockerfile .
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

