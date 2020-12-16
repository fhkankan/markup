[TOC]

# Seafile内网部署

[参考文档](https://cloud.seafile.com/published/seafile-manual-cn/overview/components.md)

## 安装部署
### 脚本安装

[参考文档](wget https://raw.githubusercontent.com/haiwen/seafile-server-installer-cn/master/seafile-server-7.1-ubuntu-amd64-http)

```shell
# 获取安装脚本
wget https://raw.githubusercontent.com/haiwen/seafile-server-installer-cn/master/seafile-server-7.1-ubuntu-amd64-http

bash seafile-server-ubuntu-amd64-http 6.0.13

# 启动关闭服务
service seafile-server stop
service seafile-server start
```

### docker安装

[参考文档](https://cloud.seafile.com/published/seafile-manual-cn/overview/components.md)

- 旧方法

```shell
# 下载docker镜像
sudo docker pull seafileltd/seafile:6.3.2

# 运行
sudo docker run -itd --name seafile  
	--restart=always   
	-e SEAFILE_SERVER_HOSTNAME=seafile.example.com 
	-e SEAFILE_ADMIN_EMAIL=admin@example.com 
	-e SEAFILE_ADMIN_PASSWORD=0kV3GT4wSv{V  
	-v /opt/seafile-data:/shared  
	-p 10001:10001 -p 12001:12001 -p 8000:8000 -p 8080:8080 -p 8082:8082  
	seafileltd/seafile:6.3.2
```

这条命令会将宿主机上的`/opt/seafile-data`目录挂载到 Seafile 容器中,你可以在这里找到日志或其他的数据文件.

- 新方法

下载docker-compose

```shell
sudo apt-get install docker-compose -y
```

下载下载 [docker-compose.yml](https://docs.seafile.com/d/cb1d3f97106847abbf31/files/?p=/docker/docker-compose.yml) 示例文件到您的服务器上，然后根据您的实际环境修改该文件。尤其是以下几项配置：

- MySQL root 用户的密码 (MYSQL_ROOT_PASSWORD and DB_ROOT_PASSWD)
- 持久化存储 MySQL 数据的 volumes 目录 (volumes)
- 持久化存储 Seafile 数据的 volumes 目录 (volumes)

docker-compose.yml文件内容

```
version: '2.0'
services:
  db:
    image: mariadb:10.1
    container_name: seafile-mysql
    environment:
      - MYSQL_ROOT_PASSWORD=db_dev  # Requested, set the root's password of MySQL service.
      - MYSQL_LOG_CONSOLE=true
    volumes:
      - /opt/seafile-mysql/db:/var/lib/mysql  # Requested, specifies the path to MySQL data persistent store.
    networks:
      - seafile-net

  memcached:
    image: memcached:1.5.6
    container_name: seafile-memcached
    entrypoint: memcached -m 256
    networks:
      - seafile-net
          
  seafile:
    image: seafileltd/seafile-mc:latest
    container_name: seafile
    ports:
      - "80:80"
#      - "443:443"  # If https is enabled, cancel the comment.
    volumes:
      - /opt/seafile-data:/shared   # Requested, specifies the path to Seafile data persistent store.
    environment:
      - DB_HOST=db
      - DB_ROOT_PASSWD=db_dev  # Requested, the value shuold be root's password of MySQL service.
#      - TIME_ZONE=Asia/Shanghai # Optional, default is UTC. Should be uncomment and set to your local time zone.
      - SEAFILE_ADMIN_EMAIL=me@example.com # Specifies Seafile admin user, default is 'me@example.com'.
      - SEAFILE_ADMIN_PASSWORD=asecret     # Specifies Seafile admin password, default is 'asecret'.
      - SEAFILE_SERVER_LETSENCRYPT=false   # Whether use letsencrypt to generate cert.
      - SEAFILE_SERVER_HOSTNAME=seafile.example.com # Specifies your host name.
    depends_on:
      - db
      - memcached
    networks:
      - seafile-net

networks:
  seafile-net:
```

启动Seafile服务

```shell
# 在docker-compose.yml文件所在的目下执行以上命令
sudo docker-compose up -d
```

浏览器访问Seafile 主页

```
http://seafile.example.com
```

### Seafile 目录结构

`/shared`

共享卷的挂载点,您可以选择在容器外部存储某些持久性信息.在这个项目中，我们会在外部保存各种日志文件和上传数据。 这使您可以轻松重建容器而不会丢失重要信息。

- /shared/seafile: Seafile 服务的配置文件以及数据文件
- /shared/logs: 日志目录
    - /shared/logs/var-log: 我们将容器内的`/var/log`链接到本目录。您可以在`/shared/logs/var-log/nginx/`中找到 nginx 的日志文件
    - /shared/logs/seafile: Seafile 服务运行产生的日志文件目录。比如您可以在 `/shared/logs/seafile/seafile.log` 文件中看到 seaf-server 的日志
- /shared/ssl: 存放证书的目录，默认不存在

## 更多配置

### 自定义管理员用户名和密码

- 旧方法

修改启动命令

```shell
sudo docker run -itd --name seafile  
	--restart=always   
	-e SEAFILE_SERVER_HOSTNAME=seafile.example.com 
	-e SEAFILE_ADMIN_EMAIL=admin@example.com 
	-e SEAFILE_ADMIN_PASSWORD=0kV3GT4wSv{V  
	-v /opt/seafile-data:/shared  
	-p 10001:10001 -p 12001:12001 -p 8000:8000 -p 8080:8080 -p 8082:8082  
	seafileltd/seafile:6.3.2
```

- 新方法

默认的管理员账号是 `me@example.com` 并且该账号的密码是 `asecret`，您可以在 `docker-compose.yml` 中配置不同的用户名和密码，为此您需要做如下配置：

```
seafile:    
	...
    environment:        
    	...        
    	- SEAFILE_ADMIN_EMAIL=me@example.com        
    	- SEAFILE_ADMIN_PASSWORD=a_very_secret_password        
    	...
```

### 添加SSL证书

如果您把 `SEAFILE_SERVER_LETSENCRYPT` 设置为 `true`，该容器将会自动为您申请一个 letsencrypt 机构颁发的 SSL 证书，并开启 https 访问，为此您需要做如下配置：

```
seafile:    
	...    
	ports:        
		- "80:80"        
		- "443:443"    
		...    
	environment:        
		...        
		- SEAFILE_SERVER_LETSENCRYPT=true        
		- SEAFILE_SERVER_HOSTNAME=docs.seafile.com        
		...
```

如果您想要使用自己的 SSL 证书，而且如果用来持久化存储 Seafile 数据的目录为 `/opt/seafile-data`，您可以做如下处理：

- 创建 `/opt/seafile-data/ssl` 目录，然后拷贝您的证书文件和密钥文件到ssl目录下。
- 假设您的站点名称是 `seafile.example.com`，那么您的证书名称必须就是 `seafile.example.com.crt`，密钥文件名称就必须是 `seafile.example.com.key`。

### 修改服务的配置

Seafile 服务的配置会存放在`/shared/seafile/conf`目录下，你可以根据 [Seafile 手册](https://manual-cn.seafile.com/)修改配置

修改之后需要重启容器

```shell
# 旧方法
docker restart seafile
# 新方法
docker-compose restart
```

### 查询日志

Seafile 容器中 Seafile 服务本身的日志文件存放在 `/shared/logs/seafile` 目录下，或者您可以在宿主机上 Seafile 容器的卷目录中找到这些日志，例如：`/opt/seafile-data/logs/seafile`

同样 Seafile 容器的系统日志存放在 `/shared/logs/var-log` 目录下，或者宿主机目录 `/opt/seafile-data/logs/var-log`。

### 添加新管理员

```
docker exec -it seafile /opt/seafile/seafile-server-latest/reset-admin.sh
```

然后根据提示输入用户名以及密码即可

### 升级 Seafile 服务

如果要升级 Seafile 服务到最新版本：

```shell
# 旧方法
docker pull seafileltd/seafile

# 新方法
docker pull seafileltd/seafile-mc:latestdocker-compose downdocker-compose up -d
```

## 备份和恢复

### 目录结构

我们假设您的 seafile 数据卷路径是 `/opt/seafile-data`，并且您想将备份数据存放到 `/opt/seafile-backup` 目录下。

您可以创建一个类似以下 `/opt/seafile-backup` 的目录结构：

```
/opt/seafile-backup---- databases/  用来存放 MySQL 容器的备份数据---- data/  用来存放 Seafile 容器的备份数据
TextHTMLCSSJavascriptCC++C#JavaPythonSqlSwift
```

要备份的数据文件：

```
/opt/seafile-data/seafile/conf  # configuration files/opt/seafile-data/seafile/seafile-data # data of seafile/opt/seafile-data/seafile/seahub-data # data of seahub
TextHTMLCSSJavascriptCC++C#JavaPythonSqlSwift
```

### 备份数据

 步骤：

1. 备份 MySQL 数据库数据；
2. 备份 Seafile 数据目录；

- 备份数据库：

    ```
    # 建议每次将数据库备份到一个单独的文件中。至少在一周内不要覆盖旧的数据库备份。cd /opt/seafile-backup/databasesdocker exec -it seafile-mysql mysqldump  -uroot --opt ccnet_db > ccnet_db.sqldocker exec -it seafile-mysql mysqldump  -uroot --opt seafile_db > seafile_db.sqldocker exec -it seafile-mysql mysqldump  -uroot --opt seahub_db > seahub_db.sql
    TextHTMLCSSJavascriptCC++C#JavaPythonSqlSwift
    ```

- 备份 Seafile 资料库数据：

    - 直接复制整个数据目录

        ```
        cp -R /opt/seafile-data/seafile /opt/seafile-backup/data/cd /opt/seafile-backup/data && rm -rf ccnet
        TextHTMLCSSJavascriptCC++C#JavaPythonSqlSwift
        ```

    - 使用 rsync 执行增量备份

        ```
        rsync -az /opt/seafile-data/seafile /opt/seafile-backup/data/cd /opt/seafile-backup/data && rm -rf ccnet
        TextHTMLCSSJavascriptCC++C#JavaPythonSqlSwift
        ```

### 恢复数据

- 恢复数据库：

    ```
    docker cp /opt/seafile-backup/databases/ccnet_db.sql seafile-mysql:/tmp/ccnet_db.sqldocker cp /opt/seafile-backup/databases/seafile_db.sql seafile-mysql:/tmp/seafile_db.sqldocker cp /opt/seafile-backup/databases/seahub_db.sql seafile-mysql:/tmp/seahub_db.sql﻿
    docker exec -it seafile-mysql /bin/sh -c "mysql -uroot ccnet_db < /tmp/ccnet_db.sql"docker exec -it seafile-mysql /bin/sh -c "mysql -uroot seafile_db < /tmp/seafile_db.sql"docker exec -it seafile-mysql /bin/sh -c "mysql -uroot seahub_db < /tmp/seahub_db.sql"
    TextHTMLCSSJavascriptCC++C#JavaPythonSqlSwift
    ```

- 恢复 seafile 数据：

    ```
    cp -R /opt/seafile-backup/data/* /opt/seafile-data/seafile/
    ```

## 问题排查

```
docker logs -f seafile
# or
docker exec -it seafile bash
```

## 与onlyOffice集成

https://manual.seafile.com/deploy/deploy/deploy/only_office.md#deploy-onlyoffice-documentserver-docker-image

### 部署ood

下载运行onlyoffice/documentserver

```
docker run -dit -p 70:80 -p 7443:443 --restart always --name oods onlyoffice/documentserver
```

验证服务部署是否ok

```
在浏览器中输入
http//{ip地址}:70
或
https://{域名}:7443
```

### 配置seafile

```
# 查看docker容器信息
sudo docker ps -a
# 进入seafile的docker容器中
sudo docker exec -it 容器id /bin/bash
# 进入配置信息
cd config
# 删除编译文件
rm -rf seahub_settings.pyc
# 添加如下的信息
vim seahub_settings.py
# 退出
exit
# 重启seafile容器
sduo docker stop 容器id
sduo docker start 容器id
```

在seahub_settings.py中添加

```
# Enable Only Office
ENABLE_ONLYOFFICE = True
VERIFY_ONLYOFFICE_CERTIFICATE = False
ONLYOFFICE_APIJS_URL = 'http{s}://{your OnlyOffice server's domain or IP}/web-apps/apps/api/documents/api.js'
ONLYOFFICE_FILE_EXTENSION = ('doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'odt', 'fodt', 'odp', 'fodp', 'ods', 'fods')
ONLYOFFICE_EDIT_FILE_EXTENSION = ('docx', 'pptx', 'xlsx')
```

# 客户端

客户端有同步盘、挂载盘、web页三种选择方式，进入[官网选择](https://www.seafile.com/en/download/)



