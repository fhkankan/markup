[TOC]

# Seafile内网部署(ubuntu)

## 参考文档

<https://manual-cn.seafile.com/deploy/deploy_with_docker.html>

## Docker部署
### 安装Docker

```
sudo apt install docker.io
```

### 下载运行seafile镜像

```
sudo docker run -itd --name seafile \ 
	--restart=always \  
	-e SEAFILE_SERVER_HOSTNAME=seafile.example.com \ 
	-e SEAFILE_ADMIN_EMAIL=parobot@pingan.com.cn \ 
	-e SEAFILE_ADMIN_PASSWORD=0kV3GT4wSv{V \  
	-v /opt/seafile-data:/shared  \ 
	-p 10001:10001 -p 12001:12001 -p 8000:8000 -p 8080:8080 -p 8082:8082 \  		
	seafileltd/seafile:6.3.2
```

这条命令会将宿主机上的`/opt/seafile-data`目录挂载到 Seafile 容器中,你可以在这里找到日志或其他的数据文件.

## 更多配置

### 修改服务的配置

Seafile 服务的配置会存放在`/shared/seafile/conf`目录下，你可以根据 [Seafile 手册](https://manual.seafile.com/)修改配置

修改之后需要重启容器

```
docker restart seafile
```

### 查询日志

Seafile 服务的日志会存放在`/shared/logs/seafile`目录下, 由于是将`/opt/seafile-data`挂载到`/shared`，所以同样可以在宿主机上的`/opt/seafile-data/logs/seafile`目录下找到.

系统日志会存放在`/shared/logs/var-log`目录下.

### 添加新管理员

```
docker exec -it seafile /opt/seafile/seafile-server-latest/reset-admin.sh
```

然后根据提示输入用户名以及密码即可

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



