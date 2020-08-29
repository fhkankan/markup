# GitLab内网部署
## 参考文档
https://www.cnblogs.com/kowloon/p/7504140.html

## Docker部署

### 安装

```
# 查找镜像
docker search gitlab
# 拉取镜像
docker pull gitlab/gitlab-ce:latest
# 启动镜像
sudo docker run --detach \
--hostname gitlab.example.com \
--publish 443:443 --publish 80:80 --publish 22:22 \
--name gitlab \
--restart always \
--volume /srv/gitlab/config:/etc/gitlab \
--volume /srv/gitlab/logs:/var/log/gitlab \
--volume /srv/gitlab/data:/var/opt/gitlab \
gitlab/gitlab-ce:latest
```

配置

```
# 进入gitlab的docker环境中
docker exec -it gitlab /bin/bash
# 修改配置文件
/etc/gitlab/gitlab.rb

# 或者
docker exec -it gitlab vim /etc/gitlab/gitlab.rb
docker restart gitlab
```

由于权限而启动失败

```
docker exec -it gitlab update-permissions
docker restart gitlab
```

### 更新

```
# 首先需要停止并删除当前的Gitlab实例
sudo docker stop gitlab
sudo docker rm gitlab
# 拉取最新版的Gitlab
sudo docker pull gitlab/gitlab-ce:latest
# 使用上次的配置运行gitlab
sudo docker run --detach \
--hostname gitlab.example.com \
--publish 443:443 --publish 80:80 --publish 22:22 \
--name gitlab \
--restart always \
--volume /srv/gitlab/config:/etc/gitlab \
--volume /srv/gitlab/logs:/var/log/gitlab \
--volume /srv/gitlab/data:/var/opt/gitlab \
gitlab/gitlab-ce:latest
```

### 备份

```
 # 备份文件所在路径为：/var/opt/gitlab/backups/
gitlab-rake gitlab:backup:create
# 备份得到的文件格式如：1504860571_2017_09_08_9.5.3_gitlab_backup.tar   时间戳_年_月_日_gitlap版本_gitlab_backup.tar

# 若修改备份文件的存放路径：
vim  /etc/gitlab/gitlab.rb 
# 修改 
gitlab_rails['backup_path'] = "/var/opt/gitlab/backups"

# 自动备份:每天的10:37执行备份。
crontab -e    37 10 * * * /opt/gitlab/bin/gitlab-rake gitlab:backup:create 

# 自动清除备份文件
# 创建备份脚本，删除30以前的备份文件
vim /var/opt/gitlab/backups/remove.sh  
!/bin/bash
find "/var/opt/gitlab/backups/" -name "*.tar" -ctime +30  -exec rm -rf {} \;  
# 每天的10:45执行删除备份的脚本
chmod +x /var/opt/gitlab/backups/remove.sh
contab -e 45 10 * * * sh /var/opt/gitlab/backups/remove.sh 
```

### 还原

```
# 停止相关数据连接服务
gitlab-ctl stop unicorn
gitlab-ctl stop sidekiq

# 还原操作
# 假设从1505097437_2017_09_11_9.5.3_gitlab_backup.tar备份文件中恢复
gitlab-rake gitlab:backup:restore BACKUP=1505097437_2017_09_11_9.5.3

# 启动gitlab服务
gitlab-ctl start

# 注意：不能直接在终端执行gitlab-ctl stop停止所有服务。因gitlab删除和还原操作还需要使用到redis和postgresql连接
```

### 迁移

在新的服务器上搭建好gitlab环境但gitlab版本需跟原有版本一致。然后将原gitlab备份拷贝到新gitlab环境的对应位置，再执行还原过程即可。

> 同步镜像

方法一：

````
从官方下载同一版本的gitlab镜像
````

方法二

```
# 基于当前容器创建备份镜像
# 备份
docker ps -a  # 查看容器
docker commit -p 30b8f18f20b4 container-backup  # 创建快照
docker images  # 查看镜像
# 方式一：云备份
docker login 172.16.101.192  # 登录Docker注册中心，
docker tag a25ddfec4d2a arunpyasi/container-backup:test # 打标签
docker push arunpyasi/container-backup  # 推送镜像
# 方式二：本地备份
docker save -o ~/container-backup.tar container-backup  # 本地保存镜像
# 加载
# 方式一：云拉取
docker pull arunpyasi/container-backup:test  # 拉取镜像
# 方式二：本地加载
docker load -i ~/container-backup.tar  # 本地加载
docker images  # 查看镜像
mkdir /srv/gitlab/config
mkdir /srv/gitlab/logs
mkdir /srv/gitlab/data
```

运行镜像

```
sudo docker run --detach \
--hostname gitlab.example.com \
--publish 443:443 --publish 80:80 --publish 22:22 \
--name gitlab \
--restart always \
--volume /srv/gitlab/config:/etc/gitlab \
--volume /srv/gitlab/logs:/var/log/gitlab \
--volume /srv/gitlab/data:/var/opt/gitlab \
gitlab
```

> 同步数据

```
# 备份数据(旧服务器)
docker ps -a  # 查看容器
dicker exec -it gitlab名 /bin/bash. # 进入docker的gitlab
cd /var/opt/gitlab/backups/  # 进入备份文件夹
gitlab-rake gitlab:backup:create  # 生成备份文件
exit  # 退出docker的gitlab
sudo docker cp id:/var/opt/gitlab/backups/备份文件名 ./  #移动备份文件至外部
sudo scp 备份文件名 新服务器地址

# 还原数据(新服务器)
docker ps -a  # 查看容器
sudo docker cp ./备份文件名 id:/var/opt/gitlab/backups/  #移动备份文件至内部
dicker exec -it gitlab名 /bin/bash. # 进入docker的gitlab
cd /var/opt/gitlab/backups/  # 进入备份文件夹
chmod 777 备份文件名  # 更改权限
gitlab-ctl stop unicorn  # 停止相关数据连接服务
gitlab-ctl stop sidekiq
# 假设从1505097437_2017_09_11_9.5.3_gitlab_backup.tar备份文件中恢复
gitlab-rake gitlab:backup:restore BACKUP=1505097437_2017_09_11_9.5.3
gitlab-ctl start  # 启动gitlab服务

# 注意：不能直接在终端执行gitlab-ctl stop停止所有服务。因gitlab删除和还原操作还需要使用到redis和postgresql连接
```

**注意：**cp的时候属主和属组会改变，导致权限不够，需要修改成git为所属者。

在终端执行：

```
chown git:git  /var/opt/gitlab/backups/1505097437_2017_09_11_9.5.3_gitlab_backup.tar
```

