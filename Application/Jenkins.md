# Jenkins

[官网](https://www.jenkins.io/zh/doc/)

## 安装配置

- 安装

docker

```shell
# 下载
docker pull jenkinsci/blueocean
# 运行
docker run \
  -u root \
  --rm \
  -d \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins-data:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  jenkinsci/blueocean
```

mac

```
brew install jenkins-lts
```

- 设置

解锁

```
1.浏览到 http://localhost:8080（或安装时为Jenkins配置的任何端口），并等待 解锁 Jenkins 页面出现。
2.从Jenkins控制台日志输出中，复制自动生成的字母数字密码（在两组星号之间）
3.在 解锁Jenkins 页面上，将此 密码 粘贴到管理员密码字段中，然后单击 继续 。
```

自定义插件

创建管理员用户

