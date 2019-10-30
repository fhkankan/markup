# mosquitto

[参考](http://mosquitto.org)

## 安装

> mac

安装命令

```
brew install mosquitto
```
配置文件
```
/usr/local/etc/mosquitto/mosquitto.conf
```

> ubuntu

```
sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa
sudo apt-get update
sudo apt-get install mosquitto
```

## 使用

> mac

```
# 后台启动
brew services start mosquitto
# 临时启动
mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf

# 停止
brew services stop mosquitto
# 重启
brew services restart mosquitto
```

## 配置

### 默认配置

```
# =================================================================
# General configuration
# =================================================================
 
# 客户端心跳的间隔时间
#retry_interval 20
 
# 系统状态的刷新时间
#sys_interval 10
 
# 系统资源的回收时间，0表示尽快处理
#store_clean_interval 10
 
# 服务进程的PID
#pid_file /var/run/mosquitto.pid
 
# 服务进程的系统用户
#user mosquitto
 
# 客户端心跳消息的最大并发数
#max_inflight_messages 10
 
# 客户端心跳消息缓存队列
#max_queued_messages 100
 
# 用于设置客户端长连接的过期时间，默认永不过期
#persistent_client_expiration
 
# =================================================================
# Default listener
# =================================================================
 
# 服务绑定的IP地址
#bind_address
 
# 服务绑定的端口号
#port 1883
 
# 允许的最大连接数，-1表示没有限制
#max_connections -1
 
# cafile：CA证书文件
# capath：CA证书目录
# certfile：PEM证书文件
# keyfile：PEM密钥文件
#cafile
#capath
#certfile
#keyfile
 
# 必须提供证书以保证数据安全性
#require_certificate false
 
# 若require_certificate值为true，use_identity_as_username也必须为true
#use_identity_as_username false
 
# 启用PSK（Pre-shared-key）支持
#psk_hint
 
# SSL/TSL加密算法，可以使用“openssl ciphers”命令获取
# as the output of that command.
#ciphers
 
# =================================================================
# Persistence
# =================================================================
 
# 消息自动保存的间隔时间
#autosave_interval 1800
 
# 消息自动保存功能的开关
#autosave_on_changes false
 
# 持久化功能的开关
persistence true
 
# 持久化DB文件
#persistence_file mosquitto.db
 
# 持久化DB文件目录
#persistence_location /var/lib/mosquitto/
 
# =================================================================
# Logging
# =================================================================
 
# 4种日志模式：stdout、stderr、syslog、topic
# none 则表示不记日志，此配置可以提升些许性能
log_dest none
 
# 选择日志的级别（可设置多项）
#log_type error
#log_type warning
#log_type notice
#log_type information
 
# 是否记录客户端连接信息
#connection_messages true
 
# 是否记录日志时间
#log_timestamp true
 
# =================================================================
# Security
# =================================================================
 
# 客户端ID的前缀限制，可用于保证安全性
#clientid_prefixes
 
# 允许匿名用户
#allow_anonymous true
 
# 用户/密码文件，默认格式：username:password
#password_file
 
# PSK格式密码文件，默认格式：identity:key
#psk_file
 
# pattern write sensor/%u/data
# ACL权限配置，常用语法如下：
# 用户限制：user <username>
# 话题限制：topic [read|write] <topic>
# 正则限制：pattern write sensor/%u/data
#acl_file
 
# =================================================================
# Bridges
# =================================================================
 
# 允许服务之间使用“桥接”模式（可用于分布式部署）
#connection <name>
#address <host>[:<port>]
#topic <topic> [[[out | in | both] qos-level] local-prefix remote-prefix]
 
# 设置桥接的客户端ID
#clientid
 
# 桥接断开时，是否清除远程服务器中的消息
#cleansession false
 
# 是否发布桥接的状态信息
#notifications true
 
# 设置桥接模式下，消息将会发布到的话题地址
# $SYS/broker/connection/<clientid>/state
#notification_topic
 
# 设置桥接的keepalive数值
#keepalive_interval 60
 
# 桥接模式，目前有三种：automatic、lazy、once
#start_type automatic
 
# 桥接模式automatic的超时时间
#restart_timeout 30
 
# 桥接模式lazy的超时时间
#idle_timeout 60
 
# 桥接客户端的用户名
#username
 
# 桥接客户端的密码
#password
 
# bridge_cafile：桥接客户端的CA证书文件
# bridge_capath：桥接客户端的CA证书目录
# bridge_certfile：桥接客户端的PEM证书文件
# bridge_keyfile：桥接客户端的PEM密钥文件
#bridge_cafile
#bridge_capath
#bridge_certfile
#bridge_keyfile
 
# 自己的配置可以放到以下目录中
include_dir /etc/mosquitto/conf.d
```

### 用户密码

修改配置文件

```
# 不允许匿名
allow_anonymous false
# 配置用户名密码
password_file /etc/mosquitto/pwfile
# 配置topic和用户关系
acl_file /etc/mosquitto/acl
```

添加用户信息

```shell
# 使用下面命令输入用户名chisj， 按照提示输入密码
# 自动生成相应的用户名和密码记录到指定的password_file
mosquitto_passwd -c /etc/mosquitto/pwfile chisj
```

添加Topic和用户关系

```shell
# 打开制定的acl_file
vim /etc/mosquitto/acl
# 输入如下信息
user chisj
topic  write mtopic

user chisj
topic  read mtopic
```

### 认证测试

重启

```shell
brew services restart mosquitto
或
CTRL+C关闭mosquitto
mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
```

订阅窗口

```shell
mosquitto_sub -h 192.168.1.100 -t mtopic -u chisj -P chisj
```

发布窗口

```shell
mosquitto_pub -h 192.168.1.100 -t mtopic -u chisj -P chisj -m "test"
```

