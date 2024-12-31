# Loki

[文档](https://grafana.com/docs/loki/latest/)

## 概览

与其他日志系统不同，Loki的构建理念是只对日志标签的元数据进行索引（就像Prometheus 标签一样）。然后，日志数据本身被压缩并存储在对象存储中的块中，如Amazon Simple Storage Service（S3）或Google Cloud Storage（GCS），甚至本地存储在文件系统上。

Loki是一个水平可扩展，高可用性，多租户的日志聚合系统。它的设计非常经济高效且易于操作，因为它不会为日志内容编制索引，而是为每个日志流编制一组标签。项目受 Prometheus 启发，官方的介绍就是： Like Prometheus, but for logs ，类似于 Prometheus 的日志系统。

不对日志进行全文索引。通过存储压缩非结构化日志和仅索引元数据， Loki 操作起来会更简单，更省成本。通过使用与 Prometheus 相同的标签记录流对日志进行索引和分组，这使得日志的扩展和操作效率更高。特别适合储存 Kubernetes Pod 日志 ; 诸如 Pod 标签之类的元数据会被自动删除和编入索引

主要是分为三个部分：loki，主服务器存储日志和处理查询；promtail，代理收集日志并发给Loki；Grafana查询和显示日志

对比

| 名称    | 组件                                     | 优点                                                         |
| ------- | ---------------------------------------- | ------------------------------------------------------------ |
| ELK/EFK | es/logstash/kibana/filebeat/kafaka/redis | 支持自定义grol正则解析复杂日志内容；dashboard支持丰富的可视化展示 |
| Loki    | grafana/loki/promtail                    | 占用资源小，grafana原生支持，查询速度快                      |

## 安装

可以使用源文件、安装包、docker镜像、helm包、k8s等

- 安装loki

```shell
yum install -y https://github.com/grafana/loki/releases/download/v2.9.8/loki-2.9.8.x86_64.rpm
 
# 配置文件详解
[root@k8s-master01 ~]# cat /etc/loki/config.yml 
auth_enabled: false
 
server:
  http_listen_port: 3100    # http访问端口
  grpc_listen_port: 9096    # rpc访问端口
 
common:
  instance_addr: 192.168.186.100    # 修改为自己的IP或localhost
  path_prefix: /tmp/loki
  storage:
    filesystem:
      chunks_directory: /tmp/loki/chunks  # 记录块存储目录，默认chunks块上的日志数量或到期后，将chunks数据打标签后存储
      rules_directory: /tmp/loki/rules    # 规则配置目录
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory
 
query_range:        # 查询规则
  results_cache:    # 结果缓存
    cache:
      embedded_cache:    # 默认开启后会有提示，未配置缓存项，可以暂不开启
        enabled: true
        max_size_mb: 100
 
schema_config:    # 配置索引信息
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_    # 索引前缀
        period: 24h       # 索引时长
 
ruler:
  alertmanager_url: http://localhost:9093
 
# By default, Loki will send anonymous, but uniquely-identifiable usage and configuration
# analytics to Grafana Labs. These statistics are sent to https://stats.grafana.org/
#
# Statistics help us better understand how Loki is used, and they show us performance
# levels for most users. This helps us prioritize features and documentation.
# For more information on what's sent, look at
# https://github.com/grafana/loki/blob/main/pkg/usagestats/stats.go
# Refer to the buildReport method to see what goes into a report.
#
# If you would like to disable reporting, uncomment the following lines:
#analytics:
#  reporting_enabled: false
 
# 启动服务
systemctl enable --now loki
```

- 安装promtail

```shell
yum install -y https://github.com/grafana/loki/releases/download/v2.9.8/promtail-2.9.8.x86_64.rpm
 
# 配置文件详解 /etc/promtail/config.yml
[root@k8s-master01 ~]# cat /etc/promtail/config.yml 
# This minimal config scrape only single log file.
# Primarily used in rpm/deb packaging where promtail service can be started during system init process.
# And too much scraping during init process can overload the complete system.
# https://github.com/grafana/loki/issues/11398
 
server:
  http_listen_port: 9080
  grpc_listen_port: 0
 
positions:
  filename: /tmp/positions.yaml    # 用于记录每次读取日志文件的索引行数，如：promtail重启后从该配置中恢复日志文件的读取位置
 
clients:
- url: http://192.168.186.100:3100/loki/api/v1/push    # 推送日志流到Loki中的api
 
scrape_configs:      # 发现日志文件的位置并从中提取标签
- job_name: system   # 任务名称
  static_configs:    # 目录配置
  - targets:         # 标签
      - localhost
    labels:
      job: varlogs    # 子任务名称，通常以项目命令
      #NOTE: Need to be modified to scrape any additional logs of the system.
      __path__: /var/log/messages    # 要读取的日志文件的位置，允许使用通配符/*log或/**/*.log
  - targets:
      - localhost
    labels:
      job: securelogs
      #NOTE: Need to be modified to scrape any additional logs of the system.
      __path__: /var/log/secure     # 定义不同的日志文件路径
 
# 赋予权限
[root@k8s-master01 ~]# setfacl -m u:promtail:r /var/log/secure 
[root@k8s-master01 ~]# setfacl -m u:promtail:r /var/log/messages 
 
 
# 启动服务
systemctl enable --now promtail
# 检查 promtial 配置
http://IP:9080/targets
```

- 安装granafa

```shell
yum install -y https://dl.grafana.com/enterprise/release/grafana-enterprise-10.0.2-1.x86_64.rpm
# 启动服务
systemctl enable --now grafana-server
```

## LogQL

选择器
```
# 对于查询表达式的标签部分，将放在 {} 中，多个标签表达式用逗号分隔
{app="mysql",name="mysql-backup"}
```
支持的符号
```
= ：完全相同。
!= ：不平等。
=~ ：正则表达式匹配。
!~ ：不要正则表达式匹配。
```
过滤表达式
```
# 编写日志流选择器后，您可以通过编写搜索表达式进一步过滤结果。搜索表达式可以文本或正则表达式。 如

{job=“mysql”} |= “error”
{name=“kafka”} |~ “tsdb-ops.*io:2003”
{instance=~“kafka-[23]”,name=“kafka”} != kafka.server:type=ReplicaManager

# 支持多个过滤：
{job=“mysql”} |= “error” != “timeout”

# 目前支持的操作符
|= line 包含字符串。
!= line 不包含字符串。
|~ line 匹配正则表达式。
!~ line 与正则表达式不匹配。
```

## nginx日志

修改nginx的日志格式

```shell
log_format json escape=json '{'
	'"remote_addr": "$remote_addr", '
	'"request_uri": "$request_uri", '
	'"request_length": "$request_length", '
	'"request_time": "$request_time", '
	'"request_method": "$request_method", '
	'"status": "$status", '
	'"body_bytes_sent": "$body_bytes_sent", '
	'"http_referer": "$http_referer", '
	'"http_user_agent": "$http_user_agent", '
	'"http_x_forwarded_for": "$http_x_forwarded_for", '
	'"http_host": "$http_host", '
	'"server_name": "$server_name", '
	'"upstream": "$upstream_addr", '
	'"upstream_response_time":"$upstream_response_time", '
	'"upstream_status": "$upstream_status", '
	#'"geoip_country_code": "$geoip2_data_country_code", '
	#'"geoip_country_name": "$geoip2_data_country_name", '
	#'"geoip_city_name": "$geoip2_data_city_name"'
	'}';
    access_log  /var/log/nginx/json_access.log json;
 

"""
参数 描述
remote_addr 客户端的IP地址
request_uri 客户端请求的URI
request_length 请求的内容长度
request_time 请求处理时间
request_method 请求方法（GET、POST等）
status HTTP响应状态码
body_bytes_sent 发送给客户端的字节数
http_referer 请求中的Referer头部
http_user_agent 客户端的User-Agent头部
http_x_forwarded_for X-Forwarded-For头部，客户端真实IP
http_host 请求的Host头部
server_name 服务器名称
upstream 后端服务器的地址
upstream_response_time 后端服务器响应时间
upstream_status 后端服务器响应的HTTP状态码
geoip_country_code GeoIP国家代码（已注释）
geoip_country_name GeoIP国家名称（已注释）
geoip_city_name GeoIP城市名称（已注释）
"""
```

配置promtail收集nignx日志

```shell
# 进入配置文件中
vim /etc/promtail/config.yml
# 修改配置信息
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
- url: http://192.168.186.100:3100/loki/api/v1/push

scrape_configs:
- job_name: nginx
  static_configs:
  - targets:
      - localhost
    labels:
      job: nginxlogs
      host: 192.168.186.100
      #NOTE: Need to be modified to scrape any additional logs of the system.
      __path__: /var/log/nginx/*.log

# 修改日志目录权限
setfacl -R -m u:promtail:rx /var/log/nginx/
# 重启服务
systemctl restart promtail
```



