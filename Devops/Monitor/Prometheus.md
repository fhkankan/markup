# Promethueus

## 概述

Prometheus 是一款基于时序数据库的开源监控告警系统，非常适合Kubernetes集群的监控。Prometheus的基本原理是通过HTTP协议周期性抓取被监控组件的状态，任意组件只要提供对应的HTTP接口就可以接入监控。不需要任何SDK或者其他的集成过程。这样做非常适合做虚拟化环境监控系统，比如VM、Docker、Kubernetes等。输出被监控组件信息的HTTP接口被叫做exporter 。目前互联网公司常用的组件大部分都有exporter可以直接使用，比如Varnish、Haproxy、Nginx、MySQL、Linux系统信息(包括磁盘、内存、CPU、网络等等)。

- 特点

```
支持多维数据模型：由度量名和键值对组成的时间序列数据
内置时间序列数据库TSDB
支持PromQL查询语言，可以完成非常复杂的查询和分析，对图表展示和告警非常有意义
支持HTTP的Pull方式采集时间序列数据
支持PushGateway采集瞬时任务的数据
支持服务发现和静态配置两种方式发现目标
支持接入Grafana
```

- 组件

```
- prometheus server 
是 Prometheus 组件中的核心部分，负责实现对监控数据的获取，存储以及查询。
- exporter 
简单说是采集端，通过 http 服务的形式保留一个 url 地址，prometheus server 通过 访问该 exporter 提供的 endpoint 端点，即可获取到需要采集的监控数据。
- Pushgateway
由于 Prometheus 数据采集采用 pull 方式进行设置的， 内置必须保证 prometheus server 和 对应的 exporter 必须通信，当网络情况无法直接满足时，可以使用 pushgateway 来进行中转， 可以通过 pushgateway 将内部网络数据主动 push 到 gateway 里面去，而 prometheus 采用 pull 方式拉取pushgateway 中数据。
- AlertManager
在 prometheus 中，支持基于 PromQL 创建告警规则，如果满足定义的规则，则会产生一条 告警信息，进入 AlertManager 进行处理。可以集成邮件，微信或者通过 webhook 自定义报 警。
```

- 原理

Prometheus直接从目标主机 上运行的代理程序（exporter） 中抓取指标，并将收集的样本集中存储在自己服务器上（主要以拉模式为主），也可以使用像 collectd_exporter 这样的插件推送指标，尽管这不是 Promethius 的默认行为，但在主机位于防火墙后面或位于安全策略限制打开端口的某些环境中它可能很有用。另外，后者可通过HTTP协议周期性抓取被监控组件的状态，任意组件只要提供对应的HTTP接口就可以接入监控。不需要任何SDK或者其他的集成过程。这样做非常适合做虚拟化环境监控系统，比如：VM、Docker、Kubernetes等；它以给定的时间间隔从已配置的目标收集指标，评估规则表达式，显示结果，并在发现某些情况为真时触发警报。

prometheus 负责从 pushgateway 和 job 中采集数据， 存储到后端 Storatge 中，可以通过 PromQL 进行查询， 推送 alerts 信息到 AlertManager。 AlertManager 根据不同的路由规则 进行报警通知。

- 优点

```
1>数据格式是Key/Value形式，简单、速度快；采用多维数据模型(由指标名称和键/值维集定义的timeseries)
2>timeseries收集是通过HTTP上的拉取（pull mode）模型进行，通过中间网关支持timeseries的推送，通过服务发现或静态配置来发现目标，监控数据的精细程度可达到秒级（数据采集精度高情况下，对磁盘消耗大，存在性能瓶颈，且不支持集群，但可以通过联邦能力进行扩展）；
3>不依赖分布式存储，数据直接保存在本地，单节点是自治的，可独立运行管理，可以不需要额外的数据库配置。但是如果对历史数据有较高要求，可以结合OpenTSDB；支持分层和水平联合。
4> 周边插件丰富，如果对监控要求不是特别严格的话，默认的几个成品插件已经足够使用；支持多种图形和仪表板。
5>本身基于数学计算模型，有大量的函数可用，可以实现很复杂的监控（故学习成本高，需要有一定数学思维，独有的数学命令行很难入门）；
6>可以嵌入很多开源工具的内部去进行监控，数据更可信。
7>使用PromQL，它是一种强大而灵活的查询语言，PromQL作为Prometheus强大的查询语言，可以灵活地处理监视数据。
```

- 局限

```
1.更多地展示的是 趋势性 的监控
Prometheus作为一个基于度量的系统，不适合存储事件或者日志等，它更多地展示的是趋势性的监控。如果用户需要数据的精准性（不足），可以考虑ELK或其他日志架构。另外，APM更适用于链路追踪的场景。
2.Prometheus本地不适合存储大量历史数据存储
Prometheus认为只有最近的监控数据才有查询的需要，所有Prometheus本地存储的设计初衷只是保存短期（如一个月）的数据，不会针对大量的历史数据进行存储。如果需要历史数据，则建议：使用Prometheus的远端存储，如：OpenTSDB、M3DB等。
3.成熟度没有 InfluxDB高
Prometheus在集群上不论是采用联邦集群还是采用Improbable开源的Thanos等方案，都没有InfluxDB成熟度高，需要解决很多细节上的技术问题（如耗尽CPU、消耗机器资源等问题），部分互联网公司拥有海量业务，出于集群的原因会考虑对单机免费但是集群收费的InfluxDB进行自主研发。
```

## 部署

### 安装ptometheus

下载解压

```
mkdir -pv /usr/local/soft/package && cd /usr/local/soft/package
wget https://github.com/prometheus/prometheus/releases/download/v2.29.2/prometheus-2.29.2.linux-amd64.tar.gz
tar -xf prometheus-2.29.2.linux-amd64.tar.gz -C /usr/local/soft
cd /usr/local/soft
mv prometheus-2.29.2.linux-amd64 prometheus
```

配置

```shell
# 进入文件
vim /usr/local/soft/prometheus/prometheus.yml
#全局配置
global:
  scrape_interval: 15s #每隔15秒向目标抓取一次数，默认为一分钟
  evaluation_interval: 15s #每隔15秒执行一次告警规则，默认为一分钟
  # scrape_timeout: 600s  #抓取数据的超时时间，默认为10s

#告警配置
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093	 #alertmanager所部署机器的ip和端口

#定义告警规则和阈值的yml文件
rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

#收集数据配置
#以下是Prometheus自身的一个配置.
scrape_configs:
  #这个配置是表示在这个配置内的时间序例，每一条都会自动添加上这个{job_name:"prometheus"}的标签.
  - job_name: "prometheus"
    # metrics_path defaults to '/metrics'
    # scheme defaults to 'http'.
    static_configs:			#静态配置
      - targets: ["localhost:9090"]

```

启停

```shell
cd /usr/local/soft/prometheus

#校验配置文件
./promtool check config ./prometheus.yml

#启动
nohup ./prometheus --config.file=./prometheus.yml \
--web.listen-address=0.0.0.0:9090 \
--web.enable-lifecycle \
--storage.tsdb.retention=90d \
--storage.tsdb.path=./data &

##启动参数介绍
--config.file      	   #加载prometheus的配置文件
--web.listen-address   #监听prometheus的web地址和端口
--web.enable-lifecycle #热启动参数，可以在不中断服务的情况下重启加载配置文件
--storage.tsdb.retention   #数据持久化的时间                         
--storage.tsdb.path        #数据持久化的保存路径

#停止
ps -ef | grep prometheus | grep  -v grep | awk '{print $2}' | xagrs kill -9
#或者
curl -XPOST http://localhost:9090/-/quit

#重载
curl -XPOST http://localhost:9090/-/reload
```

以systemd方式管理（非必选）

```shell
cat > /usr/lib/systemd/system/prometheus.service <<EOF
[Unit]
Description=The Prometheus Server
After=network.target
[Service]
ExecStart=/usr/local/soft/prometheus/prometheus \
  --config.file=/usr/local/soft/prometheus/prometheus.yml \
  --web.listen-address=0.0.0.0:9090 \
  --web.enable-lifecycle \
  --storage.tsdb.retention=90d \
  --storage.tsdb.path="/usr/local/soft/prometheus/data/"
Restart=on-failure
RestartSec=15s
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload && systemctl enable prometheus && systemctl restart prometheus
```

简单访问

```
http://IP:9090
```

### 安装node_exporter

下载解压

```
cd /usr/local/soft/package
wget https://github.com/prometheus/node_exporter/releases/download/v1.2.2/node_exporter-1.2.2.linux-amd64.tar.gz
tar -xf node_exporter-1.2.2.linux-amd64.tar.gz -C /usr/local/soft
cd /usr/local/soft
mv node_exporter-1.2.2.linux-amd64 node_exporter
```

启停

```shell
#启动
nohup ./node_exporter --web.listen-address=0.0.0.0:4220 &

##启动参数介绍
注意：相关启动的参数
--web.listen-address     #node_expoetrt暴露的端口
--collector.systemd	     #从systemd中收集
--collector.systemd.unit-whitelist   ##白名单，收集目标
		".+"         		      #从systemd中循环正则匹配单元
		"(docker|sshd|nginx).service"  #白名单，收集目标，收集参数node_systemd_unit_state
		
#停止
ps -ef | grep node_exporter | grep  -v grep | awk '{print $2}' | xagrs kill -9

```

以systemd方式管理（非必选）

```shell
cat > /usr/lib/systemd/system/node_exporter.service   <<EOF
[Unit]
Description=The node_exporter Server
After=network.target
[Service]
ExecStart=/usr/local/soft/node_exporter/node_exporter \
  --web.listen-address=0.0.0.0:4220 \
  --collector.systemd \
  --collector.systemd.unit-whitelist=(sshd|docker).service
Restart=on-failure
RestartSec=15s
SyslogIdentifier=node_exporter
[Install]
WantedBy=multi-user.target 
EOF

systemctl daemon-reload && systemctl enable node_exporter && systemctl restart node_exporter
```

### 安装alertmanager

下载解压

```
cd /usr/local/soft/package
wget https://github.com/prometheus/alertmanager/releases/download/v0.23.0/alertmanager-0.23.0.linux-amd64.tar.gz
tar -xf alertmanager-0.23.0.linux-amd64.tar.gz -C /usr/local/soft
cd /usr/local/soft
mv alertmanager-0.23.0.linux-amd64 alertmanager
```

配置

```shell
#global配置
global:
  resolve_timeout: 5m  #在报警恢复的时候不是立马发送的，在接下来的这个时间内，如果没有此报警信息触发，才发送报警恢复消息
  smtp_smarthost: 'smtp.exmail.qq.com:465' #发件人对应邮件提供商的smtp地址，此处为腾讯企业邮箱stmp配置
  smtp_from: 'xxx@company.com'          #发件人邮箱地址
  smtp_auth_username: 'xxx@company.com' #发件人的登陆用户名，默认和发件人地址一致
  smtp_auth_password: 'xxxxxxx'       #发件人的登陆密码，也可以是授权码。
  smtp_require_tls: false		      #是否需要tls协议，默认是true
#templates配置
templates:
- '/usr/local/soft/alertmanager/email.tmpl'	 #自定义通知的模板的目录或者文件
#route配置
route:						#每个输入警报进入根路由
  group_by: ['alertname','cluster','service']	#将传入的报警中有这些标签的分为一个组,比如, cluster=A 和 alertname=LatencyHigh 会分成一个组
  group_wait: 30s	#指分组创建多久后才可以发送压缩的警报，也就是初次发警报的延时,这样会确保第一次通知的时候, 有更多的报警被压缩在一起
  group_interval: 5m	#当第一个通知发送，等待多久发送压缩的警报
  repeat_interval: 1h	#如果报警发送成功, 等待多久重新发送一次
  receiver: 'email'	 #默认警报接收者
#receivers配置
receivers:
- name: 'email'		#警报名称
  email_configs:
  - to: 'xxx@xxx.com'		#接收警报的email
    send_resolved: true		#是否发送警报解除邮件
    html: '{{ template "email.htm" . }}'	#模板
    headers: { Subject: "{{ .CommonLabels.severity }} {{ .CommonAnnotations.summary }}" }	#标题
#报警抑制规则
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']	#通过上面的配置，可以在alertname相同的情况下，critaical的报警会抑制warning级别的报警信息。
#静默配置
#静默配置是通过web界面配置的，通常用于服务升级或者长时间的服务故障，确保在接下来的时间内不会在收到同样报警信息

```

启停

```shell
#启动
nohup ./alertmanager --config.file="alertmanager.yml" --web.listen-address=":9093" &

#停止
ps -ef |grep alertmanager |grep -v grep  |awk '{print $2}' | xargs kill -9

#重载
curl -XPOST http://localhost:9093/-/reload
```

以systemd方式管理（非必选）

```shell
cat > /usr/lib/systemd/system/alertmanager.service   <<EOF
[Unit]
Description=The Prometheus Server
After=network.target
[Service]
ExecStart=/usr/local/soft/alertmanager/alertmanager \
  --config.file=/usr/local/soft/alertmanager/alertmanager.yml \
  --web.listen-address=0.0.0.0:9093
Restart=on-failure
RestartSec=15s
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload && systemctl enable alertmanager && systemctl restart alertmanager
```

web访问

```
http://IP:9093
```

### 安装grafana

下载解压

```
cd /usr/local/soft/package
wget https://dl.grafana.com/enterprise/release/grafana-enterprise-8.1.2.linux-amd64.tar.gz
tar -xf grafana-enterprise-8.1.2.linux-amd64.tar.gz -C /usr/local/soft
cd /usr/local/soft
mv grafana-enterprise-8.1.2.linux-amd64 grafana
```

启停

```shell
#启动
nohup /usr/local/soft/grafana/bin/grafana-server &

#停止
ps -ef |grep grafana-server |grep -v grep  |awk '{print $2}' | xargs kill -9
```

web访问

```
http://IP:3000
```

## 监控mysql

### 安装mysqld_export

下载解压

```
mkdir -pv /usr/local/soft/package && cd /usr/local/soft/package
wget https://github.com/prometheus/mysqld_exporter/releases/download/v0.13.0/mysqld_exporter-0.13.0.linux-amd64.tar.gz
tar -xf mysqld_exporter-0.13.0.linux-amd64.tar.gz -C /usr/local/soft
cd  /usr/local/soft
mv mysqld_exporter-0.13.0.linux-amd64 mysqld_exporter
```

配置mysql监控账号

```shell
cd  /usr/local/soft/mysqld_exporter
vim my.cnf
[client]
user=root
password=123456
host=localhost
port=3306
#sock="/tmp/mysql_3306.sock"   #如果数据库需要sock文件就配置上，否则不需要此配置
```

启停

```shell
#启动
nohup /usr/local/soft/mysqld_exporter/mysqld_exporter \
	--web.listen-address=":9104" \
	--config.my-cnf=/usr/local/soft/mysqld_exporter/my-38.cnf &

#停止
ps -ef | grep mysqld_exporter | grep -v grep | awk '{print $2}' | xargs kill -9  
```

如果在一台服务器上，启动多个 mysqld_exporter ，可通过监听不同端口实现：

```shell
nohup /usr/local/soft/mysqld_exporter/mysqld_exporter \
	--web.listen-address=":9104" \
	--config.my-cnf=/usr/local/soft/mysqld_exporter/my-38.cnf &
	
nohup /usr/local/soft/mysqld_exporter/mysqld_exporter \
	--web.listen-address=":9105" \
	--config.my-cnf=/usr/local/soft/mysqld_exporter/my-22.cnf &
	
nohup /usr/local/soft/mysqld_exporter/mysqld_exporter \
--web.listen-address=":9106" \
--config.my-cnf=/usr/local/soft/mysqld_exporter/my-245.cnf &
```

### 配置prometheus

配置

```shell
vim /usr/local/soft/prometheus/prometheus.yml
global:
  scrape_interval: 15s .
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
           - 172.16.16.18:9093	#alertmanager地址

rule_files:		#报警规则文件
  - "/usr/local/soft/prometheus/rules/mysql_rule.yml"

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
  
  - job_name: "node-exporter"
    static_configs:
      - targets: ["localhost:4220"]

  - job_name: "mysql"	#配置 mysql job
    file_sd_configs:	#配置文件自动发现
      - files: 
        - /usr/local/soft/prometheus/conf.d/mysql_export.json

```

配置文件自动发现

```shell
cd /usr/local/soft/prometheus
mkdir conf.d

vim conf.d/mysql_export.json
[
    {
        "labels":{						#配置自定义标签
         	"desc": "172.16.16.38",
        	"group": "mysql",
        	"host_ip": "172.16.16.38",
        	"hostname": "mysql-master"
        },
        "targets": ["172.16.16.18:9104"]	#配置metrics地址
    },
    {
        "labels":{
            "desc": "172.16.16.22",
            "group": "mysql",
            "host_ip": "172.16.16.22",
            "hostname": "mysql-slave01"
        },
        "targets": ["172.16.16.18:9105"]
    },
    {
        "labels":{
            "desc": "172.16.16.245",
            "group": "mysql",
            "host_ip": "172.16.16.245",
            "hostname": "mysql-slave02"
        },
        "targets": ["172.16.16.18:9106"]
    }
]

```

配置告警规则

```shell
cd /usr/local/soft/prometheus
mkdir rules

vim rules/mysql_rule.yml
groups:
- name: mysqlAlerts
  rules:
  - alert: mysql告警
    expr: mysql_up{job="mysql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "{{$labels.desc}} mysql已停止运行超过1m"
      description: "{{$labels.desc}} mysql中断超过1m"
      message: "{{$labels.desc}} mysql已停止运行"
      console: "请检查{{$labels.desc}}节点的mysql是否正常"
  - alert: mysql slave节点IO线程异常
    expr: mysql_slave_status_slave_io_running{job="mysql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "{{$labels.desc}} mysql slave节点IO线程已停止运行超过1m"
      description: "{{$labels.desc}} mysql slave节点IO线程中断超过1m"
      message: "{{$labels.desc}} mysql slave节点IO线程已停止运行"
      console: "请检查{{$labels.desc}}节点的mysql是否正常"
  - alert: mysql slave节点SQL线程异常
    expr: mysql_slave_status_slave_sql_running{job="mysql"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "{{$labels.desc}} mysql slave节点SQL线程已停止运行超过1m"
      description: "{{$labels.desc}} mysql slave节点SQL线程中断超过1m"
      message: "{{$labels.desc}} mysql slave节点SQL线程已停止运行"
      console: "请检查{{$labels.desc}}节点的mysql是否正常"
```

### 配置altertmanager

#### 配置邮件告警

配置文件

```
cd /usr/local/soft/alertmanager

vim alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_smarthost: "smtp.qq.com:465"			#配置邮件告警
  smtp_from: "xxxx@qq.com"
  smtp_auth_username: "xxxx@qq.com"
  smtp_auth_password: "fafafafafafafa"
  smtp_require_tls: false
templates:
- '/usr/local/soft/alertmanager/email.tmpl'		#配置模板
route:
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 30m
  receiver: 'email'
receivers:
- name: 'email'
  email_configs:
  - to: 'xxx@company.com'					#配置邮件接收人
    send_resolved: true
    html: '{{ template "email.htm" . }}'	#模板
    headers: { Subject: "[{{ .Status }}]{{ .CommonLabels.severity }} {{ .CommonAnnotations.summary }}" }			 #标题
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']

```

创建模板文件

```shell
vim email.tmpl
{{ define "email.htm" }}
{{ range .Alerts.Resolved }}
<pre>
告警已解除！

历史告警信息如下：
</pre>
{{ end }}
{{ range .Alerts }}
<pre>
========start==========
告警程序: {{ .Labels.job }}
告警级别: {{ .Labels.severity }} 级别
告警类型: {{ .Labels.alertname }}
故障主机: {{ .Labels.desc }}
告警主题: {{ .Annotations.summary }}
告警详情: {{ .Annotations.description }}
处理方法: {{ .Annotations.console }}
触发时间: {{ (.StartsAt.Add 28800e9).Format "2006-01-02 15:04:05" }}
========end==========
</pre>
{{ end }}
{{ end }}
```

#### 配置钉钉通知

下载解压

```
cd /usr/local/soft/package
wget https://github.com/timonwong/prometheus-webhook-dingtalk/releases/download/v2.0.0/prometheus-webhook-dingtalk-2.0.0.linux-amd64.tar.gz
tar -xf prometheus-webhook-dingtalk-2.0.0.linux-amd64.tar.gz -C /usr/local/soft/
cd /usr/local/soft/
mv prometheus-webhook-dingtalk-2.0.0.linux-amd64 prometheus-webhook-dingtalk
```

创建钉钉自定义机器人

```
在电脑端钉钉的任一个群点击 群设置 --> 智能群助手 --> 添加机器人，进入后添加“自定义机器人”，然后按要求操作，获取 Webhook 和 加密串。
```

修改钉钉告警插件配置

```shell
cd /usr/local/soft/prometheus-webhook-dingtalk
cp config.example.yml config.yml

vim config.yml
## Request timeout
# timeout: 5s

## Uncomment following line in order to write template from scratch (be careful!)
#no_builtin_template: true

## Customizable templates path
#templates:
#  - contrib/templates/legacy/template.tmpl

## You can also override default template using `default_message`
## The following example to use the 'legacy' template from v0.3.0
#default_message:
#  title: '{{ template "legacy.title" . }}'
#  text: '{{ template "legacy.content" . }}'

## Targets, previously was known as "profiles"
##将webhook地址复制到url,加密串复制到secret
targets:
  webhook1:
    url: https://oapi.dingtalk.com/robot/send?access_token=b7a4392cacc6ac5962e8671486ffafa28aa14337e037455dbd11dc3e4a7b8db8
    # secret for signature
    secret: SEC4b4c021b752a0216ced1ba953a6ed78bf4202ca8c9106c6841a92ba502547a5f
    message:
      title: '{{ template "legacy.title" . }}'
      text: '{{ template "legacy.content" . }}'
      #mobiles: ['156xxxx8827', '189xxxx8325']
```

启动

```shell
cd /usr/local/soft/prometheus-webhook-dingtalk
nohup ./prometheus-webhook-dingtalk --config.file=./config.yml &
```

测试

```
curl http://localhost:8060/dingtalk/webhook1/send -H 'Content-Type: application/json' -d '{"msgtype": "text","text": {"content": "监控告警"}}'
```

配置alertmanager

```shell
cd /usr/local/soft/alertmanager/

vim alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 5m
  receiver: 'dingtalk'
receivers:						#配置钉钉告警
- name: 'dingtalk'
  webhook_configs:
  - url: 'http://localhost:8060/dingtalk/webhook1/send'
    send_resolved: true
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
```

结合两种方式告警的示例如下：

```shell
global:
  resolve_timeout: 5m
  smtp_smarthost: "smtp.qq.com:465"
  smtp_from: "1954938301@qq.com"
  smtp_auth_username: "1954938301@qq.com"
  smtp_auth_password: "wgwgolvivcogfabj"
  smtp_require_tls: false
templates:
- '/usr/local/soft/alertmanager/email.tmpl'
route:
  group_by: ['alertname']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 5m
  receiver: 'dingtalk'			#默认告警接收者
  routes:						#子路由
  - receiver: 'email'			
    match:
      severity: 'critical'		#标签severity为critical时触发
receivers:
- name: 'email'
  email_configs:
  - to: 'huy@ktpis.com'
    send_resolved: true
    html: '{{ template "email.htm" . }}'
    headers: { Subject: "[{{ .Status | title }}]{{ .CommonLabels.severity }} {{ .CommonAnnotations.summary }}" }
- name: 'dingtalk'
  webhook_configs:
  - url: 'http://localhost:8060/dingtalk/webhook1/send'
    send_resolved: true
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
```

