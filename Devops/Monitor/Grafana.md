# grafana

## 概述

Grafana是一款用Go语言开发的开源数据可视化工具，可以做数据监控和数据统计，带有告警功能。

主要功能：

面板：从热图到直方图，从图表到地图。Grafana 提供快速灵活的可视化功能，供您以任何所需方式可视化数据。

插件：使用 Grafana 插件关联您的工具和团队。数据源插件可通过 API 关联现有数据源，并实时渲染数据，而无需迁移或提取数据。

警报：借助 Grafana Alerting ，您可以在简洁的界面中创建、管理和禁用所有警报，轻松实现警报的整合和集中

转换：通过“转换”功能，您可以对多个查询和数据源执行重命名、汇总、合并和计算操作。

注释：使用来自不同数据源的事件来注释图表。将鼠标悬停在事件上可显示完整的事件元数据和标签。

面板编辑器：可让您轻松配置、自定义和浏览所有面板，在统一的界面中为所有可视化内容设置数据选项。

## 安装

[文档](https://grafana.com/docs/grafana/latest/setup-grafana/installa)

下载安装包

```
wget https://dl.grafana.com/oss/release/grafana-6.0.1-1.x86_64.rpm
```

安装依赖

```
yum install initscripts fontconfig  
yum install freetype
yum install urw-fonts
```

安装grafana

```
rpm -Uvh grafana-6.0.1-1.x86_64.rpm
```

安装插件

```shell
# 使用grafana-cli工具安装

# 获取可用插件列表
grafana-cli plugins list-remote  

# 修改图形为饼状
grafana-cli plugins install grafana-piechart-panel
# 安装其他图形插件
grafana-cli plugins install grafana-clock-panel
# 钟表形展示
grafana-cli plugins install briangann-gauge-panel
# 字符型展示
grafana-cli plugins install natel-discrete-panel
# 服务器状态
grafana-cli plugins install vonage-status-panel
```

卸载插件

```shell
grafana-cli plugins uninstall vonage-status-panel
# 安装和卸载后需要重启grafana才能够生效
```

启动/重启/关闭

```shell
# 启动
service grafana-server start
# 停止
service grafana-server stop
# 重启
service grafana-server restart
# 加入开机自启动
chkconfig --add grafana-server on
```

启动测试

```
默认用户密码：admin/admin, 
访问地址: [http://grafana服务地址:3000](http://localhost:3000/)
如果出现登录界面，代表安装启动成功
```

## 告警通知

- 开启告警

grafana只有graph支持告警通知。
grafana的告警通知渠道有很多种，像Email、Teams、钉钉等都有支持。
在`grafana.ini`中开启告警：

```shell
#################################### Alerting ############################
[alerting]
# Disable alerting engine & UI features
enabled = true   #开启
# Makes it possible to turn off alert rule execution but alerting UI is visible
execute_alerts = true  #开启
# Default setting for new alert rules. Defaults to categorize error and timeouts as alerting. (alerting, keep_state)
;error_or_timeout = alerting
# Default setting for how Grafana handles nodata or null values in alerting. (alerting, no_data, keep_state, ok)
;nodata_or_nullvalues = no_data
# Alert notifications can include images, but rendering many images at the same time can overload the server
# This limit will protect the server from render overloading and make sure notifications are sent out quickly
;concurrent_render_limit = 5
```

- 邮件通知

要能发送邮件通知，首先需要在配置文件`grafana.ini`中配置邮件服务器等信息：

```shell
#################################### SMTP / Emailing ##########################
[smtp]
enabled = true #是否允许开启
host =  #发送服务器地址，可以再邮箱的配置教程中找到：
user = 你的邮箱
# If the password contains # or ; you have to wrap it with trippel quotes. Ex """#password;"""
password = 这个密码是你开启smtp服务生成的密码
;cert_file =
;key_file =
skip_verify = true
from_address = 你的邮箱
from_name = Grafana
# EHLO identity in SMTP dialog (defaults to instance_name)
;ehlo_identity = dashboard.example.com
[emails]
;welcome_email_on_sign_up = false
```

在页面中配置邮件通知渠道。