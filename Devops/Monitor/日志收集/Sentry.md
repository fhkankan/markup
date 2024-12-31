# Sentry

[参考1](http://www.projectsedu.com/2019/12/11/centos7下搭建sentry错误日志服务器/)

[参考2](https://www.jianshu.com/p/ca4ad23a2dd6)

[参考3](https://www.cnblogs.com/Shadow3627/p/10767023.html)

Sentry是一个实时事件的日志聚合平台。它专门监测错误并提取所有有用信息用于分析，不再麻烦地依赖用户反馈来定位问题。

Sentry发展多年，几乎没有同类产品可与其媲美。它能**覆盖大部分的主流编程语言与框架**，很适合应用到实际生产环境中采集异常日志。

最近我在设计持续交付流程过程时，公司一位前辈提到这个工具与用法。简单搭建并使用之后，基本确定在CD的灰度发布环节应用Sentry：若在灰度过程中获取到异常则触发灰度结束，将可能出现的异常由
 “上线-客户发现问题- 反馈问题-运维手动回滚”
 变为
 “灰度-Sentry捕获异常-自动停止灰度”，杜绝了回滚带来的不好形象，同时也能缩短问题发现的周期。

官方[git地址](https://github.com/getsentry/sentry)

官方安装[git地址](https://github.com/getsentry/onpremise)

## 安装

通过docker安装sentry，前提是安装好docker、docker-compose、git

- 安装docker

```shell
# 卸载已有的docker 
yum remove docker docker-common docker-selinux docker-engine

# 安装docker的依赖
yum install -y yum-utils device-mapper-persistent-data lvm2

# 安装docker-ce
yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
yum install docker-ce
 
# 启动docker后台服务
service docker start

# 测试运行
 docker run hello-world

# 设置开机启动
sudo systemctl enable docker
```

- 安装docker-compose

```shell
# 下载
curl -L https://get.daocloud.io/docker/compose/releases/download/1.25.0/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose

# 更改权限
chmod +x /usr/local/bin/docker-compose

# 测试
docker-compose version
```

- 安装sentry

```shell
# 安装git 
yum install git

# 下载onpremise
git clone https://github.com/getsentry/onpremise.git

# 安装
cd onpremise
./install.sh

# 启动
docker-compose up -d

# 创建账号
docker-compose run --rm web createuser
```

## 访问

```
http://localhost:9000/
```

> 注意
>
> 启动顺序为：woker->cron->web

## 配置

账户、团队、权限、邮箱

## 集成django

[参考](https://docs.sentry.io/platforms/python/django/)

安装sentry-sdk

```
pip install --upgrade 'sentry-sdk==0.13.5'
```

要配置SDK，请使用settings.py文件中的Django集成对其进行初始化：

```python
# settings.py
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn="https://<key>@sentry.io/<project>",
    integrations=[DjangoIntegration()]
)
```

您可以通过创建触发错误的路由来轻松验证Sentry安装：

```python
from django.urls import path

def trigger_error(request):
    division_by_zero = 1 / 0

urlpatterns = [
    path('sentry-debug/', trigger_error),
    # ...
]
```

访问此路线将触发一个错误，Sentry将捕获该错误。

- 报告其他状态码

在某些情况下，将404 Not Found和其他错误（未捕获的异常（500内部服务器错误））报告给Sentry可能很有意义。您可以通过为这些状态代码编写自己的Django视图来实现。例如：

```python
# urls.py

handler404 = 'mysite.views.my_custom_page_not_found_view'

# views.py

from django.http import HttpResponseNotFound
from sentry_sdk import capture_message


def my_custom_page_not_found_view(*args, **kwargs):
    capture_message("Page not found!", level="error")

    # return any response here, e.g.:
    return HttpResponseNotFound("Not found")
```

您发送给Sentry的错误消息将附加通常的请求数据。有关更多信息，请参阅Django文档中的 [Customizing Error Views](https://docs.djangoproject.com/en/2.0/topics/http/views/#customizing-error-views)。