# 优鸿

## App_django

```
FROM registry.cn-beijing.aliyuncs.com/uhongedu/basic_v2:1.0

# 镜像加速器
RUN echo -e "http://mirrors.aliyun.com/alpine/v3.7/main\nhttp://mirrors.aliyun.com/alpine/v3.7/community" > /etc/apk/repositories

RUN apk update

# 设置香港时区
RUN apk add -U tzdata \
    && rm -rf /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# bash在entrypoint.py脚本中使用，uwsgi/uwsgi-python3  uwsgi uwsgi-python uwsgi-python3 python相关uwsgi服务。
# build-base linux-headers pcre-dev openssl pip3安装uwsgi的依赖包
# 升级pip
#  && pip3 install --upgrade pip \
RUN apk add --no-cache python3 python3-dev git openssh openssl bash curl gcc  \
    build-base linux-headers pcre-dev \
    # 升级pip
  && pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip \
  # 资源SDK
  && pip3 install --no-cache-dir -i --trusted-host http://pypi.uhongedu.com/packages/ziyuan_sdk-0.0.14.tar.gz \
  # aly日志
  && pip3 install -i --trusted-host http://pypi.uhongedu.com/packages/ucm_sdk-0.0.1.tar.gz \
  && pip3 install -i --trusted-host http://pypi.uhongedu.com/packages/yh_sls-0.0.5.tar.gz \
  && pip3 install -i --trusted-host http://pypi.uhongedu.com/packages/yh_sdk-0.0.4.tar.gz 

RUN apk add --no-cache --virtual .build-deps libc-dev libffi-dev openssl-dev \
  && pip3 install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    # 数据库连接引擎
    PyMySQL==0.8.0 \
    # jsonfield
    jsonfield==2.0.2 \
    # django
    Django==2.0.4 \
    # oss2
    oss2==2.4.0 \
    aliyunsdkcore==1.0.2 \
    aliyun-python-sdk-sts==3.0.0 \
    # celery
    celery==4.1.0 \
    # 序列化
    msgpack==0.5.6 \
    # kombu
    kombu==4.1.0 \
    xlrd==1.1.0 \
    xlwt==1.3.0 \
    # pypinyin
    pypinyin==0.30.0 \
    # 异步请求
    grequests==0.3.0 \
    # yaml
    PyYAML==3.12 \
    # uwsgi
    uwsgi==2.0.17 \
    # 代码分析工具
    line-profiler==2.1.2 \
    memory-profiler==0.52.0 \
    # memcache
    python-memcached==1.57 \
    # redis
    django-redis==4.10.0 \
    # Twisted
    Twisted==18.4.0 \
    # 阿里云日志
    aliyun-log-python-sdk==0.6.30.4 \
    uwsgidecorators==1.1.0 \
        # excel
    xlsxwriter==1.0.9 \
    # python-alipay-sdk
    python-alipay-sdk==1.7.1 \
    # weixin
    weixinpy==0.0.9 \
    # orm
    orator==0.9.7 \
    # cmd
    fire==0.1.3 \
    # https://github.com/ipipdotnet/datx-python
    ipip-datx \
    # 消息服务mqtt
    paho-mqtt==1.4.0 \
    # 阿里云ACM配置中心 
    acm-sdk-python \
    apscheduler==3.5.3 \

  && apk del .build-deps \
  && rm -rf /var/lib/apt/lists/*
```

## uhong_celery

```
FROM registry.cn-beijing.aliyuncs.com/uhongedu/basic_v2:1.0
RUN echo -e "https://mirrors.aliyun.com/alpine/v3.7/main\nhttps://mirrors.aliyun.com/alpine/v3.7/community" > /etc/apk/repositories
RUN apk update

# 设置东八区时区
RUN apk add -U tzdata \
    && rm -rf /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

RUN apk add --no-cache python3 git openssh bash curl yasm ffmpeg \
  # 升级pip
  && pip3 install --upgrade pip

RUN apk add --no-cache --virtual .build-deps python3-dev gcc musl-dev libc-dev libffi-dev openssl-dev \
  && pip install --no-cache-dir -i  https://mirrors.aliyun.com/pypi/simple/ \
      oss2==2.4.0 \
    aliyunsdkcore==1.0.2 \
    aliyun-python-sdk-sts==3.0.0 \
      kombu==4.2.2.post1 \
      tornado==5.1.1 \
    # 数据库连接工具
    pymysql==0.7.11 \
    # orm
    orator==0.9.7 \
    lorm==1.0.11 \
    # 序列化工具
    msgpack==0.5.0 \
    # celery
    celery==4.1.1 \
    # 异步库
    gevent==1.2.2 \
    # 任务管理工具
    flower==0.9.2 \
    # 异步请求
    grequests==0.3.0 \
    # yaml
    PyYAML==3.12 \
        # 阿里云日志
        aliyun-log-python-sdk==0.6.30.4 \
        uwsgidecorators==1.1.0 \
        # redis
        redis==2.10.6 \
        # dockerpy
        docker==3.4.1 \
  && apk del .build-deps \
  && rm -rf /var/lib/apt/lists/* \
    # 安装 ziyuan_sdk
    && pip install -i -trusted-host http://pypi.uhongedu.com/packages/ziyuan_sdk-0.0.13.tar.gz \
    # 安装 yh-sls
    && pip install -i --trusted-host http://pypi.uhongedu.com/packages/yh_sls-0.0.4.tar.gz \
  # 安装 yh-sdk
  && pip install -i --trusted-host http://pypi.uhongedu.com/packages/yh_sdk-0.0.4.tar.gz

# 安装supervisor(python3版本) 管理工具
RUN git clone https://github.com/Supervisor/supervisor.git \
  && cd supervisor \
  && python3 setup.py install \
  && cd .. \
  && rm -rf supervisor
```

## Dsj_celery

```
FROM registry.cn-beijing.aliyuncs.com/uhongedu/django2.0_dsj:base-opencv4.1.0-pyzbar-alpine

RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip
  # 资源SDK
  && pip3 install --no-cache-dir -i --trusted-host http://pypi.uhongedu.com/packages/ziyuan_sdk-0.0.5.tar.gz \
  # aly日志
  && pip3 install -i --trusted-host http://pypi.uhongedu.com/packages/yh_sls-0.0.4.tar.gz \
    && pip3 install -i --trusted-host http://pypi.uhongedu.com/packages/yh_sdk-0.0.4.tar.gz \
    && pip3 install --trusted-host --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ \
    # 数据库连接引擎
    PyMySQL==0.8.0 \
    # orm
    orator==0.9.7 \
    lorm==1.0.11 \
    # 连接rabbitmq的工具
    kombu==4.2.2.post1 \
    # 序列化工具
    msgpack==0.5.0 \
    # celery
    celery==4.1.1 \
    # 异步库
    gevent==1.2.2 \
    # 任务管理工具
    flower==0.9.2 \
    # 异步请求
    grequests==0.3.0 \
    # yaml
    PyYAML==3.12 \
        # 安装tornado
        tornado==5.1.1 \
        # 阿里云日志
        aliyun-log-python-sdk==0.6.30.4 \
        uwsgidecorators==1.1.0 \
        # redis
        redis==2.10.6 \
        # dockerpy
        docker==3.4.1 \
    numpy==1.17.0 \
    supervisor==4.0.4
```

