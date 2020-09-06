#Scrapy部署

```
# 服务端
pip install scrapyd

# 客户端
pip install scrapyd-client

# 配置服务端conf
bind:0.0.0.0

# 开启服务
scrapyd

# 配置项目中的部署文件
以前：
[deploy]
#url = http://localhost:6800/
project = Tencent
修改为：
[deploy:scrapyd_Tencent]
url = http://localhost:6800/
project = Tencent

# 将爬虫项目部署到服务器
cd 项目路径
scrapyd-deploy scrapyd_Tencent -p Tencent

# 开启爬虫
curl http://localhost:6800/schedule.json -d project=Tencent -d spider=tencent

# 终止爬虫
curl http://localhost:6800/cancel.json -d project=Tencent -d job=jobid的值
```

# Scrapy-redis部署

```
1. master slaver 链接成功

2.代码
# 导库
scrapy_redis.spiders import RedisSpiders
redis_key = ""

3.配置文件

```

