# Python2的urllib2库

```
urllib2 是 Python2.7 自带的模块(不需要下载，导入即可使用)

urllib2 官方文档：https://docs.python.org/2/library/urllib2.html

urllib2 源码：https://hg.python.org/cpython/file/2.7/Lib/urllib2.py

在 python3 中，urllib2 被改为urllib.request
```

## 基本使用

###urlopen

```
# urllib2_urlopen.py

# 导入urllib2 库
import urllib2

# 向指定的url发送请求，并返回服务器响应的类文件对象
# 参数是url地址
response = urllib2.urlopen("http://www.baidu.com")

# 类文件对象支持 文件对象的操作方法，如read()方法读取文件全部内容，返回字符串
html = response.read()

# 打印字符串
print html
```

###Request

```
# urllib2_request.py

import urllib2

# url 作为Request()方法的参数，构造并返回一个Request对象
# 参数还可以是data（默认空）：提交的Form表单数据，同时 HTTP 请求方法将从默认的 "GET"方式 改为 "POST"方式。
# headers（默认空）：参数为字典类型，包含了需要发送的HTTP报头的键值对。
request = urllib2.Request("http://www.baidu.com")

# Request对象作为urlopen()方法的参数，发送给服务器并接收响应
response = urllib2.urlopen(request)

html = response.read()

print html
```

###User-Agent

urllib2默认的User-Agent头为：Python-urllib/x.y （x和y 是Python 主.次 版本号，例如 Python-urllib/2.7）

```
#urllib2_useragent.py

import urllib2

url = "http://www.itcast.cn"

# IE 9.0 的 User-Agent，包含在 user_agent里
user_agent = {"User-Agent" : "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)"} 

#  url 连同 headers，一起构造Request请求，这个请求将附带 IE9.0 浏览器的User-Agent
request = urllib2.Request(url, headers = user_agent)

# 向服务器发送这个请求
response = urllib2.urlopen(request)

html = response.read()
print html
```

###添加Header信息

在 HTTP Request 中加入特定的 Header，来构造一个完整的HTTP请求消息。

```
# 添加/修改一个特定的header
Request.add_header()  
# 查看已有的header
Request.get_header()
```

- 添加一个特定的header

```
# urllib2_headers.py

import urllib2

url = "http://www.itcast.cn"

#IE 9.0 的 User-Agent
user_agent = {"User-Agent" : "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)"} 
request = urllib2.Request(url, headers = user_agent)

#也可以通过调用Request.add_header() 添加/修改一个特定的header
request.add_header("Connection", "keep-alive")

# 也可以通过调用Request.get_header()来查看header信息
# request.get_header(header_name="Connection")

response = urllib2.urlopen(request)

print response.code     #可以查看响应状态码
html = response.read()

print html
```

- 随机添加/修改User-Agent

```
# urllib2_add_headers.py

import urllib2
import random

url = "http://www.itcast.cn"

ua_list = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
    "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6"
]

user_agent = random.choice(ua_list)

request = urllib2.Request(url)

#也可以通过调用Request.add_header() 添加/修改一个特定的header
request.add_header("User-Agent", user_agent)

# get_header()的字符串参数，第一个字母大写，后面的全部小写
request.get_header("User-agent")

response = urllib2.urlopen(request)

html = response.read()
print html
```

## GET/POST

### URL编码转换

```
urllib 和 urllib2 都是接受URL请求的相关模块,但是
1.urllib 模块仅可以接受URL，不能创建 设置了headers 的Request 类实例；
2.urllib 提供 urlencode方法用来产生GET查询字符串，而 urllib2 则没有。

# 编码：将key:value键值对，转换成"key=value"
urllib.urlencode()

# 解码工作
urllib.unquote()
```

### GET

####直接对url进行获取

GET请求一般用于我们向服务器获取数据，我们用百度搜索`传智播客`：<https://www.baidu.com/s?wd=传智播客>，浏览器的url会跳转成:<https://www.baidu.com/s?wd=%E4%BC%A0%E6%99%BA%E6%92%AD%E5%AE%A2>

```
# urllib2_get.py

import urllib      #负责url编码处理
import urllib2

url = "http://www.baidu.com/s"
word = {"wd":"传智播客"}
# 转换成url编码格式（字符串）
# 1.中文转换，2.字典转换
word = urllib.urlencode(word) 
newurl = url + "?" + word    # url首个分隔符就是 ?

headers={ "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}

request = urllib2.Request(newurl, headers=headers)

response = urllib2.urlopen(request)

print response.read()
```
####获取AJAX加载的内容

AJAX请求一般返回给网页的是JSON文件，只要对AJAX请求地址进行POST或GET，就能返回JSON数据了。

```
# demo1

url = "https://movie.douban.com/j/chart/top_list?type=11&interval_id=100%3A90&action=&"

headers={"User-Agent": "Mozilla...."}

# 变动的是这两个参数，从start开始往后显示limit个
formdata = {
    'start':'0',
    'limit':'10'
}
data = urllib.urlencode(formdata)

request = urllib2.Request(url + data, headers = headers)
response = urllib2.urlopen(request)

print response.read()


# demo2

url = "https://movie.douban.com/j/chart/top_list?"
headers={"User-Agent": "Mozilla...."}

# 处理所有参数
formdata = {
    'type':'11',
    'interval_id':'100:90',
    'action':'',
    'start':'0',
    'limit':'10'
}
data = urllib.urlencode(formdata)

request = urllib2.Request(url + data, headers = headers)
response = urllib2.urlopen(request)

print response.read()
```

### POST

