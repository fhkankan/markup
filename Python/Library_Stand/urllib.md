# urlib

urllib提供了一系列用于操作URL的功能。

urllib提供的功能就是利用程序去执行各种HTTP请求。如果要模拟浏览器完成特定功能，需要把请求伪装成浏览器。伪装的方法是先监控浏览器发出的请求，再根据浏览器的请求头来伪装，`User-Agent`头就是用来标识浏览器的。

## Get

urllib的`request`模块可以非常方便地抓取URL内容，也就是发送一个GET请求到指定的页面，然后返回HTTP的响应：

例如，对豆瓣的一个URL`https://api.douban.com/v2/book/2129650`进行抓取，并返回响应：

```
from urllib import request

with request.urlopen('https://api.douban.com/v2/book/2129650') as f:
    data = f.read()
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', data.decode('utf-8'))
```

可以看到HTTP响应的头和JSON数据：

```
Status: 200 OK
Server: nginx
Date: Tue, 26 May 2015 10:02:27 GMT
Content-Type: application/json; charset=utf-8
Content-Length: 2049
Connection: close
Expires: Sun, 1 Jan 2006 01:00:00 GMT
Pragma: no-cache
Cache-Control: must-revalidate, no-cache, private
X-DAE-Node: pidl1
Data: {"rating":{"max":10,"numRaters":16,"average":"7.4","min":0},"subtitle":"","author":["廖雪峰编著"],"pubdate":"2007-6",...}
```

如果我们要想模拟浏览器发送GET请求，就需要使用`Request`对象，通过往`Request`对象添加HTTP头，我们就可以把请求伪装成浏览器。例如，模拟iPhone 6去请求豆瓣首页：

```
from urllib import request

req = request.Request('http://www.douban.com/')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
with request.urlopen(req) as f:
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', f.read().decode('utf-8'))
```

这样豆瓣会返回适合iPhone的移动版网页：

```
...
    <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0">
    <meta name="format-detection" content="telephone=no">
    <link rel="apple-touch-icon" sizes="57x57" href="http://img4.douban.com/pics/cardkit/launcher/57.png" />
...
```

## Post

如果要以POST发送一个请求，只需要把参数`data`以bytes形式传入。

我们模拟一个微博登录，先读取登录的邮箱和口令，然后按照weibo.cn的登录页的格式以`username=xxx&password=xxx`的编码传入：

```
from urllib import request, parse

print('Login to weibo.cn...')
email = input('Email: ')
passwd = input('Password: ')
login_data = parse.urlencode([
    ('username', email),
    ('password', passwd),
    ('entry', 'mweibo'),
    ('client_id', ''),
    ('savestate', '1'),
    ('ec', ''),
    ('pagerefer', 'https://passport.weibo.cn/signin/welcome?entry=mweibo&r=http%3A%2F%2Fm.weibo.cn%2F')
])

req = request.Request('https://passport.weibo.cn/sso/login')
req.add_header('Origin', 'https://passport.weibo.cn')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
req.add_header('Referer', 'https://passport.weibo.cn/signin/login?entry=mweibo&res=wel&wm=3349&r=http%3A%2F%2Fm.weibo.cn%2F')

with request.urlopen(req, data=login_data.encode('utf-8')) as f:
    print('Status:', f.status, f.reason)
    for k, v in f.getheaders():
        print('%s: %s' % (k, v))
    print('Data:', f.read().decode('utf-8'))
```

如果登录成功，我们获得的响应如下：

```
Status: 200 OK
Server: nginx/1.2.0
...
Set-Cookie: SSOLoginState=1432620126; path=/; domain=weibo.cn
...
Data: {"retcode":20000000,"msg":"","data":{...,"uid":"1658384301"}}
```

如果登录失败，我们获得的响应如下：

```
...
Data: {"retcode":50011015,"msg":"\u7528\u6237\u540d\u6216\u5bc6\u7801\u9519\u8bef","data":{"username":"example@python.org","errline":536}}
```

## Handler

如果还需要更复杂的控制，比如通过一个Proxy去访问网站，我们需要利用`ProxyHandler`来处理，示例代码如下：

```
proxy_handler = urllib.request.ProxyHandler({'http': 'http://www.example.com:3128/'})
proxy_auth_handler = urllib.request.ProxyBasicAuthHandler()
proxy_auth_handler.add_password('realm', 'host', 'username', 'password')
opener = urllib.request.build_opener(proxy_handler, proxy_auth_handler)
with opener.open('http://www.example.com/login.html') as f:
    pass
```
## 抓取网页资源

### 最简单

```
import urllib.request
response = urllib.request.urlopen('http://python.org/')
html = response.read()
```

### request

```
import urllib.request

req = urllib.request.Request('http://python.org/')
response = urllib.request.urlopen(req)
the_page = response.read()
```

### 发送数据

```
import urllib.parse
import urllib.request

url = 'http://localhost/login.php'
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
values = {
'act' : 'login',
'login[email]' : 'yzhang@i9i8.com',
'login[password]' : '123456'
}

data = urllib.parse.urlencode(values)
req = urllib.request.Request(url, data)
req.add_header('Referer', 'http://www.python.org/')
response = urllib.request.urlopen(req)
the_page = response.read()

print(the_page.decode("utf8"))
```

### 发送数据和header

```
import urllib.parse
import urllib.request

url = 'http://localhost/login.php'
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
values = {
'act' : 'login',
'login[email]' : 'yzhang@i9i8.com',
'login[password]' : '123456'
}
headers = { 'User-Agent' : user_agent }

data = urllib.parse.urlencode(values)
req = urllib.request.Request(url, data, headers)
response = urllib.request.urlopen(req)
the_page = response.read()

print(the_page.decode("utf8"))
```

### http错误

```
import urllib.request

req = urllib.request.Request('http://www.111cn.net ')
try:
urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
print(e.code)
print(e.read().decode("utf8"))
```

### 异常处理

> 1

```
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
req = Request("http://www.111cn.net /")
try:
response = urlopen(req)
except HTTPError as e:
print('The server couldn't fulfill the request.')
print('Error code: ', e.code)
except URLError as e:
print('We failed to reach a server.')
print('Reason: ', e.reason)
else:
print("good!")
print(response.read().decode("utf8"))
```

> 2

```
from urllib.request import Request, urlopen
from urllib.error import  URLError
req = Request("http://www.111cn.net /")
try:
response = urlopen(req)
except URLError as e:
if hasattr(e, 'reason'):
print('We failed to reach a server.')
print('Reason: ', e.reason)
elif hasattr(e, 'code'):
print('The server couldn't fulfill the request.')
print('Error code: ', e.code)
else:
print("good!")
print(response.read().decode("utf8"))
```

### HTTP认证

```
import urllib.request

# create a password manager
password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

# Add the username and password.
# If we knew the realm, we could use it instead of None.
top_level_url = "https://www.111cn.net /"
password_mgr.add_password(None, top_level_url, 'rekfan', 'xxxxxx')

handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

# create "opener" (OpenerDirector instance)
opener = urllib.request.build_opener(handler)

# use the opener to fetch a URL
a_url = "https://www.111cn.net /"
x = opener.open(a_url)
print(x.read())

# Install the opener.
# Now all calls to urllib.request.urlopen use our opener.
urllib.request.install_opener(opener)

a = urllib.request.urlopen(a_url).read().decode('utf8')
print(a)
```

### 使用代理

```
import urllib.request

proxy_support = urllib.request.ProxyHandler({'sock5': 'localhost:1080'})
opener = urllib.request.build_opener(proxy_support)
urllib.request.install_opener(opener)


a = urllib.request.urlopen("http://www.111cn.net ").read().decode("utf8")
print(a)
```

### 超时

```
import socket
import urllib.request

# timeout in seconds
timeout = 2
socket.setdefaulttimeout(timeout)

# this call to urllib.request.urlopen now uses the default timeout
# we have set in the socket module
req = urllib.request.Request('http://www.111cn.net /')
a = urllib.request.urlopen(req).read()
print(a)
```





