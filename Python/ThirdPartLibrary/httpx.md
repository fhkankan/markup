# httpx

[参考](https://www.python-httpx.org/)

HTTPX是Python 3的功能齐全的HTTP客户端，它提供同步和异步API，并支持HTTP / 1.1和HTTP / 2。

依赖：python3.6+

## 概述

安装

```
pip install httpx
```

同步请求

```shell
>>> import httpx
>>> r = httpx.get('https://www.example.org/')
>>> r
<Response [200 OK]>
>>> r.status_code
200
>>> r.headers['content-type']
'text/html; charset=UTF-8'
>>> r.text
'<!doctype html>\n<html>\n<head>\n<title>Example Domain</title>...'
```

异步请求

```shell
>>> import httpx
>>> async with httpx.AsyncClient() as client:
>>>     r = await client.get('https://www.example.org/')
>>> r
<Response [200 OK]>
```

## 请求

### 请求函数

- 不同请求类型

```python
import httpx

r = httpx.get('https://httpbin.org/get')
r = httpx.post('https://httpbin.org/post', data={'key': 'value'})
r = httpx.put('https://httpbin.org/put', data={'key': 'value'})
r = httpx.delete('https://httpbin.org/delete')
r = httpx.head('https://httpbin.org/get')
r = httpx.options('https://httpbin.org/g
```

- URL中传递参数

```python
# 方式一：k:v
params = {'key1': 'value1', 'key2': 'value2'}
r = httpx.get('https://httpbin.org/get', params=params)
# 方式二：k:[v]
params = {'key1': 'value1', 'key2': ['value2', 'value3']}
r = httpx.get('https://httpbin.org/get', params=params)

# 查看编译后的url
r.url
```

### 发送数据

- 表单编码数据

某些类型的HTTP请求（例如POST和PUT请求）可以在请求正文中包含数据。一种常见的添加方式是作为表单编码数据，用于HTML表单。

```shell
>>> data = {'key1': 'value1', 'key2': 'value2'}
>>> r = httpx.post("https://httpbin.org/post", data=data)
>>> print(r.text)
{
  ...
  "form": {
    "key2": "value2",
    "key1": "value1"
  },
  ...
}
```

表单编码数据还可以包含来自给定键的多个值。

```shell
>>> data = {'key1': ['value1', 'value2']}
>>> r = httpx.post("https://httpbin.org/post", data=data)
>>> print(r.text)
{
  ...
  "form": {
    "key1": [
      "value1",
      "value2"
    ]
  },
  ...
}
```

- 发送分段文件上传

您还可以使用HTTP分段编码上传文件：

```shell
>>> files = {'upload-file': open('report.xls', 'rb')}
>>> r = httpx.post("https://httpbin.org/post", files=files)
>>> print(r.text)
{
  ...
  "files": {
    "upload-file": "<... binary content ...>"
  },
  ...
}
```

您还可以通过使用项目元组作为文件值来显式设置文件名和内容类型：

```shell
# 如果将元组用作值，则它必须包含2到3个元素：
# 第一个元素是可选文件名，可以将其设置为None。
# 第二个元素可以是类似文件的对象，也可以是将以UTF-8自动编码的字符串。
# 可选的第三个元素可用于指定要上传的文件的MIME类型。如果未指定，HTTPX将尝试基于文件名猜测MIME类型，未知文件扩展名默认为“ application / octet-stream”。如果文件名明确设置为None，则HTTPX将不包含内容类型的MIME标头字段。
# 例1
files = {'upload-file': ('report.xls', open('report.xls', 'rb'), 'application/vnd.ms-excel')}
r = httpx.post("https://httpbin.org/post", files=files)
print(r.text)

# 例2
files = {'upload-file': (None, 'text content', 'text/plain')}
r = httpx.post("https://httpbin.org/post", files=files)
print(r.text)
```

- json编码数据

如果您只需要一个简单的键值数据结构，那么表单编码的数据就可以了。对于更复杂的数据结构，您通常需要改用JSON编码。

```shell
>>> data = {'integer': 123, 'boolean': True, 'list': ['a', 'b', 'c']}
>>> r = httpx.post("https://httpbin.org/post", json=data)
>>> print(r.text)
{
  ...
  "json": {
    "boolean": true,
    "integer": 123,
    "list": [
      "a",
      "b",
      "c"
    ]
  },
  ...
}
```

- 发送二进制请求数据

对于其他编码，应使用字节类型或产生字节的生成器。在上传二进制数据时，您可能还需要设置自定义的Content-Type标头。

### headers

要在传出请求中包含其他标头，请使用headers关键字参数：

```shell
>>> url = 'http://httpbin.org/headers'
>>> headers = {'user-agent': 'my-app/0.0.1'}
>>> r = httpx.get(url, headers=headers)
```

### 超时

HTTPX默认为所有网络操作都包括合理的超时，这意味着，如果未正确建立连接，则它应始终引发错误，而不是无限期地挂起。网络不活动的默认超时为五秒。

```python
httpx.get('https://github.com/', timeout=0.001)  # 修改超时时长
httpx.get('https://github.com/', timeout=None)  # 禁用超时行为
```

## 响应

### 响应数据

- unicode

HTTPX将自动处理将响应内容解码为Unicode文本。

```python
r = httpx.get('https://www.example.org/')
r.text  # 响应内容
r.encoding  # 响应编码
r.encoding = 'ISO-8859-1'  # 指定响应编码规则
```

- 二进制

对于非文本响应，响应内容也可以字节形式访问

```shell
>>> r.content
b'<!doctype html>\n<html>\n<head>\n<title>Example Domain</title>...'
```

任何gzip和deflate HTTP响应编码都会自动为您解码。如果安装了brotlipy，则还将支持brotli响应编码。

例如，要根据请求返回的二进制数据创建图像，可以使用以下代码：

```shell
>>> from PIL import Image
>>> from io import BytesIO
>>> i = Image.open(BytesIO(r.content))
```

- JSON

```shell
>>> r = httpx.get('https://api.github.com/events')
>>> r.json()
[{u'repository': {u'open_issues': 0, u'url': 'https://github.com/...' ...  }}]
```

### 响应状态码

```python
r = httpx.get('https://httpbin.org/get')
r.status_code  # 查看状态码
r.status_code == httpx.codes.OK  # 简便方式
r.raise_for_status()  # 正常状态返回None，4xx和5xx会抛出异常
```

### 响应headers

```python
# 查看响应头
r.headers  
# 查看响应头具体字段(不区分大小写)
r.headers['Content-Type']
r.headers.get('content-type')
```

### 流式响应

对于大型下载，您可能需要使用不将整个响应主体立即加载到内存中的流式响应。HTTPX将使用通用行结尾，将所有情况标准化为\ n。

二进制内容

```python
with httpx.stream("GET", "https://www.example.com") as r:
    for data in r.iter_bytes():
        print(data)
```

文本内容

```python
with httpx.stream("GET", "https://www.example.com") as r:
    for text in r.iter_text():
        print(text)
```

逐行播放文本

```python
with httpx.stream("GET", "https://www.example.com") as r:
    for line in r.iter_lines():
        print(line)
```

在某些情况下，您可能希望在不应用任何HTTP内容解码的情况下访问响应上的原始字节。在这种情况下，Web服务器已应用的任何内容编码（例如gzip，deflate或brotli）都不会自动解码。

```python
with httpx.stream("GET", "https://www.example.com") as r:
    for chunk in r.iter_raw():
        print(chunk)
```

如果您以任何上述方式使用流式响应，则`response.content`和`response.text`属性将不可用，并且在访问时会引发错误。但是，您还可以使用响应流功能来有条件地加载响应主体：

```python
with httpx.stream("GET", "https://www.example.com") as r:
    if r.headers['Content-Length'] < TOO_LONG:
        r.read()
        print(r.text)
```

## cookies

响应上的cookies：

```python
r = httpx.get('http://httpbin.org/cookies/set?chocolate=chip', allow_redirects=False)
r.cookies['chocolate']  # 响应上的cookies
```

请求携带cookies

```python
cookies = {"peanut": "butter"}
r = httpx.get('http://httpbin.org/cookies', cookies=cookies)  # 请求携带
r.json()  # 返回值{'cookies': {'peanut': 'butter'}}
```

Cookies在Cookies实例中返回，该实例是一种类似dict的数据结构，带有用于按其域或路径访问Cookies的其他API。

```python
>>> cookies = httpx.Cookies()
>>> cookies.set('cookie_on_domain', 'hello, there!', domain='httpbin.org')
>>> cookies.set('cookie_off_domain', 'nope.', domain='example.org')
>>> r = httpx.get('http://httpbin.org/cookies', cookies=cookies)
>>> r.json()
{'cookies': {'cookie_on_domain': 'hello, there!'}}
```

## 重定向和历史

默认情况下，除HEAD请求外，HTTPX都将跟随重定向。

响应的历史记录属性可用于检查所有后续重定向。它包含按照重定向顺序进行的任何重定向响应的列表。

例如，GitHub将所有HTTP请求重定向到HTTPS。

```shell
>>> r = httpx.get('http://github.com/')
>>> r.url
URL('https://github.com/')
>>> r.status_code
200
>>> r.history
[<Response [301 Moved Permanently]>]
```

您可以使用`allow_redirects`参数修改默认的重定向处理：

```shell
>>> r = httpx.get('http://github.com/', allow_redirects=False)
>>> r.status_code
301
>>> r.history
[]
```

如果您要发送HEAD请求，则可以使用它来启用重定向：

```shell
>>> r = httpx.head('http://github.com/', allow_redirects=True)
>>> r.url
'https://github.com/'
>>> r.history
[<Response [301 Moved Permanently]>]
```

## Client

> 若用过Requests，则可以使用`httpx.Client()`代替`request.Session()`
>
> 如果除了实验，一次性脚本或原型外，其他情况需要使用Client实例

- 特性

更有效的利用网络资源

```
普通的httpx.get等每次请求都建立一次连接，对client实例使用HTTP连接池，会对基础的TCP连接进行重用，带来了性能的改进（减少了请求之间的延迟（无握手）。减少CPU使用率和往返次数。减少网络拥塞。）
```

Cookie跨请求的持久性。

在所有传出请求中应用配置。

通过HTTP代理发送请求。

使用HTTP / 2。

- 基本方式

```python
# 方式一：建议使用with语句
with httpx.Client() as client:
    ...
    
# 方式二：手动关闭
client = httpx.Client()
try:
    ...
finally:
    client.close()
```

- 请求与配置

单独请求

```python
with httpx.Client() as client:
    r = client.get('https://example.com')  # 跟普通请求方法、参数和返回值一样
```

共用配置

```python
url = 'http://httpbin.org/headers'
headers = {'user-agent': 'my-app/0.0.1'}  # 公用配置
with httpx.Client(headers=headers) as client:
    r = client.get(url)

print(r.json()['headers']['User-Agent'])  # 'my-app/0.0.1'
```

合并配置

```python
# 不同key时合并
headers = {'X-Auth': 'from-client'}
params = {'client_id': 'client1'}
with httpx.Client(headers=headers, params=params) as client:
    headers = {'X-Custom': 'from-request'}
    params = {'request_id': 'request1'}
    r = client.get('https://example.com', headers=headers, params=params)

print(r.request.url)  # URL('https://example.com?client_id=client1&request_id=request1')
print(r.request.headers['X-Auth'])  # 'from-client'
print(r.request.headers['X-Custom'])  # 'from-request'

# 相同key时，请求级别优先
with httpx.Client(auth=('tom', 'mot123')) as client:
    r = client.get('https://example.com', auth=('alice', 'ecila123'))
    
_, _, auth = r.request.headers['Authorization'].partition(' ')
import base64
print(base64.b64decode(auth))  # b'alice:ecila123'
```

其他仅客户端配置选项

```python
with httpx.Client(base_url='http://httpbin.org') as client:
    r = client.get('/headers')
    
print(r.request.url)  # URL('http://httpbin.org/headers')
```

- 超时

```python
# 设置超时时长
httpx.get('http://example.com/api/v1/example', timeout=10.0)
with httpx.Client() as client:
    client.get("http://example.com/api/v1/example", timeout=10.0)
    
    
# 取消超时行为
httpx.get('http://example.com/api/v1/example', timeout=None)
with httpx.Client() as client:
    client.get("http://example.com/api/v1/example", timeout=None)

# 默认超时client
client = httpx.Client()              # 5s
client = httpx.Client(timeout=10.0)  # 设置为10s
client = httpx.Client(timeout=None)  # 取消超时行为

# 配置超时行为
# A client with a 60s timeout for connecting, and a 10s timeout elsewhere.
# connect_timeout/read_timeout/write_timeout/pool_timeout
timeout = httpx.Timeout(10.0, connect_timeout=60.0)
client = httpx.Client(timeout=timeout)
response = client.get('http://example.com/')
```

- 在python的web apps中调用

可以配置Httpx客户端以使用WSGI协议直接调用Python Web应用程序。

这对于两个主要用例特别有用：在测试用例中使用httpx作为客户端。在测试期间或开发/登台环境中模拟外部服务。

```python
# 这是一个与Flask应用程序集成的示例
from flask import Flask
import httpx


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

with httpx.Client(app=app, base_url="http://testserver") as client:
    r = client.get("/")
    assert r.status_code == 200
    assert r.text == "Hello World!"
```

对于某些更复杂的情况，您可能需要自定义WSGI调度。

这使您可以：

1. 通过设置`raise_app_exceptions = False`，检查500错误响应，而不是引发异常。
2. 通过设置`script_name`（WSGI）将WSGI应用程序安装在子路径中。
3. 通过设置`remote_addr`（WSGI），将给定的客户端地址用于请求。

```python
# Instantiate a client that makes WSGI requests with a client IP of "1.2.3.4".
dispatch = httpx.WSGIDispatch(app=app, remote_addr="1.2.3.4")
with httpx.Client(dispatch=dispatch, base_url="http://testserver") as client:
    ...
```

## Request

为了最大程度地控制通过网络发送的内容，HTTPX支持构建显式的Request实例

```python
request = httpx.Request("GET", "https://example.com")
```

要将Request实例跨网络分发，请创建Client实例并使用`.send()`：

```python
with httpx.Client() as client:
    response = client.send(request)
    ...
```

如果需要以默认的“合并参数”所不支持的方式混合客户端级别和请求级别的选项，则可以使用`.build_request()`，然后对Request实例进行任意修改。例如：

```python
headers = {"X-Api-Key": "...", "X-Client-ID": "ABC123"}

with httpx.Client(headers=headers) as client:
    request = client.build_request("GET", "https://api.example.com")

    print(request.headers["X-Client-ID"])  # "ABC123"

    # Don't send the API key for this particular request.
    del request.headers["X-Api-Key"]

    response = client.send(request)
    ...
```

## 代理

HTTPX支持通过请求参数通过proxies参数设置HTTP代理的方法。例如，将所有HTTP流量转发到http://127.0.0.1:3080，将所有HTTPS流量转发到http://127.0.0.1:3081，您的代理配置应如下所示：

```python
proxies = {
    "http": "http://127.0.0.1:3080",
    "https": "http://127.0.0.1:3081"
}
with httpx.Client(proxies=proxies) as client:
    ...
```

凭证可以以标准方式作为URL的一部分传递

```
http://username:password@127.0.0.1:3080
```

可以为特定的协议和地址，地址的所有协议，协议的所有地址或所有请求配置代理。在确定用于给定请求的代理配置时，将使用相同的顺序。

```python
proxies = {
    "http://example.com":  "...",  # Host+Scheme
    "all://example.com":  "...",  # Host
    "http": "...",  # Scheme
    "all": "...",  # All
}
with httpx.Client(proxies=proxies) as client:
    ...
    proxy = "..."  # Shortcut for {'all': '...'}
    with httpx.Client(proxies=proxy) as client:
        ...
```

默认情况下，`httpx.Proxy`将作为`http://...`请求的转发代理，并为`https://`请求建立`CONNECT` TCP隧道。无论代理URL是http还是https，此设置都不会改变。

可以将代理配置为具有不同的行为，例如转发或隧穿所有请求

```python
proxy = httpx.Proxy(
    url="https://127.0.0.1",
    mode="TUNNEL_ONLY"  # May be "TUNNEL_ONLY" or "FORWARD_ONLY". Defaults to "DEFAULT".
)
with httpx.Client(proxies=proxy) as client:
    # This request will be tunneled instead of forwarded.
    r = client.get("http://example.com")
```

## 认证

在发出请求或实例化客户端时，可以使用auth参数传递要使用的身份验证方案。

auth参数可能是以下其中之一
```
- 用户名/密码的二元组，用于基本身份验证。
- httpx.BasicAuth()或httpx.DigestAuth()的实例。
- 可调用的，接受请求并返回经过身份验证的请求实例。
- httpx.Auth的子类。
```
- 基本认证

要提供基本身份验证凭据，请将2个元组的纯文本str或bytes对象作为auth参数传递给请求函数：

```python
httpx.get("https://example.com", auth=("my_user", "password123"))
```

- 摘要身份验证

要提供用于Digest身份验证的凭据，您需要使用纯文本用户名和密码作为参数实例化DigestAuth对象。然后可以将该对象作为auth参数传递给上述请求方法：

```python
auth = httpx.DigestAuth("my_user", "password123")
httpx.get("https://example.com", auth=auth)
```

- `httpx.Auth`

允许创建涉及一个或多个请求的身份验证流。

```python
class MyCustomAuth(httpx.Auth):
    def __init__(self, token):
        self.token = token

    def auth_flow(self, request):
        # Send the request, with a custom `X-Authentication` header.
        request.headers['X-Authentication'] = self.token
        yield request
```

如果身份验证流程需要一个以上的请求，则可以发出多个收益，并在每种情况下获取响应...

```python
class MyCustomAuth(httpx.Auth):
    def __init__(self, token):
        self.token = token

    def auth_flow(self, request):
      response = yield request
      if response.status_code == 401:
          # If the server issues a 401 response then resend the request,
          # with a custom `X-Authentication` header.
          request.headers['X-Authentication'] = self.token
          yield request
```

自定义身份验证类被设计为不执行任何I / O，因此它们可以与同步和异步客户端实例一起使用。

如果要实现要求请求主体的身份验证方案，则需要使用`require_request_body`属性在类上进行指示。然后，您将可以访问`.auth_flow()`方法内的`request.content`。

```python
class MyCustomAuth(httpx.Auth):
    requires_request_body = True

    def __init__(self, token):
        self.token = token

    def auth_flow(self, request):
      response = yield request
      if response.status_code == 401:
          # If the server issues a 401 response then resend the request,
          # with a custom `X-Authentication` header.
          request.headers['X-Authentication'] = self.sign_request(...)
          yield request

    def sign_request(self, request):
        # Create a request signature, based on `request.method`, `request.url`,
        # `request.headers`, and `request.content`.
        ...
```

同样，如果要实现需要访问响应主体的方案，请使用`require_response_body`属性。然后，您将能够访问响应主体属性和方法，例如`response.content`，`response.text`，`response.json()`等。

```python
class MyCustomAuth(httpx.Auth):
    requires_response_body = True

    def __init__(self, access_token, refresh_token, refresh_url):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.refresh_url = refresh_url

    def auth_flow(self, request):
        request.headers["X-Authentication"] = self.access_token
        response = yield request

        if response.status_code == 401:
            # If the server issues a 401 response, then issue a request to
            # refresh tokens, and resend the request.
            refresh_response = yield self.build_refresh_request()
            self.update_tokens(refresh_response)

            request.headers["X-Authentication"] = self.access_token
            yield request

    def build_refresh_request(self):
        # Return an `httpx.Request` for refreshing tokens.
        ...

    def update_tokens(self, response):
        # Update the `.access_token` and `.refresh_token` tokens
        # based on a refresh response.
        data = response.json()
        ...
```

## SSL
通过HTTPS发出请求时，HTTPX需要验证所请求主机的身份。为此，它使用由受信任的证书颁发机构（CA）交付的SSL证书捆绑包（也称为CA捆绑包）。

- 更改验证默认值

默认情况下，HTTPX使用Certifi提供的CA捆绑软件。在大多数情况下，这是您想要的，即使某些高级情况可能要求您使用一组不同的证书。

```python
import httpx

# 如果要使用自定义CA捆绑包，则可以使用verify参数。
r = httpx.get("https://example.org", verify="path/to/client.pem")

# 您还可以禁用SSL验证...
r = httpx.get("https://example.org", verify=False)
```

- 客户端实例上的SSL配置

client.get（...）方法和其他请求方法不支持根据每个请求更改SSL设置。如果在不同情况下需要不同的SSL设置，则应使用一个以上的客户端实例，每个实例均具有不同的设置。然后，每个客户端将在该池中的所有连接上使用具有特定固定SSL配置的隔离连接池。

```python
client = httpx.Client(verify=False)
```

- 向本地服务器发出HTTPS请求

向本地服务器（例如在本地主机上运行的开发服务器）发出请求时，通常将使用未加密的HTTP连接。

如果确实需要建立与本地服务器的HTTPS连接（例如，测试仅HTTPS服务），则需要创建并使用自己的证书。

```python
# 1. 使用trustme-cli生成一对服务器密钥/证书文件和一个客户端证书文件。
# 2. 启动本地服务器时，传递服务器密钥/证书文件。（这取决于您使用的特定Web服务器。例如，Uvicorn提供了--ssl-keyfile和--ssl-certfile选项。）
# 3. 告诉HTTPX使用存储在client.pem中的证书
r = httpx.get("https://localhost:8000", verify="/tmp/client.pem")
```

## 异步支持

### 发出异步请求

> 将IPython或Python 3.8+与`python -m asyncio`配合使用可交互地尝试此代码，因为它们支持在控制台中执行async / await表达式。

```shell
>>> async with httpx.AsyncClient() as client:
>>>     r = await client.get('https://www.example.com/')
>>> r
<Response [200 OK]>
```

### API变化

请求

```python
# 请求方法都是异步的，因此对于以下所有情况，都应使用response = await client.get（...）样式
AsyncClient.get(url, ...)
AsyncClient.options(url, ...)
AsyncClient.head(url, ...)
AsyncClient.post(url, ...)
AsyncClient.put(url, ...)
AsyncClient.patch(url, ...)
AsyncClient.delete(url, ...)
AsyncClient.request(method, url, ...)
AsyncClient.send(request, ...)
```

开关clients

```python
# 上下文管理器
async with httpx.AsyncClient() as client:
    ...
    
# 手动处理
client = httpx.AsyncClient()
...
await client.aclose()
```

流式响应

```python
# AsyncClient.stream（method，url，...）方法是一个异步上下文块。
client = httpx.AsyncClient()
async with client.stream('GET', 'https://www.example.com/') as response:
    async for chunk in response.aiter_bytes():
        ...

# 异步响应流方法
Response.aread()-用于有条件地读取流块中的响应。Response.aiter_bytes()-用于以字节形式流式传输响应内容。Response.aiter_text()-用于将响应内容作为文本流式传输。Response.aiter_lines()-用于将响应内容作为文本流传输。Response.aiter_raw()-用于流传输原始响应字节，而无需应用内容解码。Response.aclose()-用于关闭响应。您通常不需要这样做，因为.stream块在退出时会自动关闭响应。
```

流式请求

```python
# 当发送带有AsyncClient实例的流请求主体时，应使用异步字节生成器，而不是字节生成器
async def upload_bytes():
    ...  # yield byte content

await client.post(url, data=upload_bytes())
```

### 支持的异步环境

HTTPX支持asyncio或trio作为异步环境。它将自动检测这两个中的哪一个用作套接字操作和并发原语的后端。

```python
# Asyncio,python内置库,用async/await语法编写并发代码
import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://www.example.com/')
        print(response)

asyncio.run(main())

# Trio,围绕结构化并发原则进行设计。
import httpx
import trio

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://www.example.com/')
        print(response)

trio.run(main)
```

### python web apps调用

就像`httpx.Client`允许您直接调用WSGI Web应用程序一样，`httpx.AsyncClient`类也允许您直接调用ASGI Web应用程序。

```python
# 让我们以这个Starlette应用程序为例
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route


async def hello():
    return HTMLResponse("Hello World!")


app = Starlette(routes=[Route("/", hello)])

# 直接针对应用发出请求
>>> import httpx
>>> async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
...     r = await client.get("/")
...     assert r.status_code == 200
...     assert r.text == "Hello World!"
```

对于某些更复杂的情况，您可能需要自定义ASGI调度。这使您可以：

1. 通过设置`raise_app_exceptions = False`，检查500错误响应，而不是引发异常。
2. 通过设置`root_path`将ASGI应用程序安装在子路径上。
3. 通过设置`client`，将给定的客户端地址用于请求。

```python
# Instantiate a client that makes ASGI requests with a client IP of "1.2.3.4",
# on port 123.
dispatch = httpx.ASGIDispatch(app=app, client=("1.2.3.4", 123))
async with httpx.AsyncClient(dispatch=dispatch, base_url="http://testserver") as client:
    ...
```

### Unix域sockets

异步客户端提供了对通过uds参数通过Unix域套接字进行连接的支持。当向绑定到套接字文件而不是IP地址的服务器发出请求时，此功能很有用。

这是一个请求Docker Engine API的示例：

```python
import httpx


async with httpx.AsyncClient(uds="/var/run/docker.sock") as client:
    # This request will connect through the socket file.
    resp = await client.get("http://localhost/version")
```

该功能当前在同步客户端中不可用。

## HTTP/2

HTTPX客户端提供HTTP / 2支持，目前仅异步客户端可用。默认情况下，不启用HTTP / 2支持

开启功能

```python
client = httpx.AsyncClient(http2=True)
...

# 上下文
async with httpx.AsyncClient(http2=True) as client:
    ...
```

检查版本

```python
client = httpx.AsyncClient(http2=True)
response = await client.get(...)
print(response.http_version)  # "HTTP/1.0", "HTTP/1.1", or "HTTP/2".
```

