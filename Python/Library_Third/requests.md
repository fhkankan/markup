# requests

requests 的底层实现其实就是 urllib3

Requests的文档非常完备

开源地址：<https://github.com/kennethreitz/requests>

中文文档 API： <http://docs.python-requests.org/zh_CN/latest/index.html>

## 安装

```python
pip install requests
```

## GET

```python
response = requests.get("http://www.baidu.com/")

# 也可以这么写
response = requests.request("get", "http://www.baidu.com/")
```

### 添加headers和查询参数

如果想添加 headers，可以传入`headers`参数来增加请求头中的headers信息。如果要将参数放在url中传递，可以利用 `params` 参数。

```python
import requests

kw = {'wd':'长城'}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}

# params 接收一个字典或者字符串的查询参数，字典类型自动转换为url编码，不需要urlencode()
response = requests.get("http://www.baidu.com/s?", params = kw, headers = headers)

# 查看响应内容，response.text 返回的是Unicode格式的数据
print response.text

# 查看响应内容，response.content返回的字节流数据
print respones.content

# 查看完整url地址
print response.url

# 查看响应头部字符编码
print response.encoding

# 查看响应码
print response.status_code
```

## POST

```python
response = requests.post("http://www.baidu.com/", data = data)
```

### 传入data数据

data的常见格式：

- application/x-www-form-urlencoded 

浏览器的原生 form 表单，如果header中不设置 Content-Type 属性，默认以 `application/x-www-form-urlencoded` 方式提交数据。

```python
requests.post(url='',data={'key1':'value1','key2':'value2'},headers={'Content-Type':'application/x-www-form-urlencoded'})
```

样例

```python
import requests

url = 'http://httpbin.org/post'
# headers={'Content-Type':'application/x-www-form-urlencoded'}
d = {'key1': 'value1', 'key2': 'value2'} 
r = requests.post(url, data=d)
print r.text
```

- multipart/form-data 

上传文件用的表单

```python
requests.post(url='',data={'key1':'value1','key2':'value2'},headers={'Content-Type':'multipart/form-data'})
```

需要文件

```python
from requests_toolbelt import MultipartEncoder
import requests

url = 'http://httpbin.org/post'
m = MultipartEncoder(
    fields={'field0': 'value', 'field1': 'value',
            'field2': ('filename', open('file.py', 'rb'), 'text/plain')}
    )
h={'Content-Type': m.content_type}
r = requests.post(url, data=m, headers=h)
```

不需要文件

```python
from requests_toolbelt import MultipartEncoder
import requests

url = 'http://httpbin.org/post'
m = MultipartEncoder(fields={'field0': 'value', 'field1': 'value'})
h = {'Content-Type': m.content_type}
r = requests.post(url, data=m, headers=h
```

- application/json 

传入json格式

```python
requests.post(url='',data=json.dumps({'key1':'value1','key2':'value2'}),headers={'Content-Type':'application/json'})
```

样例

```python
import requests

url = 'http://httpbin.org/post'
# headers={'Content-Type':'application/json'}
d_s = json.dumps({'key1': 'value1', 'key2': 'value2'})
r = requests.post(url, data=d_s)
print r.text
```

- text/xml 

传入xml格式

```python
requests.post(url='',data='<?xml  ?>',headers={'Content-Type':'text/xml'})
```

- binary

二进制文件

```python
requests.post(url='',files={'file':open('test.xls','rb')},headers={'Content-Type':'binary'})
```

样例

```python
import requests

url = 'http://httpbin.org/post'
# headers={'Content-Type':'binary'}
files = {'file': open('report.txt', 'rb')}
r = requests.post(url, files=files)
print r.text
```

### 模拟浏览器

```python
import requests

formdata = {
    "type":"AUTO",
    "i":"i love python",
    "doctype":"json",
    "xmlVersion":"1.8",
    "keyfrom":"fanyi.web",
    "ue":"UTF-8",
    "action":"FY_BY_ENTER",
    "typoResult":"true"
}

url = "http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null"

headers={ "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}

response = requests.post(url, data = formdata, headers = headers)

print response.text

# 如果是json文件可以直接显示
print response.json()
```

## 代理

-  proxies参数

你可以通过为任意请求方法提供 `proxies` 参数来配置单个请求：

```python
import requests

# 根据协议类型，选择不同的代理
proxies = {"http": "http://12.34.56.79:9527"}

response = requests.get("http://www.baidu.com", proxies = proxies)
print response.text
```

也可以通过本地环境变量 `HTTP_PROXY` 和 `HTTPS_PROXY` 来配置代理：

```python
export HTTP_PROXY="http://12.34.56.79:9527"
export HTTPS_PROXY="https://12.34.56.79:9527"
```

- 私密代理

```python
import requests

# 如果代理需要使用HTTP Basic Auth，可以使用下面这种格式：
proxy = { "http": "mr_mao_hacker:sffqry9r@61.158.163.130:16816" }

response = requests.get("http://www.baidu.com", proxies = proxy)

print response.text
```

## 验证

```python
import requests

auth=('test', '123456')

response = requests.get('http://192.168.199.107', auth = auth)

print response.text
```

## SSL

```python
import requests
# 检查证书
r = requests.get("https://www.baidu.com/", verify=True)
# 忽略证书
r = requests.get("https://www.12306.cn/mormhweb/", verify = False)
print r.text
```

## Cookies

```python
import requests

response = requests.get("http://www.baidu.com/")

# 7. 返回CookieJar对象:
cookiejar = response.cookies

# 8. 将CookieJar转为字典：
cookiedict = requests.utils.dict_from_cookiejar(cookiejar)

print cookiejar

print cookiedict
```

## Session

```python
import requests

# 1. 创建session对象，可以保存Cookie值
ssion = requests.session()

# 2. 处理 headers
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36"}

# 3. 需要登录的用户名和密码
data = {"email":"mr_mao_hacker@163.com", "password":"alarmchime"}  

# 4. 发送附带用户名和密码的请求，并获取登录后的Cookie值，保存在ssion里
ssion.post("http://www.renren.com/PLogin.do", data = data)

# 5. ssion包含用户登录后的Cookie值，可以直接访问那些登录后才可以访问的页面
response = ssion.get("http://www.renren.com/410043129/profile")

# 6. 打印响应内容
print response.text
```

## 实例

```python
#!/usr/bin/env python
# coding: utf-8
import requests


def requests_base_use():

    url = 'http://www.baidu.com'
    headers = {}
    # 1.get
    params = {}
    # 1/url自动转码
    # 2/多了params的参数
    response = requests.get(url, params=params, headers=headers)
    print response.text

    # 二进制流
    print response.content

    # 必须是返回的json文件
    print response.json()

    # 2.post
    formdata = {}
    requests.post(url, data=formdata, headers=headers)

    # 3.ssl
    requests.get(url, verify=False)

    # 4.proxy
    proxy = {'http': "ip:port"}
    requests.get(url, proxies=proxy)

    # 5.cookie
    # 生成一个可以自动保存cookie对象
    session = requests.session()
    session.post(url, data=formdata)
    session.get(url)

    # 6.webauth
    auth = ('user', 'pwd')
    requests.get(url, auth=auth)


if __name__ == "__main__":
    requests_base_use()
```

