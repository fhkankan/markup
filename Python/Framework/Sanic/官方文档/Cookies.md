# Cookies

Cookie是保存在用户浏览器中的数据。Sanic可以读取和写入cookie，它们都存储为键值对。

>警告
>
>客户可以自由更改Cookies。因此，您不能仅将登录信息之类的数据原样存储在cookie中，因为它们可以由客户端自由更改。为确保客户端不会伪造或篡改您存储在Cookie中的数据，请使用类似 [itsdangerous](https://pythonhosted.org/itsdangerous/)的内容对数据进行加密签名。

## 读取cookies

可以通过`Request`对象的`cookies`字典访问用户的Cookie。

```python
from sanic.response import text

@app.route("/cookie")
async def test(request):
    test_cookie = request.cookies.get('test')
    return text("Test cookie set to: {}".format(test_cookie))
```

## 写入cookies

返回响应时，可以在`Response`对象上设置cookie。

```python
from sanic.response import text

@app.route("/cookie")
async def test(request):
    response = text("There's a cookie up in this response")
    response.cookies['test'] = 'It worked!'
    response.cookies['test']['domain'] = '.gotta-go-fast.com'
    response.cookies['test']['httponly'] = True
    return response
```

## 删除cookies

Cookies可以通过语义或显式删除。

```python
from sanic.response import text

@app.route("/cookie")
async def test(request):
    response = text("Time to eat some cookies muahaha")

    # This cookie will be set to expire in 0 seconds
    del response.cookies['kill_me']

    # This cookie will self destruct in 5 seconds
    response.cookies['short_life'] = 'Glad to be here'
    response.cookies['short_life']['max-age'] = 5
    del response.cookies['favorite_color']

    # This cookie will remain unchanged
    response.cookies['favorite_color'] = 'blue'
    response.cookies['favorite_color'] = 'pink'
    del response.cookies['favorite_color']

    return response
```

响应cookie可以像字典值一样设置，并具有以下参数可用：

| 参数       | type     | desc                                                 |
| ---------- | -------- | ---------------------------------------------------- |
| `expires`  | datetime | Cookie在客户端浏览器上失效的时间                     |
| `path`     | string   | 此Cookie应用于的URL的子集。默认为/                   |
| `comment`  | string   | 评论（元数据）                                       |
| `domain`   | string   | 指定cookie对其有效的域。明确指定的域必须始终以点开头 |
| `max-age`  | number   | Cookie应存活的秒数                                   |
| `secure`   | boolean  | 指定是否仅通过HTTPS发送cookie                        |
| `httponly` | Boolean  | 指定是否无法用Javascript读取cookie                   |

