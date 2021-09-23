# JWT验证

[参考](https://www.jianshu.com/p/c69f08ca056d)

## 概述

`Json web token (JWT)`, 根据官网的定义，是为了在网络应用环境间传递声明而执行的一种基于JSON的开放标准（(RFC 7519).该token被设计为紧凑且安全的，特别适用于分布式站点的单点登录（SSO）场景。JWT的声明一般被用来在身份提供者和服务提供者间传递被认证的用户身份信息，以便于从资源服务器获取资源，也可以增加一些额外的其它业务逻辑所必须的声明信息，该token也可直接被用于认证，也可被加密。

- 使用场景

一次性验证

```
比如用户注册后需要发一封邮件让其激活账户，通常邮件中需要有一个链接，这个链接需要具备以下的特性：能够标识用户，该链接具有时效性（通常只允许几小时之内激活），不能被篡改以激活其他可能的账户…这种场景就和 jwt 的特性非常贴近，jwt 的 payload 中固定的参数：iss 签发者和 exp 过期时间正是为其做准备的。
```

restful api的无状态认证

```
使用 jwt 来做 restful api 的身份认证也是值得推崇的一种使用方案。客户端和服务端共享 secret；过期时间由服务端校验，客户端定时刷新；签名信息不可被修改…spring security oauth jwt 提供了一套完整的 jwt 认证体系，以笔者的经验来看：使用 oauth2 或 jwt 来做 restful api 的认证都没有大问题，oauth2 功能更多，支持的场景更丰富，后者实现简单（后者的实现还是需要覆盖认证的场景。如果不认证,jwt就没有用户标识）。
```

单点登录+会话管理

```
需要考虑如下问题
1.secret设计
jwt唯一存储在服务端的只有一个secret，这个secret应该设计成和用户相关的，而不是一个所有用户公用的统一值。这样可以有效的避免一些注销和修改密码时遇到的窘境。

2.注销和修改密码
a.清空客户端存储的jwt，这样用户访问时就不会携带jwt，服务端就认为用户需要重新登录。这是一个典型的假注销，对于用户表现出退出的行为，实际上这个时候携带对应的 jwt 依旧可以访问系统。
b.清空或修改服务端的用户对应的secret，这样在用户注销后，jwt 本身不变，但是由于 secret 不存在或改变，则无法完成校验。这也是为什么将 secret 设计成和用户相关的原因。
c.借助第三方存储自己管理 jwt 的状态，可以以 jwt 为 key，实现去 redis 一类的缓存中间件中去校验存在性。方案设计并不难，但是引入 redis 之后，就把无状态的 jwt 硬生生变成了有状态了，违背了 jwt 的初衷。实际上这个方案和 session 都差不多了。

3.续签问题
a.每次请求刷新jwt，简单但是有性能问题，不优雅
b.完善refreshToken。借鉴 oauth2 的设计，返回给客户端一个 refreshToken，允许客户端主动刷新 jwt。一般而言，jwt 的过期时间可以设置为数小时，而 refreshToken 的过期时间设置为数天。建议使用oauth2方案。
c.使用redis记录独立的过期时间。类似session存储
```

- 特点

优点

```
- 性能
体积小，因而传输速度快
- 传输方式多样
可以通过URL/POST参数/HTTP头部等方式传输
- 严格的结构化
它自身（在 payload 中）就包含了所有与用户相关的验证消息，如用户可访问路由、访问有效期等信息，服务器无需再去连接数据库验证信息的有效性，并且 payload 支持为你的应用而定制化。
- 支持跨域访问
Cookie是不允许垮域访问的，这一点对Token机制是不存在的，前提是传输的用户认证信息通过HTTP头传输.
- 无状态(也称：服务端可扩展行)
Token机制在服务端不需要存储session信息，因为Token 自身包含了所有登录用户的信息，只需要在客户端的cookie或本地介质存储状态信息
- 适用CDN
可以通过内容分发网络请求你服务端的所有资料（如：javascript，HTML,图片等），而你的服务端只要提供API即可.
- 去耦
不需要绑定到一个特定的身份验证方案。Token可以在任何地方生成，只要在你的API被调用的时候，你可以进行Token生成调用即可.
- 适用接口跨平台
当你的客户端是一个原生平台（iOS, Android，Windows 8等）时，Cookie是不被支持的（你需要通过Cookie容器进行处理），这时采用Token认证机制就会简单得多
- CSRF
因为不再依赖于Cookie，所以你就不需要考虑对CSRF（跨站请求伪造）的防范
```
缺点

```
- Token有长度限制
- Token不能撤销
- 需要token有失效时间限制(exp)
```

注意

```
- JWT默认是不加密，但也是可以加密的。生成原始 Token 以后，可以用密钥再加密一次。
- JWT不加密的情况下，不能将秘密数据写入JWT。
- 由于服务器不保存 session 状态，因此无法在使用过程中废止某个token，或者更改 token的权限。也就是说，一旦JWT签发了，在到期之前就会始终有效，除非服务器部署额外的逻辑。
- JWT 本身包含了认证信息，一旦泄露，任何人都可以获得该令牌的所有权限。为了减少盗用，JWT 的有效期应该设置得比较短。对于一些比较重要的权限，使用时应该再次对用户进行认证。
- 为了减少盗用，JWT 不应该使用 HTTP 协议明码传输，要使用 HTTPS 协议传输。
```

- 过程

```
1. 首先，前端通过Web表单将自己的用户名和密码发送到后端的接口。这一过程一般是一个HTTP POST请求。建议的方式是通过SSL加密的传输（https协议），从而避免敏感信息被嗅探。
2. 后端核对用户名和密码成功后，将用户的id等其他信息作为JWT Payload（负载），将其与头部分别进行Base64编码拼接后签名，形成一个JWT。
3. 后端将JWT字符串作为登录成功的返回结果返回给前端。前端可以将返回的结果保存在localStorage或sessionStorage上，退出登录时前端删除保存的JWT即可。
4. 前端在每次请求时将JWT放入HTTP Header中的Authorization位。(解决XSS和XSRF问题)
5. 后端检查是否存在，如存在验证JWT的有效性。例如，检查签名是否正确；检查Token是否过期；检查Token的接收方是否是自己（可选）。
6. 验证通过后后端使用JWT中包含的用户信息进行其他逻辑操作，返回相应结果
```

## 原理

JWT是Auth0提出的通过对JSON进行加密签名来实现授权验证的方案，编码之后的JWT看起来是这样的一串字符：

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ
```

由 `.` 分为三段，通过解码可以得到：

- 头部（Header）

```
// 包括类别（typ）、加密算法（alg）；
{
  "alg": "HS256",
  "typ": "JWT"
}
```

jwt的头部包含两部分信息：
```
- 声明类型，这里是jwt
- 声明加密的算法 通常直接使用 HMAC SHA256
```
然后将头部进行base64加密（该加密是可以对称解密的)，构成了第一部分。

```
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9
```

- 载荷（payload）

载荷就是存放有效信息的地方。这些有效信息包含三个部分：
> 标准中注册声明
```
iss: 该JWT的签发者，是否使用是可选的；
sub: 该JWT所面向的用户，是否使用是可选的；
aud: 接收该JWT的一方，是否使用是可选的；
exp: 什么时候过期，这里是一个Unix时间戳，是否使用是可选的；
iat: 在什么时候签发的(UNIX时间)，是否使用是可选的；
nbf：如果当前时间在nbf里的时间之前，则Token不被接受；一般都会留一些余地，比如几分钟；是否使用是可选的；
jti: jwt的唯一身份标识，主要用来作为一次性token，从而回避重放攻击。
```

> 公共的声明 

公共的声明可以添加任何的信息，一般添加用户的相关信息或其他业务需要的必要信息.但不建议添加敏感信息，因为该部分在客户端可解密。

> 私有的声明 

私有声明是提供者和消费者所共同定义的声明，一般不建议存放敏感信息，因为base64是对称解密的，意味着该部分信息可以归类为明文信息。

示例

```javascript
// 包括需要传递的用户信息；
{ "iss": "Online JWT Builder", 
  "iat": 1416797419, 
  "exp": 1448333419, 
  "aud": "www.gusibi.com", 
  "sub": "uid", 
  "nickname": "goodspeed", 
  "username": "goodspeed", 
  "scopes": [ "admin", "user" ] 
}
```

将上面的JSON对象进行`base64编码`可以得到下面的字符串。

```
eyJpc3MiOiJPbmxpbmUgSldUIEJ1aWxkZXIiLCJpYXQiOjE0MTY3OTc0MTksImV4cCI6MTQ0ODMzMzQxOSwiYXVkIjoid3d3Lmd1c2liaS5jb20iLCJzdWIiOiIwMTIzNDU2Nzg5Iiwibmlja25hbWUiOiJnb29kc3BlZWQiLCJ1c2VybmFtZSI6Imdvb2RzcGVlZCIsInNjb3BlcyI6WyJhZG1pbiIsInVzZXIiXX0
```

- 签名（signature）

jwt的第三部分是一个签证信息，这个签证信息由三部分组成：

```
- header (base64后的)
- payload (base64后的)
- secret
```

根据alg算法与私有秘钥进行加密得到的签名字串，这一段是最重要的敏感信息，只能在服务端解密

```
HMACSHA256(  
    base64UrlEncode(header) + "." +
    base64UrlEncode(payload),
    SECREATE_KEY
)
```

将上面的两个编码后的字符串都用句号.连接在一起（头部在前），就形成了:

```
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJKb2huIFd1IEpXVCIsImlhdCI6MTQ0MTU5MzUwMiwiZXhwIjoxNDQxNTk0NzIyLCJhdWQiOiJ3d3cuZXhhbXBsZS5jb20iLCJzdWIiOiJqcm9ja2V0QGV4YW1wbGUuY29tIiwiZnJvbV91c2VyIjoiQiIsInRhcmdldF91c2VyIjoiQSJ9
```

最后，我们将上面拼接完的字符串用HS256算法进行加密。在加密的时候，我们还需要提供一个密钥（secret）。如果我们用 `secret` 作为密钥的话，那么就可以得到我们加密后的内容:

```
pq5IDv-yaktw6XEa5GEv07SzS9ehe6AcVSdTj0Ini4o
```

> 签名的目的

签名实际上是对头部以及载荷内容进行签名。所以，如果有人对头部以及载荷的内容解码之后进行修改，再进行编码的话，那么新的头部和载荷的签名和之前的签名就将是不一样的。而且，如果不知道服务器加密的时候用的密钥的话，得出来的签名也一定会是不一样的。这样就能保证token不会被篡改。

将这三部分用.连接成一个完整的字符串,构成了最终的jwt:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJPbmxpbmUgSldUIEJ1aWxkZXIiLCJpYXQiOjE0MTY3OTc0MTksImV4cCI6MTQ0ODMzMzQxOSwiYXVkIjoid3d3Lmd1c2liaS5jb20iLCJzdWIiOiIwMTIzNDU2Nzg5Iiwibmlja25hbWUiOiJnb29kc3BlZWQiLCJ1c2VybmFtZSI6Imdvb2RzcGVlZCIsInNjb3BlcyI6WyJhZG1pbiIsInVzZXIiXX0.pq5IDv-yaktw6XEa5GEv07SzS9ehe6AcVSdTj0Ini4o
```

token 生成好之后，接下来就可以用token来和服务器进行通讯了。

## 实践

[参考](https://www.cnblogs.com/xujunkai/p/12359573.html)

### django+pyjwt

- 安装第三方库

```
pip install pyjwt
```

- 后端代码实现

jwt工具函数

```python
# utils/jwt_auth.py
import jwt
import datetime
from jwt import exceptions


# 加的盐
JWT_SALT = "ds()udsjo@jlsdosjf)wjd_#(#)$"


def create_token(payload,timeout=20):
    # 声明类型，声明加密算法
    headers = {
        "type":"jwt",
        "alg":"HS256"
    }
    # 设置过期时间
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=20)
    result = jwt.encode(payload=payload,key=JWT_SALT,algorithm="HS256",headers=headers).decode("utf-8")
    # 返回加密结果
    return result


def parse_payload(token):
    """
    用于解密
    :param token:
    :return:
    """
    result = {"status":False,"data":None,"error":None}
    try:
        # 进行解密
        verified_payload = jwt.decode(token,JWT_SALT,True)
        result["status"] = True
        result['data']=verified_payload
    except exceptions.ExpiredSignatureError:
        result['error'] = 'token已失效'
    except jwt.DecodeError:
        result['error'] = 'token认证失败'
    except jwt.InvalidTokenError:
        result['error'] = '非法的token'
    return result

```

views视图

```python
from django.http import JsonResponse

from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from utils.jwt_auth import create_token

# 定义method_decorator 免 csrf校验， dispatch表示所有请求，因为所有请求都先经过dispatch
@method_decorator(csrf_exempt,name="dispatch")
class LoginView(View):
    """
    登陆校验
    """
    def post(self,request,*args,**kwargs):
        user = request.POST.get("username")
        pwd = request.POST.get("password")
        # 这里简单写一个账号密码
        if user == "xjk" and pwd == "123":
            # 登陆成功进行校验
            token = create_token({"username":"xjk"})
            # 返回JWT token
            return JsonResponse({"status":True,"token":token})
        return JsonResponse({"status":False,"error":"用户名密码错误"})

# 定义method_decorator 免 csrf校验， dispatch表示所有请求，因为所有请求都先经过dispatch
@method_decorator(csrf_exempt,name="dispatch")
class OrderView(View):
    """
    登陆后可以访问
    """
    def get(self, request, *args, **kwargs):
        # 打印用户jwt信息
        print(request.user_info)
        return JsonResponse({'data': '订单列表'})

    def post(self, request, *args, **kwargs):
        print(request.user_info)
        return JsonResponse({'data': '添加订单'})

    def put(self, request, *args, **kwargs):
        print(request.user_info)
        return JsonResponse({'data': '修改订单'})

    def delete(self, request, *args, **kwargs):
        print(request.user_info)
        return JsonResponse({'data': '删除订单'})
```

中间件校验

```python
# middlewares/jwt.py
class JwtAuthorizationMiddleware(MiddlewareMixin):
    """
    用户需要通过请求头的方式来进行传输token，例如：
    Authorization:jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1NzM1NTU1NzksInVzZXJuYW1lIjoid3VwZWlxaSIsInVzZXJfaWQiOjF9.xj-7qSts6Yg5Ui55-aUOHJS4KSaeLq5weXMui2IIEJU
    """

    def process_request(self, request):

        # 如果是登录页面，则通过
        if request.path_info == '/login/':
            return

        # 非登录页面需要校验token
        authorization = request.META.get('HTTP_AUTHORIZATION', '')
        print(authorization)
        auth = authorization.split()
        # 验证头信息的token信息是否合法
        if not auth:
            return JsonResponse({'error': '未获取到Authorization请求头', 'status': False})
        if auth[0].lower() != 'jwt':
            return JsonResponse({'error': 'Authorization请求头中认证方式错误', 'status': False})
        if len(auth) == 1:
            return JsonResponse({'error': "非法Authorization请求头", 'status': False})
        elif len(auth) > 2:
            return JsonResponse({'error': "非法Authorization请求头", 'status': False})

        token = auth[1]
        # 解密
        result = parse_payload(token)
        if not result['status']:
            return JsonResponse(result)
        # 将解密后数据赋值给user_info
        request.user_info = result['data']
```

settings注册中间件

```python
MIDDLEWARE = [
    ...
    'middlewares.jwt.JwtAuthorizationMiddleware',
    ...
]
```

- 前端请求

请求后端登录接口得到token后，将token值存储到`localstorage`中

在其他接口请求时，将Header中添加`Authorization`，对应值为`jwt+空格+token` 

login.vue

```javascript
{
  submitForm(formName) {
    this.$axios
      .post('/api/admin/login', {
        userName: this.ruleForm.userName,
        password: this.ruleForm.password
      })
      .then(successResponse => {
        this.responseResult = JSON.stringify(successResponse.data)
        this.msg = JSON.stringify(successResponse.data.msg)
        if (successResponse.data.code === 200) {
          this.msg='';
          localStorage.setItem('userName',this.ruleForm.userName);
          //获取并存储服务器返回的AuthorizationToken信息
          var authorization = successResponse.data.token;
          localStorage.setItem('authorization',authorization);
          //登录成功跳转页面
          this.$router.push('/dashboard');
          
        }
      })
      .catch(failResponse => {})
  }
}
```

main.js

```javascript
//自动给同一个vue项目的所有请求添加请求头
axios.interceptors.request.use(function (config) {
  let token = localStorage.getItem('authorization');
  if (token) {
    config.headers['Authorization'] = 'jwt '+ token;
  }
  return config;
})
```

### drf+pyjwt

- 后端代码实现

setting

```python
// 引入restframework
INSTALLED_APPS = [
    'rest_framework',
]
```

认证类定义

```python
#!/usr/bin/env python
# -*- coding:utf-8 -*-
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from utils.jwt_auth import parse_payload


class JwtQueryParamAuthentication(BaseAuthentication):
    """
    用户需要在url中通过参数进行传输token，例如：
    http://www.pythonav.com?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1NzM1NTU1NzksInVzZXJuYW1lIjoid3VwZWlxaSIsInVzZXJfaWQiOjF9.xj-7qSts6Yg5Ui55-aUOHJS4KSaeLq5weXMui2IIEJU
    """

    def authenticate(self, request):
        # 从url上获取jwt token
        token = request.query_params.get('token')
        payload = parse_payload(token)
        if not payload['status']:
            raise exceptions.AuthenticationFailed(payload)

        # 如果想要request.user等于用户对象，此处可以根据payload去数据库中获取用户对象。
        return (payload, token)


class JwtAuthorizationAuthentication(BaseAuthentication):
    """
    用户需要通过请求头的方式来进行传输token，例如：
    Authorization:jwt eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE1NzM1NTU1NzksInVzZXJuYW1lIjoid3VwZWlxaSIsInVzZXJfaWQiOjF9.xj-7qSts6Yg5Ui55-aUOHJS4KSaeLq5weXMui2IIEJU
    """

    def authenticate(self, request):
        # 非登录页面需要校验token,从头信息拿去JWT Token
        authorization = request.META.get('HTTP_AUTHORIZATION', '')
        auth = authorization.split()
        if not auth:
            raise exceptions.AuthenticationFailed({'error': '未获取到Authorization请求头', 'status': False})
        if auth[0].lower() != 'jwt':
            raise exceptions.AuthenticationFailed({'error': 'Authorization请求头中认证方式错误', 'status': False})

        if len(auth) == 1:
            raise exceptions.AuthenticationFailed({'error': "非法Authorization请求头", 'status': False})
        elif len(auth) > 2:
            raise exceptions.AuthenticationFailed({'error': "非法Authorization请求头", 'status': False})

        token = auth[1]
        result = parse_payload(token)
        if not result['status']:
            raise exceptions.AuthenticationFailed(result)

        # 如果想要request.user等于用户对象，此处可以根据payload去数据库中获取用户对象。
        return (result, token)
```

view.py使用

```python
from rest_framework.views import APIView
from rest_framework.response import Response

from utils.jwt_auth import create_token
from extensions.auth import JwtQueryParamAuthentication, JwtAuthorizationAuthentication


class LoginView(APIView):
    def post(self, request, *args, **kwargs):
        """ 用户登录 """
        user = request.POST.get('username')
        pwd = request.POST.get('password')

        # 检测用户和密码是否正确，此处可以在数据进行校验。
        if user == 'xjk' and pwd == '123':
            # 用户名和密码正确，给用户生成token并返回
            token = create_token({'username': 'xjk'})
            return Response({'status': True, 'token': token})
        return Response({'status': False, 'error': '用户名或密码错误'})


class OrderView(APIView):
    # 通过url传递token
    authentication_classes = [JwtQueryParamAuthentication, ]

    # 通过Authorization请求头传递token
    # authentication_classes = [JwtAuthorizationAuthentication, ]

    def get(self, request, *args, **kwargs):
        print(request.user, request.auth)
        return Response({'data': '订单列表'})

    def post(self, request, *args, **kwargs):
        print(request.user, request.auth)
        return Response({'data': '添加订单'})

    def put(self, request, *args, **kwargs):
        print(request.user, request.auth)
        return Response({'data': '修改订单'})

    def delete(self, request, *args, **kwargs):
        print(request.user, request.auth)
        return Response({'data': '删除订单'})
```

- 前端接口请求

请求后端登录接口得到token后，将token值存储到`localstorage`中

在其他接口请求时，将`token` 添加到url上

### drf+drf-jwt

rest_framework_jwt是封装jwt符合restful规范接口

- 安装

```python
pip install djangorestframework-jwt
```

- 配置

settings.py配置

```python
INSTALLED_APPS = [
    ...
    'rest_framework'
]


import datetime
#超时时间
JWT_AUTHTIME = {
    'JWT_EXPIRATION_DELTA': datetime.timedelta(days=1),
    # token前缀
    'JWT_AUTH_HEADER_PREFIX': 'JWT',
}

# 引用Django自带的User表，继承使用时需要设置
AUTH_USER_MODEL = 'api.User'
```

models.py建立表

```python
from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    CHOICE_GENDER = (
        (1,"男"),
        (2,"女"),
        (3,"不详"),
    )
    gender = models.IntegerField(choices=CHOICE_GENDER,null=True,blank=True)
    class Meta:
        db_table = "user"
```

定义一个路由创建一个用户

```python
urlpatterns = [
    url(r'^reg/', views.RegView.as_view()),
]
```

创建注册用户视图：

```python
class RegView(APIView):
    def post(self,request,*args,**kwargs):
        receive = request.data
        username = receive.get("username")
        password = receive.get("password")
        user = User.objects.create_user(
            username=username, password=password
        )
        user.save()
        return Response({"code":200,"msg":"ok"})
```

在url添加登陆路由

```python
from django.conf.urls import url
from django.contrib import admin
from rest_framework_jwt.views import obtain_jwt_token
from api import views
urlpatterns = [
    # 登入验证，使用JWT的模块，只要用户密码正确会自动生成一个token返回
    url(r'^login/', obtain_jwt_token),
    # 访问带认证接口
    url(r'^home/', views.Home.as_view()),
]
```

定义认证视图：

```python
class Home(APIView):
    authentication_classes = [JwtAuthorizationAuthentication]
    def get(self,request,*args,**kwargs):
        return Response({"code":200,"msg":"this is home"})
```

定义认证类`JwtAuthorizationAuthentication`:

```python
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_jwt.serializers import VerifyJSONWebTokenSerializer

class JwtAuthorizationAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # 获取头信息token
        authorization = request.META.get('HTTP_AUTHORIZATION', '')
        print(authorization)
        # 校验
        valid_data = VerifyJSONWebTokenSerializer().validate({"token":authorization})
        """
        valid_data = {'token': '太长了省略一下...'
        'user': <User: xjk>
        }
        """
        user = valid_data.get("user")
        if user:
            return
        else:
            raise AuthenticationFailed("认证失败了。。。")
```

- 前端请求

请求后端登录接口得到token后，将token值存储到`localstorage`中

在其他接口请求时，将Header中添加`Authorization`，对应值为`token` 

## 刷新机制

token设置有效期，但有效期不宜过长，所以需要刷新。

解决无感知刷新流程：

- 手机号+验证码（或帐号+密码）验证后颁发接口调用token与refresh_token（刷新token）
- Token 有效期为2小时，在调用接口时携带，每2小时刷新一次
- 提供refresh_token，refresh_token 有效期14天
- 在接口调用token过期后凭借refresh_token 获取新token，之后前端重新请求
- 未携带token 、错误的token或接口调用token过期，返回401状态码
- refresh_token 过期返回403状态码，前端在使用refresh_token请求新token时遇到403状态码则进入用户登录界面从新认证。

## 禁用问题

此问题的应用场景：1. 用户修改密码，需要颁发新的token，禁用还在有效期内的老token；2.后台封禁用户。

- 解决方案

在redis中使用set类型保存新生成的token

```python
key = 'user:{}:token'.format(user_id)
pl = redis_client.pipeline()
pl.sadd(key, new_token)
pl.expire(key, token有效期)
pl.execute()
```

| 键                     | 类型 | 值      |
| ---------------------- | ---- | ------- |
| `user:{user_id}:token` | set  | 新token |

客户端使用token进行请求时，如果验证token通过，则从redis中判断是否存在该用户的`user:{}:token`记录：

1. 若不存在记录，放行，进入视图进行业务处理

2. 若存在，则对比本次请求的token是否在redis保存的set中：

若存在，则放行；若不在set的数值中，则返回403状态码，不再处理业务逻辑

```python
key = 'user:{}:token'.format(user_id)
valid_tokens = redis_client.smembers(key, token)
if valid_tokens and token not in valid_tokens:
  return {'message': 'Invalid token'.}, 403
```

> 说明

1. redis记录设置有效期的时长是一个token的有效期，保证旧token过期后，redis的记录也能自动清除，不占用空间。
2. 使用set保存新token的原因是，考虑到用户可能在旧token的有效期内，在其他多个设备进行了登录，需要生成多个新token，这些新token都要保存下来，既保证新token都能正常登录，又能保证旧token被禁用