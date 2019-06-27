# JWT验证

## 几种常用的认证机制



### OAuth

OAuth 是一个关于授权（authorization）的开放网络标准。允许用户提供一个令牌，而不是用户名和密码来访问他们存放在特定服务提供者的数据。现在的版本是2.0版。

严格来说，OAuth2不是一个标准协议，而是一个安全的授权框架。它详细描述了系统中不同角色、用户、服务前端应用（比如API），以及客户端（比如网站或移动App）之间怎么实现相互认证。

#### 名词定义

- Third-party application: 第三方应用程序，又称"客户端"（client）
- HTTP service：HTTP服务提供商
- Resource Owner：资源所有者，通常称"用户"（user）。
- User Agent：用户代理，比如浏览器。
- Authorization server：认证服务器，即服务提供商专门用来处理认证的服务器。
- Resource server：资源服务器，即服务提供商存放用户生成的资源的服务器。它与认证服务器，可以是同一台服务器，也可以是不同的服务器。

OAuth 2.0 运行流程如图：

![OAuth 2.0 运行流程](https://static.segmentfault.com/v-5cc2cd8e/global/img/squares.svg)

（A）用户打开客户端以后，客户端要求用户给予授权。
（B）用户同意给予客户端授权。
（C）客户端使用上一步获得的授权，向认证服务器申请令牌。
（D）认证服务器对客户端进行认证以后，确认无误，同意发放令牌。
（E）客户端使用令牌，向资源服务器申请获取资源。
（F）资源服务器确认令牌无误，同意向客户端开放资源。

> ```
> 优点
> ```

快速开发
实施代码量小
维护工作减少
如果设计的API要被不同的App使用，并且每个App使用的方式也不一样，使用OAuth2是个不错的选择。

> `缺点`：
> OAuth2是一个安全框架，描述了在各种不同场景下，多个应用之间的授权问题。有海量的资料需要学习，要完全理解需要花费大量时间。
> OAuth2不是一个严格的标准协议，因此在实施过程中更容易出错。

了解了以上两种方式后，现在终于到了本篇的重点，JWT 认证。

## JWT 认证

> `Json web token (JWT)`, 根据官网的定义，是为了在网络应用环境间传递声明而执行的一种基于JSON的开放标准（(RFC 7519).该token被设计为紧凑且安全的，特别适用于分布式站点的单点登录（SSO）场景。JWT的声明一般被用来在身份提供者和服务提供者间传递被认证的用户身份信息，以便于从资源服务器获取资源，也可以增加一些额外的其它业务逻辑所必须的声明信息，该token也可直接被用于认证，也可被加密。

### JWT 特点

- 体积小，因而传输速度快
- 传输方式多样，可以通过URL/POST参数/HTTP头部等方式传输
- 严格的结构化。它自身（在 payload 中）就包含了所有与用户相关的验证消息，如用户可访问路由、访问有效期等信息，服务器无需再去连接数据库验证信息的有效性，并且 payload 支持为你的应用而定制化。
- 支持跨域验证，可以应用于单点登录。

### JWT原理

JWT是Auth0提出的通过对JSON进行加密签名来实现授权验证的方案，编码之后的JWT看起来是这样的一串字符：

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.TJVA95OrM7E2cBab30RMHrHDcEfxjoYZgeFONFh7HgQ
```

由 `.` 分为三段，通过解码可以得到：

#### 1. 头部（Header）

```
// 包括类别（typ）、加密算法（alg）；
{
  "alg": "HS256",
  "typ": "JWT"
}
```

jwt的头部包含两部分信息：

- 声明类型，这里是jwt
- 声明加密的算法 通常直接使用 HMAC SHA256

然后将头部进行base64加密（该加密是可以对称解密的)，构成了第一部分。

```
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9
```

#### 2. 载荷（payload）

载荷就是存放有效信息的地方。这些有效信息包含三个部分：

- 标准中注册声明
- 公共的声名
- 私有的声明

`公共的声明 ：`
公共的声明可以添加任何的信息，一般添加用户的相关信息或其他业务需要的必要信息.但不建议添加敏感信息，因为该部分在客户端可解密。

`私有的声明 ：`
私有声明是提供者和消费者所共同定义的声明，一般不建议存放敏感信息，因为base64是对称解密的，意味着该部分信息可以归类为明文信息。

下面是一个例子：

```
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

- iss: 该JWT的签发者，是否使用是可选的；
- sub: 该JWT所面向的用户，是否使用是可选的；
- aud: 接收该JWT的一方，是否使用是可选的；
- exp(expires): 什么时候过期，这里是一个Unix时间戳，是否使用是可选的；
- iat(issued at): 在什么时候签发的(UNIX时间)，是否使用是可选的；

其他还有：

- nbf (Not Before)：如果当前时间在nbf里的时间之前，则Token不被接受；一般都会留一些余地，比如几分钟；，是否使用是可选的；
- jti: jwt的唯一身份标识，主要用来作为一次性token，从而回避重放攻击。

将上面的JSON对象进行`base64编码`可以得到下面的字符串。这个字符串我们将它称作JWT的Payload（载荷）。

```
eyJpc3MiOiJPbmxpbmUgSldUIEJ1aWxkZXIiLCJpYXQiOjE0MTY3OTc0MTksImV4cCI6MTQ0ODMzMzQxOSwiYXVkIjoid3d3Lmd1c2liaS5jb20iLCJzdWIiOiIwMTIzNDU2Nzg5Iiwibmlja25hbWUiOiJnb29kc3BlZWQiLCJ1c2VybmFtZSI6Imdvb2RzcGVlZCIsInNjb3BlcyI6WyJhZG1pbiIsInVzZXIiXX0
```

> `信息会暴露`：由于这里用的是可逆的base64 编码，所以第二部分的数据实际上是明文的。我们应该避免在这里存放不能公开的隐私信息。

#### 3. 签名（signature）

```
// 根据alg算法与私有秘钥进行加密得到的签名字串；
// 这一段是最重要的敏感信息，只能在服务端解密；
HMACSHA256(  
    base64UrlEncode(header) + "." +
    base64UrlEncode(payload),
    SECREATE_KEY
)
```

jwt的第三部分是一个签证信息，这个签证信息由三部分组成：

- header (base64后的)
- payload (base64后的)
- secret

将上面的两个编码后的字符串都用句号.连接在一起（头部在前），就形成了:

```
eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJKb2huIFd1IEpXVCIsImlhdCI6MTQ0MTU5MzUwMiwiZXhwIjoxNDQxNTk0NzIyLCJhdWQiOiJ3d3cuZXhhbXBsZS5jb20iLCJzdWIiOiJqcm9ja2V0QGV4YW1wbGUuY29tIiwiZnJvbV91c2VyIjoiQiIsInRhcmdldF91c2VyIjoiQSJ9
```

最后，我们将上面拼接完的字符串用HS256算法进行加密。在加密的时候，我们还需要提供一个密钥（secret）。如果我们用 `secret` 作为密钥的话，那么就可以得到我们加密后的内容:

```
pq5IDv-yaktw6XEa5GEv07SzS9ehe6AcVSdTj0Ini4o
```

将这三部分用.连接成一个完整的字符串,构成了最终的jwt:

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJPbmxpbmUgSldUIEJ1aWxkZXIiLCJpYXQiOjE0MTY3OTc0MTksImV4cCI6MTQ0ODMzMzQxOSwiYXVkIjoid3d3Lmd1c2liaS5jb20iLCJzdWIiOiIwMTIzNDU2Nzg5Iiwibmlja25hbWUiOiJnb29kc3BlZWQiLCJ1c2VybmFtZSI6Imdvb2RzcGVlZCIsInNjb3BlcyI6WyJhZG1pbiIsInVzZXIiXX0.pq5IDv-yaktw6XEa5GEv07SzS9ehe6AcVSdTj0Ini4o
```

> `签名的目的`：签名实际上是对头部以及载荷内容进行签名。所以，如果有人对头部以及载荷的内容解码之后进行修改，再进行编码的话，那么新的头部和载荷的签名和之前的签名就将是不一样的。而且，如果不知道服务器加密的时候用的密钥的话，得出来的签名也一定会是不一样的。
> 这样就能保证token不会被篡改。

token 生成好之后，接下来就可以用token来和服务器进行通讯了。

下图是client 使用 JWT 与server 交互过程:

![client 使用 JWT 与server 交互过程](https://static.segmentfault.com/v-5cc2cd8e/global/img/squares.svg)

这里在第三步我们得到 JWT 之后，需要将JWT存放在 client，之后的每次需要认证的请求都要把JWT发送过来。（请求时可以放到 header 的 Authorization ）

### JWT 使用场景

JWT的主要优势在于使用无状态、可扩展的方式处理应用中的用户会话。服务端可以通过内嵌的声明信息，很容易地获取用户的会话信息，而不需要去访问用户或会话的数据库。在一个分布式的面向服务的框架中，这一点非常有用。

但是，如果系统中需要使用黑名单实现长期有效的token刷新机制，这种无状态的优势就不明显了。

> ```
> 优点
> ```

快速开发
不需要cookie
JSON在移动端的广泛应用
不依赖于社交登录
相对简单的概念理解

> ```
> 缺点
> ```

Token有长度限制
Token不能撤销
需要token有失效时间限制(exp)

## python 使用JWT实践

我基本是使用 python 作为服务端语言，我们可以使用 [pyjwt：https://github.com/jpadilla/pyjwt/](https://github.com/jpadilla/pyjwt/)

使用比较方便，下边是我在应用中使用的例子：

```
import jwt
import time

# 使用 sanic 作为restful api 框架 
def create_token(request):
    grant_type = request.json.get('grant_type')
    username = request.json['username']
    password = request.json['password']
    if grant_type == 'password':
        account = verify_password(username, password)
    elif grant_type == 'wxapp':
        account = verify_wxapp(username, password)
    if not account:
        return {}
    payload = {
        "iss": "gusibi.com",
         "iat": int(time.time()),
         "exp": int(time.time()) + 86400 * 7,
         "aud": "www.gusibi.com",
         "sub": account['_id'],
         "username": account['username'],
         "scopes": ['open']
    }
    token = jwt.encode(payload, 'secret', algorithm='HS256')
    return True, {'access_token': token, 'account_id': account['_id']}
    

def verify_bearer_token(token):
    #  如果在生成token的时候使用了aud参数，那么校验的时候也需要添加此参数
    payload = jwt.decode(token, 'secret', audience='www.gusibi.com', algorithms=['HS256'])
    if payload:
        return True, token
    return False, token
```

这里，我们可以使用 jwt 直接生成 token，不用手动base64加密和拼接。

详细代码可以参考 [gusibi/Metis: 一个测试类小程序（包含前后端代码）](https://github.com/gusibi/Metis/blob/master/apis/verification.py)。

> 这个项目中，api 使用 python sanic，文档使用 [swagger-py-codegen](https://github.com/guokr/swagger-py-codegen) 生成，提供 swagger ui。

现在可以使用 swagger ui 来测试jwt。