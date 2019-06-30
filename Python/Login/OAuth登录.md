# OAuth登陆

## OAuth协议

​    `OAuth` 协议为用户资源的授权提供了一个安全又简易的标准。与以往的授权方式不同之处是 `OAuth`的授权不会使第三方触及到用户的帐号信息（如用户名与密码），即第三方无需使用用户的用户名与密码就可以申请获得该用户资源的授权，因此 `OAuth`是安全的。`OAuth` 是 **Open Authorization** 的简写

​    `OAuth` 本身不存在一个标准的实现，后端开发者自己根据实际的需求和标准的规定实现。其步骤一般如下：

```
1. 第三方要求用户给予授权
2. 用户同意授权
3. 根据上一步获得的授权，第三方向认证服务器请求令牌（`token`）
4. 认证服务器对授权进行认证，确认无误后发放令牌
5. 第三方使用令牌向资源服务器请求资源
6. 资源服务器使用令牌向认证服务器确认令牌的正确性，确认无误后提供资源
```

## OAuth2.0

### 原理概述

​    **任何身份认证，本质上都是基于对请求方的不信任所产生的**。同时，请求方是信任被请求方的，例如用户请求服务时，会信任服务方。所以，**身份认证**就是为了解决**身份的可信任**问题。

​    在`OAuth2.0`中，简单来说有三方：**用户**（这里是指属于`服务方的用户`）、**服务方**（如微信、微博等）、**第三方应用**

```
1. 服务方不信任**用户**，所以需要用户提供密码或其他可信凭据
2. 服务方不信任**第三方应用**，所以需要第三方提供自已交给它的凭据（如微信授权的`code,AppID`等）
3. 用户部分信任**第三方应用**，所以用户愿意把自已在服务方里的某些服务交给第三方使用，但不愿意把自已在服务方的密码等交给第三方应用
```

OAuth2.0成员

```
1. Resource Owner（资源拥有者：用户）
2. Client （第三方接入平台：请求者）
3. Resource Server （服务器资源：数据中心）
4. Authorization Server （认证服务器）
```

- OAuth2.0基本流程

![clipboard.png](https://segmentfault.com/img/bVSnXh?w=561&h=372)

步骤详解：

```
1. Authorization Request`， 第三方请求用户授权
2. `Authorization Grant`，用户同意授权后，会从服务方获取一次性用户**授权凭据**(如`code`码)给第三方
3. `Authorization Grant`，第三方会把**授权凭据**以及服务方给它的的**身份凭据**(如`AppId`)一起交给服务方的向认证服务器申请**访问令牌**
4. `Access Token`，认证服务器核对授权凭据等信息，确认无误后，向第三方发送**访问令牌**`Access Token`等信息
5. `Access Token`，通过这个`Access Token`向`Resource Server`索要数据
6. `Protected Resource`，资源服务器使用令牌向认证服务器确认令牌的正确性，确认无误后提供资源
```

​    这样服务方，一可以确定第三方得到了用户对此次服务的授权（根据用户授权凭据），二可以确定第三方的身份是可以信任的（根据身份凭据），所以，最终的结果就是，第三方顺利地从服务方获取到了此次所请求的服务
​    从上面的流程中可以看出，`OAuth2.0`完整地解决了**用户**、**服务方**、**第三方** 在某次服务时这三者之间的信任问题

### 实施步骤

OAuth2.0的授权可以简单分为三步：

1. 获取用户授权码Code
2. 获取用户授权令牌Token
3. 使用授权令牌Token获取用户信息

第一步，又称用户登录引导页面。在微信登录时，这个页面的域名是在微信下的，用户同意授权后，微信会把授权码Code送到服务器（通过回调URI的形式）。拿到这个Code表示`用户同意了授权`。

第二步，在微信登录时，这个token又叫`access_token`。拿到这个Token表示`服务器是合法的`。

第三步，在微信登录时，这一步可以拿到用户的`open_id`。

在微信登录中，如果要获取用户基本信息，需要用`open_id`+`access_token`才能得到。

关于OAuth2.0协议更多内容，可以参考这2篇文章：[深入理解OAuth2.0协议](https://www.cnblogs.com/hyl8218/p/3584505.html) ，[理解OAuth 2.0](http://www.ruanyifeng.com/blog/2014/05/oauth_2_0.html)

### 如何集成

一个用户可以”绑定”多个第三方账号，这是一个比较好的处理第三方用户的方式。第三方用户的管理必须重视，如果管理混乱，绑定的信息不能指向同一个用户，就会出现多身份问题，比如用户使用手机登录购买的东西，在使用微信登录时却提示没有购买。

我介绍一下我的做法，数据库两张表：

- `user`表，记录用户信息。这里有`telephone`和`email`等可用于登录的字段
- `user_third`表，记录用户绑定的第三方账号信息。

登录逻辑如下：

- 当用户使用如手机号、邮箱、登录名登录时，在`user`表里查询信息。
- 当用户使用第三方登录时，系统先去`user_third`里查询信息，如果未找到，则在`user`表里新建用户，再将第三方账号信息保存到`user_third`里，最后把新建的用户与第三方账号信息绑定；如果能找到，则返回第三方账号所绑定的`user`表里的数据。

这种做法，可以保证用户数据均来自`user`表，就不会有多身份问题，同时一个用户也可以绑定多个第三方账号，更加便于管理。

还有一种情况是绑定信息冲突，比如用户第一个账号绑定了手机号和微信账号，过段时间后，他用QQ账号登录时（此时这个QQ号没有对应系统内的用户）系统会创建第二个账号，此时他再去绑定手机号或微信号的时候，会因为`user`表的`telephone`字段、`user_third`表中已有信息，而导致绑定失败。

处理这种情况常用的方法是**解绑**，用户可以解绑QQ号，再绑定QQ号至第一次创建的账号；也可以选择解绑手机、微信，再将手机、微信绑到第二个账号上。

## 授权码模式详解

[参看微信页面授权](https://mp.weixin.qq.com/wiki?t=resource/res_main&id=mp1421140842)

### 授权码模式

​    客户端必须得到用户的授权（`authorization grant`），才能获得令牌（`access token`）。`OAuth 2.0`定义了四种授权方式：

```
1. 授权码模式（`authorization code`）
2. 简化模式（`implicit`）
3. 密码模式（`resource owner password credentials`）
4. 客户端模式（`client credentials`）
```

​    **授权码模式（authorization code）是功能最完整、流程最严密的授权模式。**它的特点就是通过客户端的后台服务器与"**服务提供商**"的认证服务器进行互动。

### 授权码流程图及步骤

​    ![clipboard.png](https://segmentfault.com/img/bVSn2t?w=715&h=414)

它的步骤如下：

```
1. 用户访问客户端，后者将前者导向认证服务器
2. 用户选择是否给予客户端授权
3. 假设用户给予授权，认证服务器将用户导向客户端事先指定的重定向`URI`，同时附上一个授权码
4. 客户端收到授权码，附上早先的重定向`URI`，向认证服务器申请令牌。这一步是在客户端的后台的服务器上完成的，对用户不可见
5. 认证服务器核对了授权码和重定向`URI`，确认无误后，向客户端发送访问令牌（`access token`）和更新令牌（`refresh token`）等
```

### 步骤详情及所需参数

#### 客户端申请认证的URI

包含以下参数：

```
- response_type：表示授权类型，必选项，此处的值固定为"code"
- client_id：表示客户端的ID，必选项。（`如微信授权登录，此ID是APPID`）
- redirect_uri：表示重定向URI，可选项
- scope：表示申请的权限范围，可选项 state：表示客户端的当前状态，可以指定任意值，认证服务器会原封不动地返回这个值
```

示例

```javascript
GET /authorize?response_type=code&client_id=s6BhdRkqt3&state=xyz
        &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb HTTP/1.1
HTTP/1.1 Host: server.example.com
```

对比网站应用微信登录：请求CODE

```
https://open.weixin.qq.com/connect/qrconnect?appid=APPID&redirect_uri=REDIRECT_URI&response_type=code&scope=SCOPE&state=STATE#wechat_redirect
```

#### 认证服务器回应客户端的URI

包含以下参数

```
- code：表示授权码，必选项。该码的有效期应该很短，通常设为10分钟，客户端`只能使用该码一次`，否则会被授权服务器拒绝。该码与客户端ID和重定向URI，是一一对应关系。
- state：如果客户端的请求中包含这个参数，认证服务器的回应也必须一模一样包含这个参数。
```

示例

```javascript
HTTP/1.1 302 Found
Location: https://client.example.com/cb?code=SplxlOBeZQQYbYS6WxSbIA
          &state=xyz
```

#### 客户端向认证服务器申请令牌的HTTP请求

包含以下参数：

```
- grant_type：表示使用的授权模式，必选项，此处的值固定为"authorization_code"。
- code：表示上一步获得的授权码，必选项。
- redirect_uri：表示重定向URI，必选项，且必须与A步骤中的该参数值保持一致。
- client_id：表示客户端ID，必选项。
```

示例

```javascript
POST /token HTTP/1.1
Host: server.example.com
Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code&code=SplxlOBeZQQYbYS6WxSbIA
&redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb
```

对比网站应用微信登录：通过code获取access_token

```javascript
https://api.weixin.qq.com/sns/oauth2/access_token?appid=APPID&secret=SECRET&code=CODE&grant_type=authorization_code
```

#### 认证服务器发送的HTTP回复

 包含以下参数：

```
- access_token：表示访问令牌，必选项。
- token_type：表示令牌类型，该值大小写不敏感，必选项，可以是bearer类型或mac类型。
- expires_in：表示过期时间，单位为秒。如果省略该参数，必须其他方式设置过期时间。
- refresh_token：表示更新令牌，用来获取下一次的访问令牌，可选项。
- scope：表示权限范围，如果与客户端申请的范围一致，此项可省略。
```

示例：

```javascript
 HTTP/1.1 200 OK
     Content-Type: application/json;charset=UTF-8
     Cache-Control: no-store
     Pragma: no-cache

     {
       "access_token":"2YotnFZFEjr1zCsicMWpAA",
       "token_type":"example",
       "expires_in":3600,
       "refresh_token":"tGzv3JOkF0XG5Qx2TlKWIA",
       "example_parameter":"example_value"
     }
```

从上面代码可以看到，相关参数使用JSON格式发送（`Content-Type: application/json`）。此外，HTTP头信息中明确指定不得缓存。

对比网站应用微信登录：返回样例

```
{ 
"access_token":"ACCESS_TOKEN", 
"expires_in":7200, 
"refresh_token":"REFRESH_TOKEN",
"openid":"OPENID", 
"scope":"SCOPE",
"unionid": "o6_bmasdasdsad6_2sgVt7hMZOPfL"
}
```

### 更新令牌

 如果用户访问的时候，客户端的**访问令牌**`access_token`已经过期，则需要使用**更新令牌**`refresh_token`申请一个新的访问令牌。
 客户端发出更新令牌的HTTP请求，包含以下参数：

```
- granttype：表示使用的授权模式，此处的值固定为"refreshtoken"，必选项。
- refresh_token：表示早前收到的更新令牌，必选项。
- scope：表示申请的授权范围，不可以超出上一次申请的范围，如果省略该参数，则表示与上一次一致。
```

示例

```javascript
POST /token HTTP/1.1
Host: server.example.com
Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&refresh_token=tGzv3JOkF0XG5Qx2TlKWIA
```