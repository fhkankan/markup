# Session

## 概述

- cookie

http请求时无状态的。就是说第一次和服务器连接并登陆成功后，第二次请求服务器仍然不知道当前请求的用户。

cookie出现就是解决了这个问题。第一次登陆后服务器返回一些数据(cookie)给浏览器，然后浏览器保存在本地，当用户第二次返回请求的时候，就会把上次请求存储的cookie数据自动携带给服务器。

Cookie 是由客户端保存的小型文本文件，其内容为一系列的键值对。Cookie 是由 HTTP 服务器设置的，保存在浏览器中。Cookie会随着 HTTP请求一起发送。 

如果关闭浏览器cookie失效(cookie就是保存在内存中)

如果关闭浏览器cookie不失效(cookie保存在磁盘中)

- session

Session 是存储在服务器端的，避免在客户端 Cookie 中存储敏感数据。Session 可以存储在 HTTP 服务器的内存中，也可以存在内存数据库（如redis）中。 

Cookie/Session认证机制就是为一次请求认证在服务端创建一个Session对象，同时在客户端的浏览器端创建了一个Cookie对象；通过客户端带上来Cookie对象来与服务器端的session对象匹配来实现状态管理的。默认的，当我们关闭浏览器的时候，cookie会被删除。但可以通过修改cookie 的expire time使cookie在一定时间内有效；

Session是一种将数据存储在服务器端的会话控制技术，我们可以使用它实现用户认证。

- 特点

优点

```
网站支持安全成熟
开发方便简单
```

缺点

```
- 服务器压力增大
通常session是存储在内存中的，每个用户通过认证之后都会将session数据保存在服务器的内存中，而当用户量增大时，服务器的压力增大。
- CSRF跨站伪造请求攻击
session是基于cookie进行用户识别的, cookie如果被截获，用户就会很容易受到跨站请求伪造的攻击。可以添加csrf_token处理。
- 扩展性不强
如果将来搭建了多个服务器，虽然每个服务器都执行的是同样的业务逻辑，但是session数据是保存在内存中的（不是共享的），用户第一次访问的是服务器1，当用户再次请求时可能访问的是另外一台服务器2，服务器2获取不到session信息，就判定用户没有登陆过。
- 手机请求不支持
Cookie不支持手机端访问的
```

## 实现

下面是一个基于Laravel5的PHP版本的用户认证：

```php
/**
 * 用户登录
 * @param string $login 登录名
 * @param string $password 登录密码
 * @return UserModel|false
 */
function userLogin($login, $password) {
  	// 按`$login`从数据库中取出匹配的第一个`用户实例`。
    $user = UserModel::where('login', $login)->first();
  	// 判断是否认证成功，`checkPassword`用于判断`$password`是否符合`$user`的密码。
    if ($user && $user->checkPassword($password)) {
      	// 将`$user`存入session中，键为`_user`。
        session()->put('_user', $user);
      	// 认证成功，返回用户实例`$user`。
        return $user;
    } else {
      	// 认证失败，返回`false`。
        return false;
    }
}

/**
 * 获取已经登录的用户实例
 * @return UserModel|null
 */
function getLoginUser() {
  	// 从session中取出用户实例。
    return session()->get('_user');
}
```

`userLogin`函数接受用户名、密码两个参数进行用户认证工作，认证成功返回`用户实例`，失败返回`false`。

`getLoginUser`函数用于获取已经登录的用户，已登录返回`用户实例`，未登录返回`null`（由session()->get函数返回的）。

这种做法的核心思想是把用户数据直接交由Session保管。

Session可以基于Cookie或URL实现，不论哪种形式，都需要先由服务器种下`session-id`（种在Cookie里或是重在URL里），后续请求带上这个session-id，服务器才能实现Session。