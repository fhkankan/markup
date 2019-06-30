# Session

Cookie 是由客户端保存的小型文本文件，其内容为一系列的键值对。Cookie 是由 HTTP 服务器设置的，保存在浏览器中。Cookie会随着 HTTP请求一起发送。 
Session 是存储在服务器端的，避免在客户端 Cookie 中存储敏感数据。Session 可以存储在 HTTP 服务器的内存中，也可以存在内存数据库（如redis）中。 

Cookie/Session认证机制就是为一次请求认证在服务端创建一个Session对象，同时在客户端的浏览器端创建了一个Cookie对象；通过客户端带上来Cookie对象来与服务器端的session对象匹配来实现状态管理的。默认的，当我们关闭浏览器的时候，cookie会被删除。但可以通过修改cookie 的expire time使cookie在一定时间内有效；

------

Session是一种将数据存储在服务器端的会话控制技术，我们可以使用它实现用户认证。

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