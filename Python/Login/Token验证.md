# Token

## 概述

基于 Token的认证机制，有着无需长期保存用户名和密码，服务器端能主动让token失效等诸多好处，非常实用于 Web 应用和 App 已经被很多大型网站采用，比如Facebook、Github、Google+等。

- 特点

优点

```
相对于Cookie/Session的好处：
- 支持跨域访问: Cookie是不允许垮域访问的，token支持
- 无状态： token无状态，session有状态的
- 去耦: 不需要绑定到一个特定的身份验证方案。Token可以在任何地方生成，只要在 你的API被调用的时候， 你可以进行Token生成调用即可.
- 更适用于移动应用: Cookie不支持手机端访问的
- 性能: 在网络传输的过程中，性能更好
- 基于标准化: 你的API可以采用标准化的 JSON Web Token (JWT). 这个标准已经存在 多个后端库（.NET, Ruby, Java,Python, PHP）和多家公司的支持（如： Firebase,Google, Microsoft）
```

缺点

```

```

- 流程

```
1. 客户端使用用户名和密码请求登录 
2. 服务端收到请求，去验证用户名与密码 
3. 验证成功后，服务器会签发一个 Token， 再把这个 Token 发送给客户端 
4. 客户端收到 Token 以后可以把它存储起来，如Cookie或者Web Storage 
5. 客户单每次向服务端请求资源的时候，都需要带着服务器端签发的 Token 
6. 服务器端收到请求，验证客户端请求里面带着的 Token，如果验证成功，就向客户端返回请求的数据；否的话，则返回对应的错误信息。
```
- 安全

在基于令牌的认证里，`token`是最为关键的信息，如果有第三方窃取到了用户的token，他就可以冒充用户的进行操作。

但是存储在客户端的 Token 存在几个问题： 

1. 存在泄露的风险。如果别人拿到你的 Token，在 Token过期之前，都可以以你的身份在别的地方登录 
2. 如果存在 Web Storage（指sessionStorage和localStorage）。由于Web Storage 可以被同源下的JavaScript直接获取到，这也就意味着网站下所有的JavaScript代码都可以获取到web Storage，这就给了XSS机会 
3. 如果存在Cookie中。虽然存在Cookie可以使用HttpOnly来防止XSS，但是使用 Cookie 却又引发了CSRF

> token加密

对于泄露的风险，可以采取对Token进行对称加密，用时再解密。 
对于XSS而言，在处理数据时，都应该 escape and encode 所有不信任的数据。 
与CSRF相比，XSS更加容易防范和意识到，因此并不建议将Token存在Cookie中。 

>  隐藏Token

啥意思呢？就是把`token`放在HTTP头里，尽量让用户感觉不到`token`的存在。比如下面的HTTP头：

```
...
X-AUTH-TOKEN: 340c6f730612769b71075d4fbbe5d337 
...
```

但是如果HTTP包被黑客获取，他仍然能够窃取到`token`。

>  使用HTTPS

HTTPS会将数据包加密，所以黑客就算截取到数据包到也无法获取`token`。

[实例参考](https://github.com/superman66/vue-axios-github)

------

## 实现

基于令牌的用户认证，本质是将登录时随机生成的`token`写在HTTP头或是写在URL上，服务器通过鉴别`token`来进行用户认证。

上代码：

```php
/**
 * 用户登录
 * @param string $login 登录名
 * @param string $password 登录密码
 * @return UserModel|false
 */
function userLogin($login, $password) {
    $user = UserModel::where('login', $login)->first();
    if ($user && $user->checkPassword($password)) {
      	// 使用`$user`生成`token`，将用户实例存入缓存系统中。
        $token = $user->generateAuthToken();
        session()->put('_token', $token);
        cache()->put('user_' . $token, $user);
        return $user;
    } else {
        return false;
    }
}

/**
 * 获取已经登录的用户实例
 * @return UserModel|null
 */
function getLoginUser($token = null) {
  	// 使用`token`从缓存系统中获取用户实例。
    if (! $token) $token = session()->get('_token');
    $cache_key = 'user_' . $token;
    return cache()->get($cache_key);
}
```

这个版本的`userLogin`函数，在认证成功后，通过用户实例生成一个`token`放入session，再把用户实例`$user`放入缓存系统中（如Redis、Memcache）。`token`一般都是32位的md5值。

`getLoginUser` 函数也有所变化，它可以接受指定的`$token`来获取用户实例，默认情况下它会从session中取出token。

的一种可用的用于生成`token`的方法：

```
/**
 * 生成认证token
 * @return string 认证token
 */
public function generateAuthToken() {
    if ($this->token) return $this-token;
    return $this->token = md5(md5($this->id . time()));
}
```

`time()`函数返回当前unix时间戳。可以看到，token与`用户id`和`登录时间`有关，这可以保证唯一性。

这样的用户认证下，API请求怎么做呢？

我们先创建一个接口 `/login` 用于登录，接口的返回值里，附上登录成功后的 `token`，HTTP Client将这个token缓存起来，在之后的请求中带上这个token即可。这样以来，用户认证就不是基于Cookie而是基于token了。

这样的用户认证已经可以满足大部分应用场景了如Cookie失效、API请求和统一认证。但还有一个场景无法满足，那就是多终端数据共享。比如用户在电脑上登录了一次，在手机上登录了一次，系统会生成2个token，这两个token对应的用户实例是不一样的，所以用户在电脑上设置的个性化信息（比如性别，名称）无法共享到手机上。

