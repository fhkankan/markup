# 403禁止

[参考](https://blog.csdn.net/ybdesire/article/details/48196843)

**CSRF**

CSRF是跨站点请求伪造，简单来说就是用户在访问受信任网站后，浏览器记录了受信任网站的cookie，此时用户在不登出受信任网站的同时访问了危险网站，危险网站的请求和cookie一起到达服务器，服务器Session未过期的时候，就误认为该请求是用户发出的，从而被危险网站利用。更多细节可参考[ 1 ]和[ 2 ]，目前解决CSRF攻击的方法，常用的方法就是进一步确认一个请求是来自于用户还是危险网站，就是客户端发送请求时，增加伪随机数，常用的方法有3种： 
(1)验证码 
(2)客户端登陆后发送token给服务器，且客户端每一次请求都带上这个token，服务器记录第一次登陆后的token，并对每一次请求进行验证 
(3)在HTTP头中自定义属性并验证

**Django防护机制**

HTTP请求，分为两类：“安全请求”和“不安全请求”。GET是“安全请求”， POST, PUT, DELETE是“不安全请求”。安不安全，主要还是看请求设计者。比如，如果银行系统的转账设计是用GET进行的，那GET也是不安全的。所以一般GET就用于获取资源，不要进行资源更新操作，资源更新交给POST来做。

对于“不安全请求”，Django设计了CSRF验证，简单来说这个机制是这样的： 
(1) 客户端访问Django站点，Django服务器向客户端发送名为”csrftoken”的cookie 
(2) 客户端对Django服务器发送不安全请求时，必须在HTTP头部加入”X-CSRFToken”字段，并将这个cookie的值作为该字段的值 
(3) Django服务器端会对HTTP头部X-CSRFToken的值进行验证，依次来判断这个请求是不是来自合法用户

**解决方法**

- 方法一

表单提交

```
<form>
{% csrf_token %}
</from>
```

- 方法二

表单提交

```
# settings.py的MIDDLEWARE_CLASSES中加入
‘django.middleware.csrf.CsrfResponseMiddleware’,
```

- 方法三

使用ajax

```
1.视图中使用render
2.使用jQuery的ajax或者post之前加入如下代码(直接写在模板中，不能在.js中)
$.ajaxSetup({
    data: {csrfmiddlewaretoken: '{{ csrf_token }}' },
});
```

- 方法四

所有情况，屏蔽django安全策略

```
# 在settings.py里面的MIDDLEWARE_CLASSES中去掉
‘django.middleware.csrf.CsrfViewMiddleware’,
```

- 方法五

所有情况

```
# 导入模块
from django.views.decorators.csrf import csrf_exempt

# 在函数前面添加修饰器
@csrf_exempt
def api_blogs(request):
	pass
```

# 跨域请求

[参考](https://blog.csdn.net/apple9005/article/details/54427902)

- 方法一

```
使用Ajax获取json数据时，存在跨域的限制。不过，在Web页面上调用js的script脚本文件时却不受跨域的影响，JSONP就是利用这个来实现跨域的传输。因此，我们需要将Ajax调用中的dataType从JSON改为JSONP（相应的API也需要支持JSONP）格式。 
JSONP只能用于GET请求。

$.ajax({
  dateType:'jsonp',  
})
```

- 方法二

修改views.py中对应API的实现函数，允许其他域通过Ajax请求数据

```python
def myview(_request): 
	response = HttpResponse(json.dumps({“key”: “value”, “key2”: “value”})) 
	response[“Access-Control-Allow-Origin”] = “*” 
	response[“Access-Control-Allow-Methods”] = “POST, GET, OPTIONS” 
	response[“Access-Control-Max-Age”] = “1000” 
	response[“Access-Control-Allow-Headers”] = “*” 
	return response
```

- 方法三

安装django-cors-headers

```
pip install django-cors-headers
```

配置setting.py文件

```python
INSTALLED_APPS = [
    ...
    'corsheaders'，
    ...
 ] 

MIDDLEWARE_CLASSES = (
    ...
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware', # 注意顺序
    ...
)
#跨域增加忽略
CORS_ALLOW_CREDENTIALS = True
CORS_ORIGIN_ALLOW_ALL = True
CORS_ORIGIN_WHITELIST = (
    '*'
)

CORS_ALLOW_METHODS = (
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
    'VIEW',
)

CORS_ALLOW_HEADERS = (
    'XMLHttpRequest',
    'X_FILENAME',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'Pragma',
)
```

