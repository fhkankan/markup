# CSRF

[参考](https://www.cnblogs.com/renpingsheng/p/9756051.html)

## 原理

```
csrf要求发送post,put或delete请求的时候，是先以get方式发送请求，服务端响应时会分配一个随机字符串给客户端，客户端第二次发送post,put或delete请求时携带上次分配的随机字符串到服务端进行校验
```

## CSRF中间件

首先，我们知道Django中间件作用于整个项目。

在一个项目中，如果想对全局所有视图函数或视图类起作用时，就可以在中间件中实现，比如想实现用户登录判断，基于用户的权限管理（RBAC）等都可以在Django中间件中来进行操作

Django内置了很多中间件,其中之一就是CSRF中间件

```python
MIDDLEWARE_CLASSES = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

上面第四个就是Django内置的CSRF中间件

#### 中间件的执行流程

Django中间件中最多可以定义5个方法

```
process_request
process_response
process_view
process_exception
process_template_response
```

Django中间件的执行顺序

```
1.请求进入到Django后，会按中间件的注册顺序执行每个中间件中的process_request方法
    如果所有的中间件的process_request方法都没有定义return语句，则进入路由映射，进行url匹配
    否则直接执行return语句，返回响应给客户端
2.依次按顺序执行中间件中的process_view方法
    如果某个中间件的process_view方法没有return语句，则根据第1步中匹配到的URL执行对应的视图函数或视图类
    如果某个中间件的process_view方法中定义了return语句，则后面的视图函数或视图类不会执行,程序会直接返回
3.视图函数或视图类执行完成之后，会按照中间件的注册顺序逆序执行中间件中的process_response方法
    如果中间件中定义了return语句，程序会正常执行，把视图函数或视图类的执行结果返回给客户端
    否则程序会抛出异常
4.程序在视图函数或视图类的正常执行过程中
    如果出现异常，则会执行按顺序执行中间件中的process_exception方法
    否则process_exception方法不会执行
    如果某个中间件的process_exception方法中定义了return语句，则后面的中间件中的process_exception方法不会继续执行了
5.如果视图函数或视图类中使用render方法来向客户端返回数据，则会触发中间件中的process_template_response方法
```

## 源码解析

Django CSRF中间件的源码

```python
class CsrfViewMiddleware(MiddlewareMixin):

    def _accept(self, request):
        request.csrf_processing_done = True
        return None

    def _reject(self, request, reason):
        logger.warning(
            'Forbidden (%s): %s', reason, request.path,
            extra={
                'status_code': 403,
                'request': request,
            }
        )
        return _get_failure_view()(request, reason=reason)

    def _get_token(self, request):
        if settings.CSRF_USE_SESSIONS:
            try:
                return request.session.get(CSRF_SESSION_KEY)
            except AttributeError:
                raise ImproperlyConfigured(
                    'CSRF_USE_SESSIONS is enabled, but request.session is not '
                    'set. SessionMiddleware must appear before CsrfViewMiddleware '
                    'in MIDDLEWARE%s.' % ('_CLASSES' if settings.MIDDLEWARE is None else '')
                )
        else:
            try:
                cookie_token = request.COOKIES[settings.CSRF_COOKIE_NAME]
            except KeyError:
                return None

            csrf_token = _sanitize_token(cookie_token)
            if csrf_token != cookie_token:
                # Cookie token needed to be replaced;
                # the cookie needs to be reset.
                request.csrf_cookie_needs_reset = True
            return csrf_token

    def _set_token(self, request, response):
        if settings.CSRF_USE_SESSIONS:
            request.session[CSRF_SESSION_KEY] = request.META['CSRF_COOKIE']
        else:
            response.set_cookie(
                settings.CSRF_COOKIE_NAME,
                request.META['CSRF_COOKIE'],
                max_age=settings.CSRF_COOKIE_AGE,
                domain=settings.CSRF_COOKIE_DOMAIN,
                path=settings.CSRF_COOKIE_PATH,
                secure=settings.CSRF_COOKIE_SECURE,
                httponly=settings.CSRF_COOKIE_HTTPONLY,
            )
            patch_vary_headers(response, ('Cookie',))

    def process_request(self, request):
        csrf_token = self._get_token(request)
        if csrf_token is not None:
            # Use same token next time.
            request.META['CSRF_COOKIE'] = csrf_token

    def process_view(self, request, callback, callback_args, callback_kwargs):
        if getattr(request, 'csrf_processing_done', False):
            return None

        if getattr(callback, 'csrf_exempt', False):
            return None

        if request.method not in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            if getattr(request, '_dont_enforce_csrf_checks', False):
                return self._accept(request)

            if request.is_secure():
                referer = force_text(
                    request.META.get('HTTP_REFERER'),
                    strings_only=True,
                    errors='replace'
                )
                if referer is None:
                    return self._reject(request, REASON_NO_REFERER)

                referer = urlparse(referer)

                if '' in (referer.scheme, referer.netloc):
                    return self._reject(request, REASON_MALFORMED_REFERER)

                if referer.scheme != 'https':
                    return self._reject(request, REASON_INSECURE_REFERER)

                good_referer = (
                    settings.SESSION_COOKIE_DOMAIN
                    if settings.CSRF_USE_SESSIONS
                    else settings.CSRF_COOKIE_DOMAIN
                )
                if good_referer is not None:
                    server_port = request.get_port()
                    if server_port not in ('443', '80'):
                        good_referer = '%s:%s' % (good_referer, server_port)
                else:
                    good_referer = request.get_host()

                good_hosts = list(settings.CSRF_TRUSTED_ORIGINS)
                good_hosts.append(good_referer)

                if not any(is_same_domain(referer.netloc, host) for host in good_hosts):
                    reason = REASON_BAD_REFERER % referer.geturl()
                    return self._reject(request, reason)

            csrf_token = request.META.get('CSRF_COOKIE')
            if csrf_token is None:
                return self._reject(request, REASON_NO_CSRF_COOKIE)

            request_csrf_token = ""
            if request.method == "POST":
                try:
                    request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
                except IOError:
                    pass

            if request_csrf_token == "":
                request_csrf_token = request.META.get(settings.CSRF_HEADER_NAME, '')

            request_csrf_token = _sanitize_token(request_csrf_token)
            if not _compare_salted_tokens(request_csrf_token, csrf_token):
                return self._reject(request, REASON_BAD_TOKEN)

        return self._accept(request)

    def process_response(self, request, response):
        if not getattr(request, 'csrf_cookie_needs_reset', False):
            if getattr(response, 'csrf_cookie_set', False):
                return response

        if not request.META.get("CSRF_COOKIE_USED", False):
            return response

        self._set_token(request, response)
        response.csrf_cookie_set = True
        return response
```

从上面的源码中可以看到，CsrfViewMiddleware中间件中定义了process_request，process_view和process_response三个方法

先来看process_request方法

```python
def _get_token(self, request):  
    if settings.CSRF_USE_SESSIONS:  
        try:  
            return request.session.get(CSRF_SESSION_KEY)  
        except AttributeError:  
            raise ImproperlyConfigured(  
                'CSRF_USE_SESSIONS is enabled, but request.session is not '  
 'set. SessionMiddleware must appear before CsrfViewMiddleware ' 'in MIDDLEWARE%s.' % ('_CLASSES' if settings.MIDDLEWARE is None else '')  
            )  
    else:  
        try:  
            cookie_token = request.COOKIES[settings.CSRF_COOKIE_NAME]  
        except KeyError:  
            return None  
  
  csrf_token = _sanitize_token(cookie_token)  
        if csrf_token != cookie_token:  
            # Cookie token needed to be replaced;  
 # the cookie needs to be reset.  request.csrf_cookie_needs_reset = True  
 return csrf_token

def process_request(self, request):  
        csrf_token = self._get_token(request)  
        if csrf_token is not None:  
            # Use same token next time.  
      request.META['CSRF_COOKIE'] = csrf_token
```

从Django项目配置文件夹中读取`CSRF_USE_SESSIONS`的值，如果获取成功，则`从session中读取CSRF_SESSION_KEY的值`，默认为`'_csrftoken'`，如果没有获取到`CSRF_USE_SESSIONS`的值，则从发送过来的请求中获取`CSRF_COOKIE_NAME`的值，如果没有定义则返回None。

再来看process_view方法

在process_view方法中，先检查视图函数是否被`csrf_exempt`装饰器装饰，如果视图函数没有被csrf_exempt装饰器装饰，则程序继续执行，否则返回None。接着从request请求头中或者cookie中获取携带的token并进行验证，验证通过才会继续执行与URL匹配的视图函数，否则就返回`403 Forbidden`错误。

实际项目中，会在发送POST,PUT,DELETE,PATCH请求时，在提交的form表单中添加

```
{% csrf_token %}
```

即可，否则会出现403的错误

![img](https://img2018.cnblogs.com/blog/1133627/201810/1133627-20181008182137878-2034768573.png)

## 装饰器

`csrf_exempt`表示不进行CSRF验证

`csrf_protect`表示进行CSRF验证

#### 基于Django FBV

在一个项目中，如果注册起用了`CsrfViewMiddleware`中间件，则项目中所有的视图函数和视图类在执行过程中都要进行CSRF验证。

此时想使某个视图函数或视图类不进行CSRF验证，则可以使用`csrf_exempt`装饰器装饰不想进行CSRF验证的视图函数

```python
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  
def index(request):  
    pass
```

也可以把csrf_exempt装饰器直接加在URL路由映射中，使某个视图函数不经过CSRF验证

```python
from django.views.decorators.csrf import csrf_exempt  
  
from users import views  
 
urlpatterns = [  
    url(r'^admin/', admin.site.urls),  
    url(r'^index/', csrf_exempt(views.index)),  
]
```

同样的，如果在一个Django项目中，没有注册起用`CsrfViewMiddleware`中间件，但是想让某个视图函数进行CSRF验证，则可以使用`csrf_protect`装饰器

`csrf_protect`装饰器的用法跟`csrf_exempt`装饰器用法相同，都可以加上视图函数上方装饰视图函数或者在URL路由映射中直接装饰视图函数

```python
from django.views.decorators.csrf import csrf_exempt  

@csrf_protect  
def index(request):  
    pass
```

或者

```python
from django.views.decorators.csrf import csrf_protect  
  
from users import views  
 
urlpatterns = [  
    url(r'^admin/', admin.site.urls),  
    url(r'^index/',csrf_protect(views.index)),  
]
```

#### 基于Django CBV

上面的情况是基于Django FBV的，如果是基于Django CBV，则不可以直接加在视图类的视图函数中了

此时有三种方式来对Django CBV进行CSRF验证或者不进行CSRF验证

- 方法一

在视图类中定义dispatch方法，为dispatch方法加csrf_exempt装饰器

```python
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

class UserAuthView(View):

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super(UserAuthView,self).dispatch(request,*args,**kwargs)

    def get(self,request,*args,**kwargs):
        pass

    def post(self,request,*args,**kwargs):
        pass

    def put(self,request,*args,**kwargs):
        pass

    def delete(self,request,*args,**kwargs):
        pass
```

- 方法二

为视图类上方添加装饰器

```python
@method_decorator(csrf_exempt,name='dispatch')
class UserAuthView(View):
    def get(self,request,*args,**kwargs):
        pass

    def post(self,request,*args,**kwargs):
        pass

    def put(self,request,*args,**kwargs):
        pass

    def delete(self,request,*args,**kwargs):
        pass
```

- 方式三：

在url.py中为类添加装饰器

```python
from django.views.decorators.csrf import csrf_exempt

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^auth/', csrf_exempt(views.UserAuthView.as_view())),
]
```

> csrf_protect装饰器的用法跟上面一样