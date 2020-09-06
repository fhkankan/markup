# Django CBV

[参考](https://www.cnblogs.com/renpingsheng/p/9531649.html)

通常来说，http请求的本质就是基于Socket

Django的视图函数，可以基于FBV模式，也可以基于CBV模式。

基于FBV的模式就是在Django的路由映射表里进行url和视图函数的关联，而基于CBV的模式则是在views.py文件中定义视图类，在视图类中定义视图函数，如get,post,put,delete等

## 使用CBV

使用Django新建一个项目，新建一个路由映射

```python
from django.conf.urls import url
from django.contrib import admin
from app01 import views

urlpatterns = [
    url(r'^cbv/$',views.CBV.as_view())
]
```

对应的views.py文件内容：

```python
from django.shortcuts import render,HttpResponse

from django.views import View

class CBV(View):
    def get(self,request):
        return HttpResponse("GET")

    def post(self,request):
        return HttpResponse("POST")
```

启动项目，使用浏览器请求URL`http://127.0.0.1:8000/cbv/`,浏览器显示结果为:

![img](https://images2018.cnblogs.com/blog/1133627/201808/1133627-20180824190006414-600781281.png)

请求到达Django会先执行Django中间件里的方法，然后进行进行路由匹配。

在路由匹配完成后，会执行CBV类中的as_view方法。

CBV中并没有定义`as_view`方法，由于CBV继承自Django的View,所以会执行Django的View类中的`as_view`方法

## view源码

- Django的View类的as_view方法的部分源码

```python
class View(object):
    """
    Intentionally simple parent class for all views. Only implements
    dispatch-by-method and simple sanity checking.
    """

    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    def __init__(self, **kwargs):
        """
        Constructor. Called in the URLconf; can contain helpful extra
        keyword arguments, and other things.
        """
        # Go through keyword arguments, and either save their values to our
        # instance, or raise an error.
        for key, value in six.iteritems(kwargs):
            setattr(self, key, value)

    @classonlymethod
    def as_view(cls, **initkwargs):
        """
        Main entry point for a request-response process.
        """
        for key in initkwargs:
            if key in cls.http_method_names:
                raise TypeError("You tried to pass in the %s method name as a "
                                "keyword argument to %s(). Don't do that."
                                % (key, cls.__name__))
            if not hasattr(cls, key):
                raise TypeError("%s() received an invalid keyword %r. as_view "
                                "only accepts arguments that are already "
                                "attributes of the class." % (cls.__name__, key))

        def view(request, *args, **kwargs):
            self = cls(**initkwargs)
            if hasattr(self, 'get') and not hasattr(self, 'head'):
                self.head = self.get
            self.request = request
            self.args = args
            self.kwargs = kwargs
            return self.dispatch(request, *args, **kwargs)
        view.view_class = cls
        view.view_initkwargs = initkwargs

        # take name and docstring from class
        update_wrapper(view, cls, updated=())

        # and possible attributes set by decorators
        # like csrf_exempt from dispatch
        update_wrapper(view, cls.dispatch, assigned=())
        return view
```

从View的源码可以看出，在View类中，先定义了http请求的八种方法

```
http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']
```

在`as_view`方法中进行判断，如果请求的方法没在`http_method_names`中，则会抛出异常，这里的cls实际上指的是自定义的CBV类

接着`as_view`方法中又定义view方法，在view方法中对CBV类进行实例化，得到self对象，然后在self对象中封装浏览器发送的request请求

```
self = cls(**initkwargs)
```

最后又调用了self对象中的dispatch方法并返回dispatch方法的值来对request进行处理

此时，由于self对象就是CBV实例化得到，所以会先执行自定义的CBV类中的`dispatch`方法。如果CBV类中没有定义`dispatch`方法则执行Django的View中的`dispatch`方法

- Django的View中的dispatch方法源码

```python
class View(object):
    """
    中间省略
    """
    def dispatch(self, request, *args, **kwargs):
        # Try to dispatch to the right method; if a method doesn't exist,
        # defer to the error handler. Also defer to the error handler if the
        # request method isn't on the approved list.
        if request.method.lower() in self.http_method_names:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
        else:
            handler = self.http_method_not_allowed
        return handler(request, *args, **kwargs)
```

在dispatch方法中，把`request.method`转换为小写再判断是否在定义的`http_method_names`中，如果request.method存在于http_method_names中，则使用getattr反射的方式来得到handler

> 在这里的dispatch方法中，self指的是自定义的CBV类实例化得到的对象

从CBV类中获取`request.method`对应的方法，再执行CBV中的方法并返回

由此，可以知道如果在Django项目中使用CBV的模式，实际上调用了getattr的方式来执行获取类中的请求方法对应的函数

结论：

```
CBV基于反射实现根据请求方式不同，执行不同的方法
```

## 自定义dispatch方法

如果想在基于CBV模式的项目中在请求某个url时执行一些操作，则可以在url对应的类中定义dispatch方法

修改views.py文件

```python
class CBV(View):
    def dispatch(self, request, *args, **kwargs):
        func = getattr(self,request.method.lower())
        return func(request,*args,**kwargs)

    def get(self,request):
        return HttpResponse("GET")

    def post(self,request):
        return HttpResponse("POST")
```

也可以使用继承的方式重写dispatch方法：

```python
class CBV(View):
    def dispatch(self, request, *args, **kwargs):
        print("before")
        res = super(CBV, self).dispatch(request, *args, **kwargs)
        print("after")
        return res

    def get(self,request):
        return HttpResponse("GET")

    def post(self,request):
        return HttpResponse("POST")
```

刷新浏览器，Django后台打印结果如下：

![img](https://images2018.cnblogs.com/blog/1133627/201808/1133627-20180824185953364-228146708.png)

浏览器页面结果

![img](https://images2018.cnblogs.com/blog/1133627/201808/1133627-20180824185944346-446758443.png)

同理，如果有基于CBV的多个类，并且有多个类共用的功能，为了避免重复，可以单独定义一个类，在这个类中重写dispatch方法，然后让url对应的视图类继承这个类

修改urls.py文件

```python
from django.conf.urls import url
from django.contrib import admin
from app01 import views

urlpatterns = [
    url(r'^cbv1/$',views.CBV1.as_view()),
    url(r'^cbv2/$',views.CBV2.as_view()),
]
```

views.py文件内容

```python
from django.shortcuts import render,HttpResponse

from django.views import View

class BaseView(object):
    def dispatch(self, request, *args, **kwargs):
        func = getattr(self, request.method.lower())
        return func(request, *args, **kwargs)

class CBV1(BaseView,View):
    def get(self,request):
        return HttpResponse("CBV1 GET")

    def post(self,request):
        return HttpResponse("CBV1 POST")

class CBV2(BaseView,View):
    def get(self,request):
        return HttpResponse("CBV2 GET")

    def post(self,request):
        return HttpResponse("CBV2 POST")
```

通过python的面向对象可以知道，请求到达视图类时，会先执行CBV1和CBV2类中的dispatch方法，然而CBV1和CBV2类中并没有dispatch方法，则会按照顺序在父类中查找dispatch方法，此时就会执行BaseView类中的dispatch方法了

用浏览器请求url`http://127.0.0.1:8000/cbv1/`，浏览器页面显示

![img](https://images2018.cnblogs.com/blog/1133627/201808/1133627-20180824185920397-1210166817.png)

用浏览器请求url`http://127.0.0.1:8000/cbv2/`，浏览器页面显示

![img](https://images2018.cnblogs.com/blog/1133627/201808/1133627-20180824185909792-1783158074.png)