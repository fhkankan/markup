# Django中请求的生命周期

[参考]((https://www.cnblogs.com/renpingsheng/p/7534897.html))

## 概述

首先我们知道HTTP请求及服务端响应中传输的所有数据都是字符串.

在Django中,当我们访问一个的url时,会通过路由匹配进入相应的html网页中.

Django的请求生命周期是指当用户在浏览器上输入url到用户看到网页的这个时间段内,Django后台所发生的事情

而Django的生命周期内到底发生了什么呢??

```
1. 当用户在浏览器中输入url时,浏览器会生成请求头和请求体发给服务端
请求头和请求体中会包含浏览器的动作(action),这个动作通常为get或者post,体现在url之中.

2. url经过Django中的wsgi,再经过Django的中间件,最后url到过路由映射表,在路由中一条一条进行匹配,
一旦其中一条匹配成功就执行对应的视图函数,后面的路由就不再继续匹配了.
3. 视图函数根据客户端的请求查询相应的数据.返回给Django,然后Django把客户端想要的数据做为一个字符串返回给客户端.
4. 客户端浏览器接收到返回的数据,经过渲染后显示给用户.
```

视图函数根据客户端的请求查询相应的数据后.如果同时有多个客户端同时发送不同的url到服务端请求数据

服务端查询到数据后,怎么知道要把哪些数据返回给哪个客户端呢??

因此客户端发到服务端的url中还必须要包含所要请求的数据信息等内容.

例如,`http://www.aaa.com/index/?nid=user`这个url中,
客户端通过get请求向服务端发送的`nid=user`的请求,服务端可以通过`request.GET.get("nid")`的方式取得nid数据

客户端还可以通过post的方式向服务端请求数据.

当客户端以post的方式向服务端请求数据的时候,请求的数据包含在请求体里,这时服务端就使用request.POST的方式取得客户端想要取得的数据

```
需要注意的是,request.POST是把请求体的数据转换一个字典,请求体中的数据默认是以字符串的形式存在的.
```

## 视图模式

- 一个url对应一个视图函数,这个模式叫做FBV(`Function Base Views`)

- 一个url对应一个类，模式叫做CBV(`Class Base views`),即

例子:使用cbv模式来请求网页

路由信息:

```python
urlpatterns = [
    url(r'^fbv/',views.fbv),
    url(r'^cbv/',views.CBV.as_view()),
]
```

视图函数配置:

```python
from django.views import View

class CBV(View):
    def get(self,request):
        return render(request, "cbv.html")

    def post(self,request):
        return HttpResponse("cbv.get")
```

cbv.html网页的内容:

```html
<body>
<form method="post" action="/cbv/">
    {% csrf_token %}
    <input type="text">
    <input type="submit">
</form>
</body>
```

请求与响应

```
浏览器中输入:http://127.0.0.1:8000/cbv/，回车
响应：html页面
在input框中输入"hell0"，回车
响应：字符串“cbv.get”
```

使用fbv的模式,在url匹配成功之后,会直接执行对应的视图函数.

而如果使用cbv模式,在url匹配成功之后,会找到视图函数中对应的类,然后这个类回到请求头中找到对应的`Request Method`.

```
如果是客户端以post的方式提交请求,就执行类中的post方法;
如果是客户端以get的方式提交请求,就执行类中的get方法
```

然后查找用户发过来的url,然后在类中执行对应的方法查询生成用户需要的数据.

### fbv方式请求的过程

用户发送url请求,Django会依次遍历路由映射表中的所有记录,一旦路由映射表其中的一条匹配成功了,
就执行视图函数中对应的函数名,这是fbv的执行流程

### cbv方式请求的过程

当服务端使用cbv模式的时候,用户发给服务端的请求包含url和method,这两个信息都是字符串类型

服务端通过路由映射表匹配成功后会自动去找dispatch方法,然后Django会通过dispatch反射的方式找到类中对应的方法并执行

类中的方法执行完毕之后,会把客户端想要的数据返回给dispatch方法,由dispatch方法把数据返回经客户端

例子,把上面的例子中的视图函数修改成如下:

```python
from django.views import View

class CBV(View):
    def dispatch(self, request, *args, **kwargs):
        print("dispatch......")
        res=super(CBV,self).dispatch(request,*args,**kwargs)
        return res

    def get(self,request):
        return render(request, "cbv.html")

    def post(self,request):
        return HttpResponse("cbv.get")
```

打印结果:

```
<HttpResponse status_code=200, "text/html; charset=utf-8">
dispatch......
<HttpResponse status_code=200, "text/html; charset=utf-8">
```

需要注意的是:

```
以get方式请求数据时,请求头里有信息,请求体里没有数据
以post请求数据时,请求头和请求体里都有数据.    
```

## 响应内容

http提交数据的方式有`"post"`,`"get"`,`"put"`,`"patch"`,`"delete"`,`"head"`,`"options"`,`"trace"`.

提交数据的时候,服务端依据method的不同会触发不同的视图函数.

```
对于from表单来说,提交数据只有get和post两种方法
```

另外的方法可以通过Ajax方法来提交

服务端根据个人请求信息的不同来操作数据库,可以使用原生的SQL语句,也可以使用Django的ORM语句.

Django从数据库中查询处理完用户想要的数据,将结果返回给用户.

从Django中返回的响应内容包含响应头和响应体

在Django中,有的时候一个视图函数,执行完成后会使用HttpResponse来返回一个字符串给客户端.
这个字符串只是响应体的部分,返回给客户端的响应头的部分应该怎么设置呢???

为返回给客户端的信息加一个响应头:

修改上面例子的视图函数为如下:

```python
from django.views import View

class CBV(View):
    def dispatch(self, request, *args, **kwargs):
        print("dispatch......")
        res=super(CBV,self).dispatch(request,*args,**kwargs)
        print(res)

        return res

    def get(self,request):
        return render(request, "cbv.html")

    def post(self,request):

        res=HttpResponse("cbv.post")
        res.set_cookie("k2","v2")
        res.set_cookie("k4","v4")

        print("res:",res)
        print("request.cookie:",request.COOKIES)
        return res
```

打印的信息:

```
res: <HttpResponse status_code=200, "text/html; charset=utf-8">
request.cookie: {'csrftoken': 'jmX9H1455MYzDRQs8cQLrA23K0aCGoHpINL50GnMVxhUjamI8wgmOP7D2wXcpjHb', 'k2': 'v2', 'k4': 'v4'}
```