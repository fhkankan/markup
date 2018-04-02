# 模板功能

```
作用：生成html界面内容，模版致力于界面如何显示，而不是程序逻辑。模板不仅仅是一个html文件，还包括了页面中的模板语言。
1) 静态内容：css,js,html。
2) 动态内容：通过模板语言，动态生成一些网页内容

模板使用： 在视图函数中，使用模板产生html内容返回给客户端
方式一： 
1) 加载模板文件（loader.get_template）
2) 创建请求上下文对象， 设置模板显示的数据
3) 模板渲染， 产生html页面内容（render）
4) 返回HttpResponse对象，响应浏览器
方式二：
通过render()函数
```

## 模板加载流程

Django会依次到以下目录查找模板文件，如果都找不到，则报错：

```
1. 项目配置的模板目录
2. admin应用的templates模板目录
3. auth应用的templates模板目录
4. 应用本身的templates模板目录
```

## 模板语言

DTL.（Django Template Language）简称： 模板语言

### 模板变量

```
# 格式：	{{ 变量名 }}

# 模板变量名： 是由数字，字母，下划线组成，不能以下划线开头。

# 点(.) 也在会变量部分中出现， 点号（.）用来访问变量的属性。
当模版系统遇到点(".")，它将以这样的顺序查询： 
  - 字典查询（Dictionary lookup）
  - 属性或方法查询（Attribute or method lookup） （调用方法时不能传参）
  - 数字索引查询（Numeric index lookup）
如果模板变量不存在，则生成html内容时用 '' 空字符串代替。
```

### 模板标签

- [官方文档--内置标签](http://python.usyiyi.cn/documents/django_182/ref/templates/builtins.html)

代码段

```
  {% 代码段 %}
```
for 循环
```
# 遍历列表：
{% for x in 列表 %}
     列表不为空时执行
{% empty %}
     列表为空时执行
{% endfor %}
    
# 若加上关键字reversed则倒序遍历： 
{% for x in 列表 reversed %}
{% endfor %}
        
# 遍历字典：    	
{% for key, value in my_dict.items %}
     {{ key }}: {{ value }}
{% endfor %}
```
forloop
```
# 可以通过 {{ forloop.counter }} 判断for循环执行到第几次，初始化值从1开始。

forloop.counter			#索引从1开始
forloop.counter0		#索引从0开始
forloop.revcounter		#索引从最大长度到1
forloop.revcounter0		#索引从最大长度到0
forloop.first			#当遍历的元素为第一项时为真
forloop.last			#当遍历的元素为最后一项时为真
forloop.parentloop		#用在嵌套的for循环中，获取上一侧for循环的forloop
```
if 语句
```
{% if 条件 %}
{% elif 条件 %}
{% else %}
{% endif %}
```

关系比较操作符：

```
<     >=     <=     ==     != 
注意：进行比较操作时，比较操作符两边必须有空格。
```

逻辑运算： 

```
not   and   or
```

### 过滤器

- [官方文档--过滤器](http://python.usyiyi.cn/documents/django_182/ref/templates/builtins.html)
- 过滤器： **用于对模板变量进行操作**，使用格式：

```
模板变量|过滤器：参数
```

#### 内置过滤器

```
- date： 改变日期的显示格式。  
value|date:"Y年m月j日  H时i分s秒"

- length： 求长度。字符串，列表，元组，字典长度

- default： 设置模板变量的默认值。
  data|default:'默认值'
  
Y表示年，格式为4位，y表示两位的年。
m表示月，格式为01,02,12等。
d表示日, 格式为01,02等。
j表示日，格式为1,2等。
H表示时，24进制，h表示12进制的时。
i表示分，为0-59。
s表示秒，为0-59。
```

#### 自定义过滤器

```python
1）在应用中创建templatetags目录，当前示例为"booktest/templatetags"，创建_init_文件，内容为空。

2）在"booktest/templatetags"目录下创建filters.py文件，代码如下：
#导入Library类
from django.template import Library
#创建一个Library类对象
register=Library()
#使用装饰器进行注册
@register.filter
#定义求余函数mod，将value对2求余
def mod(value, num):
    return value%num

3）在templates/booktest/temp_filter.html中，使用自定义过滤器。
- 首先使用load标签引入模块。
{%load filters%}
- 在遍历时根据编号判断奇偶，代码改为如下：
{% for book in list %}
{% if book.id|mode:3 %}
...
{% else %}
...
{% endif %}
{% endfor %}
```

### 模板注释

单行注释：

	{# 注释内容 #}

多行注释：

	{% comment %}
	  注释内容
	{% endcomment %}

## 模板继承

- 父模板

```
# 在父模板里可以定义块：
{% block 块名 %}
	块中的内容（也可以没有）
{% endblock 块名%}
```

- 子模板

```
# 在子模板头部声明继承父模板：
{% extends 父模板文件路径 %}

# 在子模板中，重写父模板中的块（也可以不重写）：
{% block 块名 %}
	{{ block.super}} #获取父模板中块的默认内容
	重写的内容
{% endblock 块名%}

```

## html转义

### 字符串转义介绍

- 编程语言: 转义字符（Escape character）

在编程语言中：以\开头的，用来表示ascii码中不可见的或者有特殊含义的字符，如\t,\n等。

在python中，如果只想显示原始字符串，不想让转义字符生效，可以使用`r或R`来定义。如：

```
print(r'\t\r)
>>>\t\r
```

- HTML: 转义字符

转义字符串（Escape Sequence）也称字符实体(Character Entity)

定义HTML转义字符串的原因：

1. 像“<”和“>”这类符号已经用来表示HTML标签，因此就不能在html文档内容中直接使用，需要使用转义字符； 

| 字符    | 十进制(;) | 转义字符(;) |
| ----- | ------ | ------- |
| ''    | &#34   | &quto   |
| &     | &#38   | &amp    |
| <     | &#60   | &lt     |
| >     | &#62   | $gt     |
| 不断开空格 | &#160  | &nbsp   |

2. 有些字符在ASCII字符集中没有定义，因此需要使用转义字符串来表示。[HTML特殊转义字符对照表](http://tool.oschina.net/commons?type=2)

- 转义字符的作用：

在编程语言中： 将普通字符转换为有特殊含义或不可见的字符，如`\t, \n`等

在正则表达式、或html等标记语言中： 将有特殊含义的字符转换回原来的意义，如：html中的`&amp; &lt; &gt;`等，转换回` & < >`

### Django中html转义

- 通过RequestContext（render函数）传递给html的数据，如果含有特殊字符，默认是会被转义的。

- 要关闭模板上下文字符串的转义：

    ```
    {{ 模板变量|safe}}

    # 亦可以
    {% autoescape off %}
    	模板语言代码
    {% endautoescape %}
    ```


- 模板硬编码中的字符串默认不会经过转义，如果需要转义，那需要手动进行转义。


## url动态引用

url逆向解析， 反向解析

### url标签

- 问题

当urls.py中的一个url配置项发生改变后，项目中所有硬编码引用该url链接的地方，都需要作修改。

```
# project下的urls.py
urlpatterns = [
	...		
    url(r'^', include('app01.urls')),
]

# app01下的urls.py
urlpatterns = [
    # 首页
    url(r'^index/$', views.index),

	url(r'^reverse/$', views.url_reverse),
]
```
三种url硬编码问题（不带参数，位置参数，关键字参数）： 


	硬编码: <br/>
	<a href="/index">进入首页</a> <br>
	<a href="/show_news/1/2">进入新闻</a> <br>
	<a href="/show_news2/1/2">进入新闻2</a> <br>

- 解决方法： 

1. 给url配置项起个名字，在html界面中，再通过名字引用该url：

```
# project下的urls.py
urlpatterns = [
    ...     
    url(r'^', include('app01.urls', namespace='应用名')),
]

# app01下的urls.py
urlpatterns = [
	...
    url(r'^index/$', views.index, name='url名称'),
]
```

2. 在html界面中，通过url标签进行动态引用

```
{% url '应用名:url名称' %}
{% url '应用名:url名称' 位置参数1 位置参数2 %}
{% url '应用名:url名称' 关键字参数1 关键字参数2 %}
```

3. 参考： 

```
{% url 'app01:index' %}
{% url 'app01:show_news' 1 2 %}
{% url 'app01:show_news2' category=1 pageNo=2 %}
```
### reverse函数 

- 问题： 类似的，在python代码中，同样存在上面所说的url硬编码不方便维护的问题


```
# views.py
def url_reverse(request):
	# url正则配置项修改后，此处url硬编码代码需要修改

    return redirect("/index/")
    # return redirect("/show_news/1/2")
    # return redirect("/show_news2/1/2")
```

- 解决：使用reverse函数，动态生成url。
```
# views.py
def url_reverse(request):
	# 动态引用
    # url = reverse("应用名:url名称")		
    # url = reverse("应用名:url名称", args=[位置参数])
    # url = reverse("应用名:url名称", kwargs={关键字参数})

    return redirect(url)	
```

## CSRF防护

一、什么是CSRF？  

CSRF： Cross-site request forgery，跨站请求伪造  

- 用户登录了正常的网站A， 然后再访问某恶意网站，该恶意网站上有一个指向网站A的链接，那么当用户点击该链接时，则恶意网站能成功向网站A发起一次请求，实际这个请求并不是用户想发的，而是伪造的，而网站A并不知道。

- 攻击者利用了你的身份，以你的名义发送恶意请求，比如：以你名义发送邮件，发消息，盗取你的账号，甚至于购买商品，虚拟货币转账等。

- 如果想防止CSRF，首先是重要的信息传递都采用POST方式而不是GET方式，接下来就说POST请求的攻击方式以及在Django中的避免

二、CSRF防护

1. 重要信息如金额、积分等的获取，采用POST请求
2. 开启CSRF中间件（默认就是开启的）

```
# 项目下的setting.py
MIDDLEWARE_CLASSES = (
	...
	# 开启csrf中间件（默认是开启的）
    'django.middleware.csrf.CsrfViewMiddleware',
	...
)
```

3. 表单post提交数据时加上 {% csrf_token %} 标签

```
<from>
{% csrf_token %} 
...
</from>
```

三、防御原理【了解】

- 服务器在渲染模板文件时，会在html页面中生成一个名字叫做 `csrfmiddlewaretoken` 的隐藏域。  


- 服务器会让浏览器保存一个名字为 `csrftoken` 的cookie信息 
- post提交数据时，两个值都会发给服务器，服务器进行比对，如果一样，则csrf验证通过，否则提示403 Forbidden 


## 验证码

作用：在用户注册、登录页面，为了防止暴力请求，可以加入验证码功能。如果验证码错误，则不需要继续处理，可以减轻业务服务器、数据库服务器的压力。

**案例：**

- 需求： 使用第三方Pillow包实现验证码

- 实现步骤： 


一、显示验证码： 

```
1. 使用Pillow中的api生成验证码
pip install Pillow==3.4.1 

2. 定义生成验证码的视图函数
- 提示1：随机生成字符串后存入session中，用于后续判断  
- 提示2：视图返回mime-type为image/png  

from PIL import Image, ImageDraw, ImageFont
from django.utils.six import BytesIO
# /verify_code/
def create_verify_code(request):
    """使用Pillow包生成验证码图片"""

    # 引入随机函数模块
    import random
    # 定义变量，用于画面的背景色、宽、高
    bgcolor = (random.randrange(20, 100), 
               random.randrange(20, 100), 255)  # RGB
    
    width = 100
    height = 25
    # 创建画面对象
    im = Image.new('RGB', (width, height), bgcolor)
    # 创建画笔对象
    draw = ImageDraw.Draw(im)
    
    # 调用画笔的point()函数绘制噪点
    for i in range(0, 100):
        xy = (random.randrange(0, width), random.randrange(0, height))
        fill = (random.randrange(0, 255), 255, random.randrange(0, 255))
        draw.point(xy, fill=fill)
        
    # 定义验证码的备选值
    str1 = 'ABCD123EFGHIJK456LMNOPQRS789TUVWXYZ0'
    
    # 随机选取4个值作为验证码
    rand_str = ''
    for i in range(0, 4):
        rand_str += str1[random.randrange(0, len(str1))]
        
    # 构造字体对象，ubuntu的字体路径为“/usr/share/fonts/truetype/freefont”
    font = ImageFont.truetype('FreeMono.ttf', 23)
    # 构造字体颜色
    fontcolor = (255, random.randrange(0, 255), random.randrange(0, 255))
    
    # 绘制4个字
    draw.text((5, 2), rand_str[0], font=font, fill=fontcolor)
    draw.text((25, 2), rand_str[1], font=font, fill=fontcolor)
    draw.text((50, 2), rand_str[2], font=font, fill=fontcolor)
    draw.text((75, 2), rand_str[3], font=font, fill=fontcolor)
    
    # 释放画笔
    del draw
    # 存入session，用于做进一步验证
    request.session['verifycode'] = rand_str
    # 内存文件操作
    buf = BytesIO()
    # 将图片保存在内存中，文件类型为png
    im.save(buf, 'png')
    # 将内存中的图片数据返回给客户端，MIME类型为图片png
    return HttpResponse(buf.getvalue(), 'image/png')
    
3. 在浏览器中测试： http://127.0.0.1:8000/create_verify_code/
```

二、进入验证码界面

```
1. 配置url
2. 定义视图函数
3. 编写 html 界面，显示验证码
```

三、提交并校验验证码

- 使用post提交验证码，post提交需要添加csrf_token标签
- 读取用户提交的验证码以及session中保存的验证码进行比较