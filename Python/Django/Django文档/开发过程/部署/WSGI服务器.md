# 如何使用WSGI部署

Django最重要的部署平台是[WSGI](http://www.wsgi.org/)，它是Python Web服务器和应用的标准。

Django的`startproject`管理命令会生成一个简单的默认WSGI 配置，你可以根据项目的需要做调整并指定任何与WSGI兼容的应用服务器使用。

Django包含以下WSGI服务器的入门文档：

- [如何使用Django与Apache和`mod_wsgi`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/howto/deployment/wsgi/modwsgi.html)
- [从Apache 中利用Django 的用户数据库进行认证](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/howto/deployment/wsgi/apache-auth.html)
- [如何使用Gunicorn部署Django](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/howto/deployment/wsgi/gunicorn.html)
- [如何使用uWSGI部署Django](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/howto/deployment/wsgi/uwsgi.html)

## `application`对象

使用WSGI 部署的核心概念是`application` 可调用对象，应用服务器使用它来与你的代码进行交互。 在Python 模块中，它通常一个名为`application` 的对象提供给服务器使用。

`startproject`命令创建一个`<project_name>/wsgi.py` 文件，它就包含这样一个`application` 可调用对象。

它既可用于Django 的开发服务器，也可以用于线上WSGI 的部署。

WSGI 服务器从它们的配置中获得`application` 可调用对象的路径。 Django的内置服务器，即`runserver`命令，从`WSGI_APPLICATION`设置中读取。 默认情况下，它设置为`application`，指向`<project_name>/wsgi.py` 中的`<project_name>.wsgi.application` 可调用对象。

## 配置设置模块

当WSGI 服务器加载你的应用时，Django 需要导入settings 模块 —— 这里是你的全部应用定义的地方。

Django 使用`DJANGO_SETTINGS_MODULE`环境变量来定位settings 模块。 它包含settings 模块的路径，以点分法表示。 您可以使用不同的价值进行开发和生产；这一切都取决于你如何组织你的设置。

如果这个变量没有设置，默认的`wsgi.py` 设置为`mysite`，其中`mysite.settings` 为你的项目的名称。 这是`runserver`如何找到默认的settings 文件的机制。

> 注

因为环境变量是进程范围的，当你在同一个进程中运行多个Django 站点时，它将不能工作。 使用mod_wsgi 就是这个情况。

为了避免这个问题，可以使用mod_wsgi 的守护进程模式，让每个站点位于它自己的守护进程中，或者在`wsgi.py`中通过强制使用`os.environ["DJANGO_SETTINGS_MODULE"] = "mysite.settings"` 来覆盖这个值。

## 应用WSGI中间件

你可以简单地封装application 对象来运用 [WSGI 中间件](https://www.python.org/dev/peps/pep-3333/#middleware-components-that-play-both-sides)。 例如，你可以在`wsgi.py` 的底下添加以下这些行：

```python
from helloworld.wsgi import HelloWorldApplication
application = HelloWorldApplication(application)
```

如果你结合使用 Django 的application 与另外一个WSGI application 框架，你还可以替换Django WSGI 的application 为一个自定义的WSGI application。
