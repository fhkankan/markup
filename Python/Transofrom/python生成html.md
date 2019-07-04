# python生成html

[参考](https://blog.csdn.net/reallocing1/article/details/51694967)

## 静态HTML

```python
#coding:utf-8
_author_ = "LiaoPan"
_time_  = "2016.6.16"
_myblog_ = "http://blog.csdn.net/reallocing1?viewmode=contents"

f = open("demo_1.html",'w')
message = """
<html>
<head></head>
<body>
<p>Hello,World!</p>
<p>demo</p>
</body>
</html>"""

f.write(message)
f.close()
```

- webbrowser

无变量

```python
#coding:utf-8
_author_ = "LiaoPan"
_time_  = "2016.6.16"
_myblog_ = "http://blog.csdn.net/reallocing1?viewmode=contents"
import webbrowser

GEN_HTML = "demo_1.html"  #命名生成的html

f = open(GEN_HTML,'w')
message = """
<html>
<head></head>
<body>
<p>Hello,World!</p>
<p>Add webbrowser function</p>
</body>
</html>"""

f.write(message)
f.close()

webbrowser.open(GEN_HTML,new = 1)
```

有变量

```python
#coding:utf-8
_author_ = "LiaoPan"
_time_  = "2016.6.16"
_myblog_ = "http://blog.csdn.net/reallocing1?viewmode=contents"
import webbrowser

GEN_HTML = "demo_1.html"  #命名生成的html

str_1 = "1: new contents need to be added."
str_2 = "2: new contents need to be added."

f = open(GEN_HTML,'w')
message = """
<html>
<head></head>
<body>
<p>Hello,World!</p>
<p>Add webbrowser function</p>
<p>%s</p>
<p>%s</p>
</body>
</html>"""%(str_1,str_2)

f.write(message)
f.close()

webbrowser.open(GEN_HTML,new = 1)
```

## 动态生成

### bottle

安装

```
pip install bottle
```

示例

```python
#coding:utf-8
"""
- 使用bottle来动态生成html
    - http://bottlepy.org/docs/dev/stpl.html

"""
_author_ = "LiaoPan"
_time_  = "2016.6.17"
_myblog_ = "http://blog.csdn.net/reallocing1?viewmode=contents"


from bottle import SimpleTemplate,template


#ex01
tpl = SimpleTemplate('Hello {{name}}')

print tpl.render(name='World')

#ex02
print template('Hello {{name}}',name = "World")

#ex03
my_dict = {'Name':'L','Age':23,'Grade':"A"}
print template("My name is {{Name}}, my age is {{Age}},and my grade is {{Grade}}",**my_dict)
```

动态生成

```python
#coding:utf-8
"""
- 使用bottle来动态生成html
    - https://www.reddit.com/r/learnpython/comments/2sfeg0/using_template_engine_with_python_for_generating/

"""
_author_ = "LiaoPan"
_time_  = "2016.6.17"

from bottle import template
import webbrowser

#一些我们需要展示的文章题目和内容
articles = [("Title #1","Detials #1","http://blog.csdn.net/reallocing1/article/details/51694967"),("Title #2","Detials #2","http://music.163.com"),("Title #3","Detials #3","http://douban.fm")]

#定义想要生成的Html的基本格式
#使用%来插入python代码
template_demo="""
<html>
<head><h1>demo of bottle</h1></head>
<title>Demo</title>
<body>


% for title,detail,link in items:
<h2>{{title.strip()}}</h2>
<p>{{detail}}</p>
<a href={{link}}>Link text</a>
%end


</body
</html>
"""

html = template(template_demo,items=articles)

with open("test.html",'wb') as f:
    f.write(html.encode('utf-8'))


#使用浏览器打开html
webbrowser.open("test.html")

```

### PyH

示例python代码

```python
from pyh import *
page = PyH('My wonderful PyH page')
page.addCSS('myStylesheet1.css', 'myStylesheet2.css')
page.addJS('myJavascript1.js', 'myJavascript2.js')
page << h1('My big title', cl='center')
page << div(cl='myCSSclass1 myCSSclass2', id='myDiv1') << p('I love PyH!', id='myP1')
mydiv2 = page << div(id='myDiv2')
mydiv2 << h2('A smaller title') + p('Followed by a paragraph.')
page << div(id='myDiv3')
page.myDiv3.attributes['cl'] = 'myCSSclass3'
page.myDiv3 << p('Another paragraph')
page.printOut()
```

生成Html代码

```python
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>My wonderful PyH page</title>
<link href="myStylesheet1.css" type="text/css" rel="stylesheet" />
<link href="myStylesheet2.css" type="text/css" rel="stylesheet" />
<script src="myJavascript1.js" type="text/javascript"></script>
<script src="myJavascript2.js" type="text/javascript"></script>
</head>
<body>
<h1 class="center">My big title</h1>
<div id="myDiv1" class="myCSSclass1 myCSSclass2">
<p id="myP1">I love PyH!</p>
</div>
<div id="myDiv2">
<h2>A smaller title</h2>
<p>Followed by a paragraph.</p>
</div>
<div id="myDiv3" class="myCSSclass3">
<p>Another paragraph</p>
</div>
</body>
</html>
```

### jinja2

安装

```
pip install Jinjia2
```

使用

```shell
>>> from jinja2 import Template
>>> template = Template('Hello {{ name }}!')
>>> template.render(name='John Doe')
u'Hello John Doe!'
```

