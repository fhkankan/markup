# 模板

## 基本语法

```
{% if user %}
    {{ user }}
{% else %}
    hello!
<ul>
    {% for index in indexs %}
    <li> {{ index }} </li>
    {% endfor %}
</ul>
```

## 反向路由

```
# url_for()辅助函数可以使用程序URL映射中保存的信息生成URL；url_for()接收视图函数名作为参数，返回对应的URL
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/user/')
def redirect():
    return url_for('index',_external=True)
```

## 自定义错误页面

```
from flask import Flask,render_template

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
```

## 过滤器

```
过滤器的本质就是函数

使用方式为：变量名 | 过滤器 
```

字符串操作

```
# safe：禁用转义
<p>{{ '<em>hello</em>' | safe }}</p>

# capitalize：把变量值的首字母转成大写，其余字母转小写
<p>{{ 'hello' | capitalize }}</p>

# lower：把值转成小写
<p>{{ 'HELLO' | lower }}</p>

# upper：把值转成大写
<p>{{ 'hello' | upper }}</p>

# title：把值中的每个单词的首字母都转成大写
<p>{{ 'hello' | title }}</p>

# trim：把值的首尾空格去掉
<p>{{ ' hello world ' | trim }}</p>

# reverse:字符串反转
<p>{{ 'olleh' | reverse }}</p>
 
# format:格式化输出
<p>{{ '%s is %d' | format('name',17) }}</p>

# striptags：渲染之前把值中所有的HTML标签都删掉
<p>{{ '<em>hello</em>' | striptags }}</p>
```

列表操作

```
# first：取第一个元素
<p>{{ [1,2,3,4,5,6] | first }}</p>

# last：取最后一个元素
<p>{{ [1,2,3,4,5,6] | last }}</p>

# length：获取列表长度
<p>{{ [1,2,3,4,5,6] | length }}</p>

# sum：列表求和
<p>{{ [1,2,3,4,5,6] | sum }}</p>

# sort：列表排序
<p>{{ [6,2,3,1,5,4] | sort }}</p>
```

语句块过滤

```
{% filter upper %}
  this is a Flask Jinja2 introduction
{% endfilter %}
```

自定义过滤器

自定义过滤器有两种实现方式：一种是通过Flask应用对象的add_template_filter方法。还可以通过装饰器来实现自定义过滤器。

**自定义的过滤器名称如果和内置的过滤器重名，会覆盖内置的过滤器。**

```
# 实现方式一：通过调用应用程序实例的add_template_filter方法实现自定义过滤器。该方法第一个参数是函数名，第二个参数是自定义的过滤器名称。
def filter_double_sort(ls):
    return ls[::2]
app.add_template_filter(filter_double_sort,'double_2')

# 实现方式二：用装饰器来实现自定义过滤器。装饰器传入的参数是自定义的过滤器名称。
@app.template_filter('db3')
def filter_double_sort(ls):
    return ls[::-3]
```

## Web表单

HTML页面中负责数据采集的部件。表单有三个部分组成：表单标签、表单域、表单按钮。表单允许用户输入数据，负责HTML页面数据采集，通过表单将用户输入的数据提交给服务器。

WTForms支持的HTML标准字段

| 字段对象                | 说明                         |
| ------------------- | -------------------------- |
| StringField         | 文本字段                       |
| TextAreaField       | 多行文本字段                     |
| PasswordField       | 密码文本字段                     |
| HiddenField         | 隐藏文本字段                     |
| DateField           | 文本字段，值为datetime.date格式     |
| DateTimeField       | 文本字段，值为datetime.datetime格式 |
| IntegerField        | 文本字段，值为整数                  |
| DecimalField        | 文本字段，值为decimal.Decimal     |
| FloatField          | 文本字段，值为浮点数                 |
| BooleanField        | 复选框，值为True和False           |
| RadioField          | 一组单选框                      |
| SelectField         | 下拉列表                       |
| SelectMultipleField | 下拉列表，可选择多个值                |
| FileField           | 文本上传字段                     |
| SubmitField         | 表单提交按钮                     |
| FormField           | 把表单作为字段嵌入另一个表单             |
| FieldList           | 一组指定类型的字段                  |

WTForms常用验证函数

| 验证函数         | 说明                   |
| ------------ | -------------------- |
| DataRequired | 确保字段中有数据             |
| EqualTo      | 比较两个字段的值，常用于比较两次密码输入 |
| Length       | 验证输入的字符串长度           |
| NumberRange  | 验证输入的值在数字范围内         |
| URL          | 验证URL                |
| AnyOf        | 验证输入值在可选列表中          |
| NoneOf       | 验证输入值不在可选列表中         |

使用Flask-WTF需要配置参数SECRET_KEY。CSRF_ENABLED是为了CSRF（跨站请求伪造）保护。 SECRET_KEY用来生成加密令牌，当CSRF激活的时候，该设置会根据设置的密匙生成加密令牌。

HTML页面中的form表单

```
#模板文件
<form method='post'>
    <input type="text" name="username" placeholder='Username'>
    <input type="password" name="password" placeholder='password'>
    <input type="submit">
</form>
```

视图函数中获取表单数据

```
from flask import Flask,render_template,request

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
    	# form是请求上下文对象request的属性，用来获取表单的数据
        username = request.form['username']
        password = request.form.get('password')
        print username,password
    return render_template('login.html',method=request.method)
```

使用Flask_WPF实现表单

```
# 配置参数
app.config['SECRET_KEY'] = 'silents is gold'

# 模板页面
<form method="post">
        #设置csrf_token
        {{ form.csrf_token }}
        {{ form.us.label }}
        <p>{{ form.us }}</p>
        {{ form.ps.label }}
        <p>{{ form.ps }}</p>
        {{ form.ps2.label }}
        <p>{{ form.ps2 }}</p>
        <p>{{ form.submit() }}</p>
        {% for x in get_flashed_messages() %}
            {{ x }}
        {% endfor %}
 </form>

# 视图函数
#coding=utf-8
from flask import Flask,render_template,\
    redirect,url_for,session,request,flash

#导入wtf扩展的表单类
from flask_wtf import FlaskForm
#导入自定义表单需要的字段
from wtforms import SubmitField,StringField,PasswordField
#导入wtf扩展提供的表单验证器
from wtforms.validators import DataRequired,EqualTo
app = Flask(__name__)
app.config['SECRET_KEY']='1'

#自定义表单类，文本字段、密码字段、提交按钮
class Login(FlaskForm):
    us = StringField(label=u'用户：',validators=[DataRequired()])
    ps = PasswordField(label=u'密码',validators=[DataRequired(),EqualTo('ps2','err')])
    ps2 = PasswordField(label=u'确认密码',validators=[DataRequired()])
    submit = SubmitField(u'提交')

@app.route('/login')
def login():
    return render_template('login.html')

#定义根路由视图函数，生成表单对象，获取表单数据，进行表单数据验证
@app.route('/',methods=['GET','POST'])
def index():
    form = Login()
    if form.validate_on_submit():
        name = form.us.data
        pswd = form.ps.data
        pswd2 = form.ps2.data
        print name,pswd,pswd2
        return redirect(url_for('login'))
    else:
        if request.method=='POST':
            flash(u'信息有误，请重新输入！')
        print form.validate_on_submit()

    return render_template('index.html',form=form)
if __name__ == '__main__':
    app.run(debug=True)
```

## 宏(macro)

Jinja2支持宏，还可以导入宏，需要在多处重复使用的模板代码片段可以写入单独的文件，再包含在所有模板中，以避免重复。

**定义宏 **

```
# 无参数
{% macro input() %}
  <input type="text"
         name="username"
         value=""
         size="30"/>
{% endmacro %}

# 有参数
{% macro input(name,value='',type='text',size=20) %}
    <input type="{{ type }}"
           name="{{ name }}"
           value="{{ value }}"
           size="{{ size }}"/>
{% endmacro %}
```

**调用宏 **

```
# 无参数
{{ input() }}

# 有参数
{{ input(value='name',type='password',size=40)}}
```

**抽取封装调用 **

```
# 文件名可以自定义macro.html
{% macro function() %}
    <input type="text" name="username" placeholde="Username">
    <input type="password" name="password" placeholde="Password">
    <input type="submit">
{% endmacro %}

# 在其它模板文件中先导入，再调用
{% import 'macro.html' as func %}
{ {func.function() }}
```

## 继承(extends)

**父模板:base.html **

```
 {% block top %}
    顶部菜单
 {% endblock top %}

 {% block content %}
 {% endblock content %}

 {% block bottom %}
    底部
 {% endblock bottom %}
```

**子模板 **

```
{% extends 'base.html' %}
{% block content %}
   需要填充的内容
{% endblock content %}
```

模板继承使用时注意点：

- 不支持多继承。
- 为了便于阅读，在子模板中使用extends时，尽量写在模板的第一行。
- 不能在一个模板文件中定义多个相同名字的block标签。
- 当在页面中使用多个block标签时，建议给结束标签起个名字，当多个block嵌套时，阅读性更好。

## 包含(include)

功能是将另一个模板整个加载到当前模板中，并直接渲染

```
# include的使用,如果包含的模板文件不存在时，程序会抛出TemplateNotFound异常
{% include 'hello.html' %}

# 加上关键字ignore missing,如果包含的模板文件不存在，会忽略这条include语句
{% include 'hello.html' ignore missing %}
```

宏、继承、包含：

- 宏(Macro)、继承(Block)、包含(include)均能实现代码的复用。
- 继承(Block)的本质是代码替换，一般用来实现多个页面中重复不变的区域。
- 宏(Macro)的功能类似函数，可以传入参数，需要定义、调用。
- 包含(include)是直接将目标模板文件整个渲染出来。

## 特殊变量和方法

有一些特殊的变量和方法是可以在模板文件中直接访问的

**config对象**

```
# config 对象就是Flask的config对象，也就是 app.config 对象。
{{ config.SQLALCHEMY_DATABASE_URI }}
```

**request 对象**

就是 Flask 中表示当前请求的 request 对象，request对象中保存了一次HTTP请求的一切信息。

```
{{ request.url }}
```

常用的属性如下：

| 属性      | 说明              | 类型             |
| ------- | --------------- | -------------- |
| data    | 记录请求的数据，并转换为字符串 | *              |
| form    | 记录请求中的表单数据      | MultiDict      |
| args    | 记录请求中的查询参数      | MultiDict      |
| cookies | 记录请求中的cookie信息  | Dict           |
| headers | 记录请求中的报文头       | EnvironHeaders |
| method  | 记录请求使用的HTTP方法   | GET/POST       |
| url     | 记录请求的URL地址      | string         |
| files   | 记录请求上传的文件       | *              |

**url_for()**

url_for() 会返回传入的路由函数对应的URL，所谓路由函数就是被 app.route() 路由装饰器装饰的函数。如果我们定义的路由函数是带有参数的，则可以将这些参数作为命名参数传入。

```
{{ url_for('index') }}
{{ url_for('post', post_id=1024) }}
```

**get_flashed_messages()**

返回之前在Flask中通过 flash() 传入的信息列表，把字符串对象表示的消息加入到一个消息队列中

```
from flask import Flask, flash, request
if request.method == 'GET':
    # 重启浏览器可以解决编码问题：
    flash(u'请输入用户信息和密码')
```

然后通过调用 get_flashed_messages() 方法取出

```
{% for message in get_flashed_messages() %}
    {{ message }}
{% endfor %}
```

