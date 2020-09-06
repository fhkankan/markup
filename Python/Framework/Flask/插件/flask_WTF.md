 # Web表单

HTML页面中负责数据采集的部件。表单有三个部分组成：表单标签、表单域、表单按钮。表单允许用户输入数据，负责HTML页面数据采集，通过表单将用户输入的数据提交给服务器。

## WTForms支持的HTML标准字段

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

## WTForms常用验证函数

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

## 使用Flask_WPF实现表单

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