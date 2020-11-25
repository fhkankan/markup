# Flask_RESTful

## 概述

Flask-RESTful是用于快速构建REST API的Flask扩展。

安装

```
pip install flask-restful
```

起步

```python
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class HelloWorldResource(Resource):
    def get(self):
        return {'hello': 'world'}

        def post(self):
        return {'msg': 'post hello world'}

api.add_resource(HelloWorldResource, '/')

# 此处启动对于1.0之后的Flask可有可无
if __name__ == '__main__':
    app.run(debug=True)
```

## 视图

### 为路由起名

通过endpoint参数为路由起名

```python
api.add_resource(HelloWorldResource, '/', endpoint='HelloWorld')
```

### 蓝图中使用

```python
from flask import Flask, Blueprint
from flask_restful import Api, Resource

app = Flask(__name__)

user_bp = Blueprint('user', __name__)

user_api = Api(user_bp)

class UserProfileResource(Resource):
    def get(self):
        return {'msg': 'get user profile'}

user_api.add_resource(UserProfileResource, '/users/profile')

app.register_blueprint(user_bp)
```

### 装饰器

使用`method_decorators`添加装饰器

- 为类视图中的所有方法添加装饰器

```python
def decorator1(func):
    def wrapper(*args, **kwargs):
        print('decorator1')
        return func(*args, **kwargs)
    return wrapper


def decorator2(func):
    def wrapper(*args, **kwargs):
        print('decorator2')
        return func(*args, **kwargs)
    return wrapper


class DemoResource(Resource):
    method_decorators = [decorator1, decorator2]

    def get(self):
        return {'msg': 'get view'}

    def post(self):
        return {'msg': 'post view'}
```

- 为类视图中不同的方法添加不同的装饰器

```python
class DemoResource(Resource):
    method_decorators = {
        'get': [decorator1, decorator2],
        'post': [decorator1]
    }
    
    # 使用了decorator1 decorator2两个装饰器
    def get(self):
        return {'msg': 'get view'}
    
    # 使用了decorator1 装饰器
    def post(self):
        return {'msg': 'post view'}
    
    # 未使用装饰器
    def put(self):
        return {'msg': 'put view'}
```

## 请求

Flask-RESTful 提供了`RequestParser`类，用来帮助我们检验和转换请求数据。

```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('rate', type=int, help='Rate cannot be converted', location='args')
parser.add_argument('name')
args = parser.parse_args()
```

### 使用步骤

```
1. 创建`RequestParser`对象
2. 向`RequestParser`对象中添加需要检验或转换的参数声明
3. 使用`parse_args()`方法启动检验处理
4. 检验之后从检验结果中获取参数时可按照字典操作或对象属性操作
args.rate 或 args['rate']
```

### 参数说明

- required

描述请求是否一定要携带对应参数，**默认值为False**

True 强制要求携带

若未携带，则校验失败，向客户端返回错误信息，状态码400

False 不强制要求携带

若不强制携带，在客户端请求未携带参数时，取出值为None

```python
class DemoResource(Resource):
    def get(self):
        rp = RequestParser()
        rp.add_argument('a', required=False)
        args = rp.parse_args()
        return {'msg': 'data={}'.format(args.a)}
```

- help

参数检验错误时返回的错误描述信息

```python
rp.add_argument('a', required=True, help='missing a param')
```

- action

描述对于请求参数中出现多个同名参数时的处理方式

`action='store'` 保留出现的第一个， 默认

`action='append'` 以列表追加保存所有同名参数的值

```python
rp.add_argument('a', required=True, help='missing a param', action='append')
```

- type

描述参数应该匹配的类型，可以使用python的标准数据类型string、int，也可使用Flask-RESTful提供的检验方法，还可以自己定义

**1标准类型**

```python
rp.add_argument('a', type=int, required=True, help='missing a param', action='append')
```

**2Flask-RESTful提供**

检验类型方法在`flask_restful.inputs`模块中
```python
- url
- regex(指定正则表达式)
from flask_restful import inputs
rp.add_argument('a', type=inputs.regex(r'^\d{2}&'))
- natural 自然数0、1、2、3...
- positive 正整数 1、2、3...
- int_range(low ,high)整数范围
rp.add_argument('a', type=inputs.int_range(1, 10))
- boolean
```
**3自定义**

```python
def mobile(mobile_str):
    """
    检验手机号格式
    :param mobile_str: str 被检验字符串
    :return: mobile_str
    """
    if re.match(r'^1[3-9]\d{9}$', mobile_str):
        return mobile_str
    else:
        raise ValueError('{} is not a valid mobile'.format(mobile_str))

rp.add_argument('a', type=mobile)
```
- location

描述参数应该在请求数据中出现的位置

```python
# Look only in the POST body
parser.add_argument('name', type=int, location='form')

# Look only in the querystring
parser.add_argument('PageSize', type=int, location='args')

# From the request headers
parser.add_argument('User-Agent', location='headers')

# From http cookies
parser.add_argument('session_id', location='cookies')

# From json
parser.add_argument('user_id', location='json')

# From file uploads
parser.add_argument('picture', location='files')
```

也可指明多个位置

```python
parser.add_argument('text', location=['headers', 'json'])
```

## 响应

### 序列化数据

Flask-RESTful 提供了marshal工具，用来帮助我们将数据序列化为特定格式的字典数据，以便作为视图的返回值。

```python
from flask_restful import Resource, fields, marshal_with

resource_fields = {
    'name': fields.String,
    'address': fields.String,
    'user_id': fields.Integer
}

class Todo(Resource):
    @marshal_with(resource_fields, envelope='resource')
    def get(self, **kwargs):
        return db_get_todo()
```

也可以不使用装饰器的方式

```python
class Todo(Resource):
    def get(self, **kwargs):
        data = db_get_todo()
        return marshal(data, resource_fields)
```

示例

```python
# 用来模拟要返回的数据对象的类
class User(object):
    def __init__(self, user_id, name, age):
        self.user_id = user_id
        self.name = name
        self.age = age

resoure_fields = {
        'user_id': fields.Integer,
        'name': fields.String
    }

class Demo1Resource(Resource):
    @marshal_with(resoure_fields, envelope='data1')
    def get(self):
        user = User(1, 'itcast', 12)
        return user

class Demo2Resource(Resource):
    def get(self):
        user = User(1, 'itcast', 12)
        return marshal(user, resoure_fields, envelope='data2')
```

### 自定义JSON格式

- 需求

想要接口返回的JSON数据具有如下统一的格式

```json
{"message": "描述信息", "data": {要返回的具体数据}}
```

在接口处理正常的情况下， message返回ok即可，但是若想每个接口正确返回时省略message字段

```python
class DemoResource(Resource):
    def get(self):
        return {'user_id':1, 'name': 'itcast'}
```

对于诸如此类的接口，能否在某处统一格式化成上述需求格式？

```json
{"message": "OK", "data": {'user_id':1, 'name': 'itcast'}}
```

- 解决

**Flask-RESTful的Api对象提供了一个`representation`的装饰器，允许定制返回数据的呈现格式**

```python
api = Api(app)

@api.representation('application/json')
def handle_json(data, code, headers):
    # TODO 此处添加自定义处理
    return resp
```

Flask-RESTful原始对于json的格式处理方式如下：

代码出处：`flask_restful.representations.json`

```python
from flask import make_response, current_app
from flask_restful.utils import PY3
from json import dumps


def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""

    settings = current_app.config.get('RESTFUL_JSON', {})

    # If we're in debug mode, and the indent is not set, we set it to a
    # reasonable value here.  Note that this won't override any existing value
    # that was set.  We also set the "sort_keys" value.
    if current_app.debug:
        settings.setdefault('indent', 4)
        settings.setdefault('sort_keys', not PY3)

    # always end the json dumps with a new line
    # see https://github.com/mitsuhiko/flask/pull/1262
    dumped = dumps(data, **settings) + "\n"

    resp = make_response(dumped, code)
    resp.headers.extend(headers or {})
    return resp
```

为满足需求，做如下改动即可

```python
@api.representation('application/json')
def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""

    # 此处为自己添加***************
    if 'message' not in data:
        data = {
            'message': 'OK',
            'data': data
        }
    # **************************

    settings = current_app.config.get('RESTFUL_JSON', {})

    # If we're in debug mode, and the indent is not set, we set it to a
    # reasonable value here.  Note that this won't override any existing value
    # that was set.  We also set the "sort_keys" value.
    if current_app.debug:
        settings.setdefault('indent', 4)
        settings.setdefault('sort_keys', not PY3)

    # always end the json dumps with a new line
    # see https://github.com/mitsuhiko/flask/pull/1262
    dumped = dumps(data, **settings) + "\n"

    resp = make_response(dumped, code)
    resp.headers.extend(headers or {})
    return resp
```