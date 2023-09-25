# flask_babel

翻译插件

## 安装

```shell
pip install flask_babel
```

## 配置

```python
from flask import. Flask
from flask_babel import Babel

app = Flask(__name__)
app.config['BABEL_DEFAULT_LOCALE'] = 'zh'  # 默认语言：zh中文，en英文，ja日文
babel = Babel(app)

# 注意app和babel在一个页面/目录下，这样可以正常加载translations
```

## 使用

```python
# 请求内定义字符串
from flask_babel import gettext as _, refresh

@app.route("/")
def index():
		res = dict(code=0, data={}, msg=_("hello")) 
		return res
 
# 请求外定义字符串
from flask_babel import lazy_gettext as _, refresh

msg = _("hello")

@app.route("/")
def index():
		res = dict(code=0, data={}, msg=msg)  
		return res
```

根据语言判断

```python
@babel.localeselector
def get_locale():
    lang = request.cookies.get("locale")
    if lang in ["en", "zh"]:
        return cookie
    return request.accept_languages.best_match(["zh", "en"])  # 通过headers中的Accept-language来匹配
```

设置语言

```python
@app.route("/set_locale")
def set_locale():
    lang = request.args.get("language")
    response = make_response(jsonify(message=lang))
    if lang == "zh":
        refresh()
        response.set_cookie("locale", "zh")
        return response
    elif lang == "en":
        refresh()
        response.set_cookie("locale", "en")
        return response
    return jsonify({"data": "success"})
```

## 翻译文件

编写babel.cfg

```
[python:**.py]
```

生成pot文件

```shell
# 进入babel.cfg所在路径，执行如下命令后，会在路径下生成一个messages.pot文件
pybabel extract -F babel.cfg -o messages.pot .

# 使用了lazy_gettext()
pybabel extract -F babel.cfg -k lazy_gettext -o messages.pot .
```

提供翻译文件

```shell
# 需要根据不同的语言环境来加载相应的翻译文件。翻译文件需要以.po或.mo格式存在。可以使用命令pybabel来生成翻译文件。
pybabel init -i messages.pot -d translations -l en
# 生成一个中文翻译英文的翻译文件。-i参数指定了一个模板文件，-d参数指定了翻译文件的存放目录，-l参数指定了语言代码。
# 若是目标目录不是translations，则须要设置app.config["BABEL_TRANSLATION_DIRECTORIES"]字段
```

编辑翻译文件

```shell
# 在messages.pot中会有一个msgid和msgstr，msgid是前面咱们在代码中_("Hello")中的字符，msgstr是翻译后的字符，而后在生成mo文件便可（若是使用的不是gettext方法而是ngettext方法，则会有多个msgid对应一个msgstr）
```

翻译文件的更新

```shell
# 当我们有新的翻译需要更新时，可以执行以下命令：
pybabel update -i messages.pot -d translations 
# 根据当前的翻译文件生成一个新的翻译文件。
```

编译翻译文件

```shell
# 为了让Flask-Babel能够加载翻译文件，我们需要将其编译为.mo格式。可以执行以下命令来编译翻译文件：
pybabel compile -d translations 
# 编译后的文件会被保存在翻译文件的相应目录中。
```

