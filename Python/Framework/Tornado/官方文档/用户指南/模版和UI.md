# 模版和UI

Tornado包含一种简单，快速且灵活的模板语言。本节介绍了该语言以及相关的问题，例如国际化。

Tornado也可以与任何其他Python模板语言一起使用，尽管没有规定将这些系统集成到`RequestHandler.render`中。只需将模板呈现为字符串并将其传递给`RequestHandler.write`

## 模版配置

默认情况下，Tornado在与引用模板文件的.py文件相同的目录中查找模板文件。要将模板文件放在不同的目录中，请使用`template_path`的`Application setting`（如果您为不同的处理程序使用不同的模板路径，请覆盖`RequestHandler.get_template_path`）。

要从非文件系统位置加载模板，请子类`tornado.template.BaseLoader`并将实例作为`template_loader`应用程序设置传递。

默认情况下，已编译的模板被缓存；要关闭此缓存并重新加载模板，以使对基础文件的更改始终可见，请使用应用程序设置`compiled_template_cache = False`或`debug = True`。

## 模版语法

Tornado模板只是HTML（或其他任何基于文本的格式），其中Python控制序列和表达式嵌入标记中：

```html
<html>
   <head>
      <title>{{ title }}</title>
   </head>
   <body>
     <ul>
       {% for item in items %}
         <li>{{ escape(item) }}</li>
       {% end %}
     </ul>
   </body>
 </html>
```

如果将该模板另存为`template.html`，并将其放在与Python文件相同的目录中，则可以使用以下方式呈现此模板：

```python
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        items = ["Item 1", "Item 2", "Item 3"]
        self.render("template.html", title="My title", items=items)
```

Tornado模板支持控制语句和表达式。控制语句用`{％`和`％}`包围，例如`{% if len(items) > 2％}`。表达式用`{{`和`}}`包围，例如`{{items [0]}}`。

控制语句或多或少准确地映射到Python语句。我们支持`if,for,while,try`，在以上情况下都尝试以`{％end％}`结尾。我们还支持使用`extend`和`block`语句来支持模板继承，这些声明在`tornado.template`的文档中进行了详细说明。

表达式可以是任何Python表达式，包括函数调用。模板代码在包含以下对象和功能的名称空间中执行。（请注意，此列表适用于使用RequestHandler.render和render_string呈现的模板。如果直接在RequestHandler外部使用tornado.template模块，则其中许多条目将不存在）。

```
- escape:tornado.escape.xhtml_escape的别名
- xhtml_escape：tornado.escape.xhtml_escape的别名
- url_escape：tornado.escape.url_escape的别名
- json_encode：tornado.escape.json_encode的别名
- squeeze：tornado.escape.squeeze的别名
- linkify：tornado.escape.linkify的别名
- datetime：Python日期时间模块
- handler：当前的RequestHandler对象
- request：handler.request的别名
- current_user：handler.current_user的别名
- locale：handler.locale的别名
- _：handler.locale.translate的别名
- static_url：handler.static_url的别名
- xsrf_form_html：handler.xsrf_form_html的别名
- reverse_url：Application.reverse_url的别名
- ui_methods和ui_modules应用程序设置中的所有条目
- 传递给render或render_string的任何关键字参数
```

在构建真实的应用程序时，您将要使用Tornado模板的所有功能，尤其是模板继承。在`tornado.template`部分中阅读所有有关这些功能的信息（某些功能，包括`UIModule`，在`tornado.we`b模块中实现）

在后台，Tornado模板直接转换为Python。将模板中包含的表达式逐字复制到代表模板的Python函数中。我们不会尝试阻止模板语言中的任何内容；我们明确创建它是为了提供其他更严格的模板系统无法提供的灵活性。因此，如果您在模板表达式中编写随机内容，则在执行模板时会出现随机Python错误。

默认情况下，使用`tornado.escape.xhtml_escape`函数对所有模板输出进行转义。可以通过将`autoescape = None`传递给`Application`或`tornado.template.Loader`构造函数，使用`{％autoescape None％}`指令的模板文件或通过`{％raw ...％}`替换`{{...}}`的单个表达式来全局更改此行为。此外，在每个这些地方，可以使用替代的转义函数的名称代替`None`。

请注意，虽然Tornado的自动转义有助于避免XSS漏洞，但在所有情况下是不够的。在某些位置（例如在Javascript或CSS中）出现的表达式可能需要额外的转义。此外，必须注意始终在可能包含不受信任内容的HTML属性中始终使用双引号和`xhtml_escape`，或者必须对属性使用单独的转义函数（例如[this blog post](http://wonko.com/post/html-escaping)）。

## 国际化

当前用户的语言环境（无论是否已登录）始终可以在请求处理程序中以`self.locale`的形式在模板中使用。语言环境的名称（例如`en_US`）可以作为`locale.name`使用，您可以使用`Locale.translate`方法转换字符串。模板还具有可用于字符串翻译的全局函数调用`_()`。翻译功能有两种形式：

```
_("Translate this string")
```

直接根据当前语言环境转换字符串，并且：

```
_("A person liked this", "%(num)d people liked this",
  len(people)) % {"num": len(people)}
```

根据第三个参数的值，该字符串可以转换为单数或复数的字符串。在上面的示例中，如果`len(people)`为1，则将返回第一个字符串的翻译，否则将返回第二个字符串的翻译。

最常见的翻译模式是使用Python命名的占位符作为变量（在上面的示例中为`％(num)d`），因为占位符可以在翻译时四处移动。

这是一个正确的国际化模板：

```python
<html>
   <head>
      <title>FriendFeed - {{ _("Sign in") }}</title>
   </head>
   <body>
     <form action="{{ request.path }}" method="post">
       <div>{{ _("Username") }} <input type="text" name="username"/></div>
       <div>{{ _("Password") }} <input type="password" name="password"/></div>
       <div><input type="submit" value="{{ _("Sign in") }}"/></div>
       {% module xsrf_form_html() %}
     </form>
   </body>
 </html>
```

默认情况下，我们使用用户浏览器发送的`Accept-Language`标头来检测用户的语言环境。如果找不到合适的`Accept-Language`值，则选择`en_US`。如果您让用户将其语言环境设置为首选项，则可以通过覆盖`RequestHandler.get_user_locale`来覆盖此默认语言环境选择：

```python
class BaseHandler(tornado.web.RequestHandler):
    def get_current_user(self):
        user_id = self.get_secure_cookie("user")
        if not user_id: return None
        return self.backend.get_user_by_id(user_id)

    def get_user_locale(self):
        if "locale" not in self.current_user.prefs:
            # Use the Accept-Language header
            return None
        return self.current_user.prefs["locale"]
```

如果`get_user_locale`返回`None`，我们将退回到`Accept-Language`标头。

`tornado.locale`模块支持以两种格式加载转换：`gettext`和相关工具使用的`.mo`格式，以及简单的`.csv`格式。通常，应用程序在启动时会调用一次`tornado.locale.load_translations`或`tornado.locale.load_gettext_translations`。有关支持的格式的更多详细信息，请参见这些方法。

您可以使用`tornado.locale.get_supported_locales()`获取应用程序中支持的语言环境的列表。根据支持的语言环境，将用户的语言环境选择为最接近的匹配项。例如，如果用户的语言环境是`es_GT`，并且支持`es`语言环境，则`self.locale`将是该请求的`es`。如果找不到最接近的匹配项，我们将退回到`en_US`。

## UI模型

Tornado支持UI模块，可以轻松地在整个应用程序中支持标准的，可重复使用的UI小部件。UI模块就像用于呈现页面组件的特殊函数调用一样，它们可以与自己的CSS和JavaScript打包在一起。

例如，如果您正在实现一个博客，并且希望博客条目同时出现在博客首页和每个博客条目页面上，则可以创建一个`Entry`模块在两个页面上进行渲染。首先，为您的UI模块创建一个Python模块，例如`uimodules.py`：

```python
class Entry(tornado.web.UIModule):
    def render(self, entry, show_comments=False):
        return self.render_string(
            "module-entry.html", entry=entry, show_comments=show_comments)
```

告诉Tornado在应用程序中使用`ui_modules`设置使用`uimodules.py`：

```python
from . import uimodules

class HomeHandler(tornado.web.RequestHandler):
    def get(self):
        entries = self.db.query("SELECT * FROM entries ORDER BY date DESC")
        self.render("home.html", entries=entries)

class EntryHandler(tornado.web.RequestHandler):
    def get(self, entry_id):
        entry = self.db.get("SELECT * FROM entries WHERE id = %s", entry_id)
        if not entry: raise tornado.web.HTTPError(404)
        self.render("entry.html", entry=entry)

settings = {
    "ui_modules": uimodules,
}
application = tornado.web.Application([
    (r"/", HomeHandler),
    (r"/entry/([0-9]+)", EntryHandler),
], **settings)
```

在模板中，您可以使用`{％module％}`语句来调用模块。

home.html

```html
{% for entry in entries %}
  {% module Entry(entry) %}
{% end %}
```

entry.html

```
{% module Entry(entry, show_comments=True) %}
```

通过覆盖`Embedded_css,embedded_javascript,javascript_files,css_files`方法，模块可以包含自定义CSS和JavaScript函数：

```python
class Entry(tornado.web.UIModule):
    def embedded_css(self):
        return ".entry { margin-bottom: 1em; }"

    def render(self, entry, show_comments=False):
        return self.render_string(
            "module-entry.html", show_comments=show_comments)
```

无论页面上使用模块多少次，都将包含一次模块CSS和JavaScript。CSS总是包含在页面的`<head>`中，而JavaScript总是包含在页面末尾`</ body>`标记之前。

当不需要其他Python代码时，模板文件本身可以用作模块。例如，可以重写前面的示例，以将以下内容放入`module-entry.html`中：

```html
{{ set_resources(embedded_css=".entry { margin-bottom: 1em; }") }}
<!-- more template html... -->
```

修改后的模板模块将通过以下方式调用：

```
{% module Template("module-entry.html", show_comments=True) %}
```

`set_resources`函数仅在通过`{％module Template(...)％}`调用的模板中可用。与`{％include ...％}`指令不同，模板模块与其包含的模板具有不同的名称空间-它们只能看到全局模板名称空间及其自己的关键字参数。

