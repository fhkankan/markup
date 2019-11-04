# 表单API 

> 关于这篇文档
这篇文档讲述Django 表单API 的详细细节。 你应该先阅读[introduction to working with forms](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/index.html)。

## 绑定和未绑定表单

[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)要么是**绑定的**，要么是**未绑定的**。

- 如果是**绑定的**，那么它能够验证数据，并渲染表单及其数据成HTML。
- 如果**未绑定**，则无法进行验证（因为没有数据可以验证！），但它仍然可以以HTML形式呈现空白表单。

```
class Form
```

若要创建一个未绑定的[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例，只需简单地实例化该类：

```shell
>>> f = ContactForm()
```

若要绑定数据到表单，可以将数据以字典的形式传递给[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)类的构造函数的第一个参数：

```shell
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)
```

在这个字典中，键为字段的名称，它们对应于[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)类中的属性。 值为需要验证的数据。 这些通常是字符串，但不要求它们是字符串；您传递的数据类型取决于[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field)，我们稍后将看到。

- `Form.is_bound`

如果运行时刻你需要区分绑定的表单和未绑定的表单，可以检查下表单[`is_bound`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.is_bound) 属性的值：

```shell
>>> f = ContactForm()
>>> f.is_bound
False
>>> f = ContactForm({'subject': 'hello'})
>>> f.is_bound
True
```

注意，传递一个空的字典将创建一个带有空数据的*绑定的*表单：

```shell
>>> f = ContactForm({})
>>> f.is_bound
True
```

如果你有一个绑定的[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例但是想改下数据，或者你想绑定一个未绑定的[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)表单到某些数据，你需要创建另外一个[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例。 [`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form) 实例的数据没有办法修改。 [`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例一旦创建，你应该将它的数据视为不可变的，无论它有没有数据。

## 使用表单验证数据

`Form.clean()`

当你需要为相互依赖的字段添加自定义的验证时，你可以重写`Form`的`clean()`方法。 示例用法参见[Cleaning and validating fields that depend on each other](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/validation.html#validating-fields-with-clean)。

`Form.is_valid()`

[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)对象的首要任务就是验证数据。 对于绑定的[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例，可以调用[`is_valid()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.is_valid)方法来执行验证，该方法会返回一个表示数据是否合法的布尔值。

```shell
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> f.is_valid()
True
```

让我们试下非法的数据。 下面的情形中，`subject` 为空（默认所有字段都是必需的）且`sender` 是一个不合法的邮件地址：

```shell
>>> data = {'subject': '',
...         'message': 'Hi there',
...         'sender': 'invalid email address',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> f.is_valid()
False
```

`Form.errors`

访问[`errors`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.errors) 属性可以获得错误信息的一个字典：

```shell
>>> f.errors
{'sender': ['Enter a valid email address.'], 'subject': ['This field is required.']}
```

在这个字典中，键为字段的名称，值为表示错误信息的Unicode 字符串组成的列表。 错误信息保存在列表中是因为字段可能有多个错误信息。

你可以在调用[`is_valid()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.is_valid) 之前访问[`errors`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.errors)。 表单的数据将在第一次调用[`is_valid()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.is_valid) 或者访问[`errors`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.errors) 时验证。

验证只会调用一次，无论你访问[`errors`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.errors) 或者调用[`is_valid()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.is_valid) 多少次。 这意味着，如果验证过程有副作用，这些副作用将只触发一次。

`Form.rors.as_data()`

返回一个`dict`，它映射字段到原始的`ValidationError` 实例。

```shell
>>> f.errors.as_data()
{'sender': [ValidationError(['Enter a valid email address.'])],
'subject': [ValidationError(['This field is required.'])]}
```

每当你需要根据错误的`code` 来识别错误时，可以调用这个方法。 它可以用来重写错误信息或者根据特定的错误编写自定义的逻辑。 它还可以用于以自定义格式（例如XML）序列化错误；例如，[`as_json()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.errors.as_json)依赖于`as_data()`。

需要`as_data()` 方法是为了向后兼容。 以前，`Validation` 实例在它们**渲染后** 的错误消息一旦添加到`ErrorForm.errors` 字典就立即被丢弃。 理想情况下，`as_` 应该已经保存`ValidationError` 实例而带有`Form.errors` 前缀的方法可以渲染它们，但是为了不破坏直接使用`Form.errors` 中的错误消息的代码，必须使用其它方法来实现。

`Form.rors.as_json(escape_html=False)`

返回JSON 序列化后的错误。

```shell
>>> f.errors.as_json()
{"sender": [{"message": "Enter a valid email address.", "code": "invalid"}],
"subject": [{"message": "This field is required.", "code": "required"}]}
```

默认情况下，`as_json()` 不会转义它的输出。 如果你正在使用AJAX 请求表单视图，而客户端会解析响应并将错误插入到页面中，你必须在客户端对结果进行转义以避免可能的跨站脚本攻击。 使用一个JavaScript库，比如jQuery来做这件事很简单 —— 只要使用`$(el).text(errorText)` 而不是`.html()` 就可以。

如果由于某种原因你不想使用客户端的转义，你还可以设置`escape_html=True`，这样错误消息将被转义而你可以直接在HTML 中使用它们。

`Form.errors.get_json_data(escape_html=False)`

Django 2.0的新功能。

以适合于序列化为JSON的字典的形式返回错误。`Form.errors.as_json()`返回序列化的JSON，而这将在序列化之前返回错误数据。`escape_html`参数的行为如`Form.errors.as_json()`中所述。

`Form.add_error(field, error)`

该方法允许从`Form.clean()`方法中或从表单外部向特定字段添加错误；例如从一个角度。

`field` 参数为字段的名称。 如果值为`None`，error 将作为[`Form.non_field_errors()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.non_field_errors) 返回的一个非字段错误。

`error` 参数可以是一个简单的字符串，或者最好是一个`ValidationError` 实例。 [Raising ValidationError](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/validation.html#raising-validation-error) 中可以看到定义表单错误时的最佳实践。

注意，`Form.add_error()` 会自动删除`cleaned_data` 中的相关字段。

`Form.has_error(field, code=None)`

这个方法返回一个布尔值，指示一个字段是否具有指定错误`code` 的错误。 如果关键字参数code设为None，那么方法将在该字段有任何错误时都将返回True。``````

若要检查非字段错误，使用[`NON_FIELD_ERRORS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.NON_FIELD_ERRORS) 作为`field` 参数。

`Form.non_field_errors()`

这个方法返回[`Form.errors`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.errors) 中不是与特定字段相关联的错误。 它包含在[`Form.clean()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.clean) 中引发的`ValidationError` 和使用[`Form.add_error(None, "...")`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.add_error) 添加的错误。

### 未绑定表单行为

验证没有绑定数据的表单是没有意义的，下面的例子展示了这种情况：

```shell
>>> f = ContactForm()
>>> f.is_valid()
False
>>> f.errors
{}
```

## 动态初始值

- `Form.initial`

在运行时，可使用[`initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.initial)声明表单字段的初始值。 例如，你可能希望使用当前会话的用户名填充`username`字段。

使用[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)的[`initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.initial)参数可以实现。 该参数是一个字典。 只包括您指定初始值的字段；没有必要在表单中包含每个字段。 像这样：

```shell
>>> f = ContactForm(initial={'subject': 'Hi there!'})
```

这些值只显示在没有绑定的表单中，即使没有提供特定值它们也不会作为后备的值。

如果一个[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field)包含[`initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.initial)参数，*并且*你在实例化`Form`时又包含了一个[`initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.initial)参数，那么后一个`initial`优先级高。 在下面的例子中，`initial` 在字段和表单实例化中都有定义，此时后者具有优先权：

```shell
>>> from django import forms
>>> class CommentForm(forms.Form):
...     name = forms.CharField(initial='class')
...     url = forms.URLField()
...     comment = forms.CharField()
>>> f = CommentForm(initial={'name': 'instance'}, auto_id=False)
>>> print(f)
<tr><th>Name:</th><td><input type="text" name="name" value="instance" required /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url" required /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" required /></td></tr>
```

- `Form.get_initial_for_field（field，field_name)`

**Django中的新功能1.11。**

使用[`get_initial_for_field()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.get_initial_for_field)来检索表单字段的初始数据。 它以该顺序从[`Form.initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.initial)和[`Field.initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.initial)中检索数据，并评估任何可调用的初始值。

## 检查哪个表单数据已经改变了

- `Form.has_changed()`

当你需要检查表单的数据是否从初始数据发生改变时，可以使用Form的has_changed()方法。````

```
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data, initial=data)
>>> f.has_changed()
False
```

当提交表单时，我们可以重新构建表单并提供初始值，这样可以实现比较：

```
>>> f = ContactForm(request.POST, initial=data)
>>> f.has_changed()
```

如果`request.POST` 中的数据与[`initial`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.initial) 中的不同，`has_changed()` 将为`True`，否则为`False`。 计算的结果是通过调用表单每个字段的[`Field.has_changed()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.has_changed) 得到的。

- `Form.changed_data`

changed_data属性返回一个列表，包含那些在表单的绑定数据中的值（通常为request.POST）与原始值发生改变的字段的名字。 如果没有数据不同，它返回一个空列表。

```shell
>>> f = ContactForm(request.POST, initial=data)
>>> if f.has_changed():
...     print("The following fields changed: %s" % ", ".join(f.changed_data))
```

## 访问表单中的字段

`Form.fields`

你可以从[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例的`fields`属性访问字段：

```shell
>>> for row in f.fields.values(): print(row)
...
<django.forms.fields.CharField object at 0x7ffaac632510>
<django.forms.fields.URLField object at 0x7ffaac632f90>
<django.forms.fields.CharField object at 0x7ffaac3aa050>
>>> f.fields['name']
<django.forms.fields.CharField object at 0x7ffaac6324d0>
```

你可以修改[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例的字段来改变字段在表单中的表示：

```shell
>>> f.as_table().split('\n')[0]
'<tr><th>Name:</th><td><input name="name" type="text" value="instance" required /></td></tr>'
>>> f.fields['name'].label = "Username"
>>> f.as_table().split('\n')[0]
'<tr><th>Username:</th><td><input name="name" type="text" value="instance" required /></td></tr>'
```

注意不要改变`base_fields` 属性，因为一旦修改将影响同一个Python 进程中接下来所有的`ContactForm` 实例：

```shell
>>> f.base_fields['name'].label = "Username"
>>> another_f = CommentForm(auto_id=False)
>>> another_f.as_table().split('\n')[0]
'<tr><th>Username:</th><td><input name="name" type="text" value="class" required /></td></tr>'
```

## 访问“干净”数据

`Form.cleaned_data`

[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)类中的每个字段不仅负责验证数据，还负责“清洁”它们 —— 将它们转换为正确的格式。 这是个非常好用的功能，因为它允许字段以多种方式输入数据，并总能得到一致的输出。

例如，[`DateField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.DateField) 将输入转换为Python 的 `datetime.date` 对象。 无论你传递的是`DateField` 格式的字符串、`datetime.date` 对象、还是其它格式的数字，`'1994-07-15'` 将始终将它们转换成`datetime.date` 对象，只要它们是合法的。

一旦你创建一个[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例并通过验证后，你就可以通过它的`cleaned_data` 属性访问清洁的数据：

```shell
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> f.is_valid()
True
>>> f.cleaned_data
{'cc_myself': True, 'message': 'Hi there', 'sender': 'foo@example.com', 'subject': 'hello'}
```

注意，文本字段 —— 例如，`CharField` 和`EmailField` —— 始终将输入转换为字符串。 我们将在这篇文档的后面描述编码的影响。

如果你的数据*没有* 通过验证，`cleaned_data` 字典中只包含合法的字段：

```shell
>>> data = {'subject': '',
...         'message': 'Hi there',
...         'sender': 'invalid email address',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> f.is_valid()
False
>>> f.cleaned_data
{'cc_myself': True, 'message': 'Hi there'}
```

`cleaned_data` 始终*只* 包含`Form`中定义的字段，即使你在构建`Form` 时传递了额外的数据。 在下面的例子中，我们传递一组额外的字段给`ContactForm` 构造函数，但是`cleaned_data` 将只包含表单的字段：

```shell
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True,
...         'extra_field_1': 'foo',
...         'extra_field_2': 'bar',
...         'extra_field_3': 'baz'}
>>> f = ContactForm(data)
>>> f.is_valid()
True
>>> f.cleaned_data # Doesn't contain extra_field_1, etc.
{'cc_myself': True, 'message': 'Hi there', 'sender': 'foo@example.com', 'subject': 'hello'}
```

当`Form`合法时，`cleaned_data` 将包含*所有*字段的键和值，即使传递的数据不包含某些可选字段的值。 在下面的例子中，传递的数据字典不包含`nick_name` 字段的值，但是`cleaned_data` 任然包含它，只是值为空：

```shell
>>> from django import forms
>>> class OptionalPersonForm(forms.Form):
...     first_name = forms.CharField()
...     last_name = forms.CharField()
...     nick_name = forms.CharField(required=False)
>>> data = {'first_name': 'John', 'last_name': 'Lennon'}
>>> f = OptionalPersonForm(data)
>>> f.is_valid()
True
>>> f.cleaned_data
{'nick_name': '', 'first_name': 'John', 'last_name': 'Lennon'}
```

在上面的例子中，`nick_name` 中`nick_name` 设置为一个空字符串，这是因为`cleaned_data` 是`CharField`而 `CharField` 将空值作为一个空字符串。 每个字段都知道自己的“空”值 —— 例如，`DateField` 的空值是`None` 而不是一个空字符串。 关于每个字段空值的完整细节，参见“内建的`Field` 类”一节中每个字段的“空值”提示。

你可以自己编写代码来对特定的字段（根据它们的名字）或者表单整体（考虑到不同字段的组合）进行验证。 更多信息参见[Form and field validation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/validation.html)。

## 输出表单为HTML 

`Form`对象的第二个任务是将它渲染成HTML。 很简单，`print` 它：

```shell
>>> f = ContactForm()
>>> print(f)
<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" required /></td></tr>
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" required /></td></tr>
<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" required /></td></tr>
<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" /></td></tr>
```

如果表单是绑定的，输出的HTML 将包含数据。 例如，如果字段是`<input type="text">` 的形式，其数据将位于`value` 属性中。 如果一个字段由`＆lt； input type =“checkbox”＆gt；`表示，则该HTML将包括`checked`

```shell
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> f = ContactForm(data)
>>> print(f)
<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" value="hello" required /></td></tr>
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" value="Hi there" required /></td></tr>
<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" value="foo@example.com" required /></td></tr>
<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" checked /></td></tr>
```

**在Django更改1.11：**

`checked`属性已更改为使用HTML5布尔语法，而不是`checked="checked"`。

默认的输出时具有两个列的HTML 表格，每个字段对应一个`<tr>`。 注意事项：

- 为了灵活性，输出*不*包含`</table>` 和`<table>`、`</form>` 和`<form>` 以及`<input type="submit">` 标签。 你需要添加它们。
- 每个字段类型都有一个默认的HTML表示。 `CharField` is represented by an `<input type="text">` and `EmailField` by an `<input type="email">`. `BooleanField` 表示为一个`<input type="checkbox">`。 注意，这些只是默认的表示；你可以使用Widget 指定字段使用哪种HTML，我们将稍后解释。
- 每个标签的HTML `name` 直接从`ContactForm` 类中获取。
- 每个字段的文本标签 —— 例如`'Message:'`、`'Subject:'` 和`'Cc myself:'` 通过将所有的下划线转换成空格并大写第一个字母生成。 再次注意，这些只是明智的违约；您也可以手动指定标签。
- 每个文本标签周围有一个HTML `<label>` 标签，它指向表单字段的`id`。 这个`id`，是通过在字段名称前面加上`'id_'` 前缀生成。 `id` 属性和`<label>` 标签默认包含在输出中，但你可以改变这一行为。
- 输出使用HTML5语法，目标是`＆lt；！DOCTYPE html＆gt；`。 例如，它使用布尔属性，如`checked`，而不是`checked='checked'`的XHTML样式。

虽然`<table>` 表单时`print` 是默认的输出格式，但是还有其它格式可用。 每个格式对应于表单对象的一个方法，每个方法都返回一个Unicode 对象。

### `as_p()`

`Form.as_p()`

`as_p()` 渲染表单为一系列的`<p>`标签，每个`<p>`标签包含一个字段：

```shell
>>> f = ContactForm()
>>> f.as_p()
'<p><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" required /></p>\n<p><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" required /></p>\n<p><label for="id_sender">Sender:</label> <input type="text" name="sender" id="id_sender" required /></p>\n<p><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></p>'
>>> print(f.as_p())
<p><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" required /></p>
<p><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" required /></p>
<p><label for="id_sender">Sender:</label> <input type="email" name="sender" id="id_sender" required /></p>
<p><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></p>
```

### `as_ul()`

`Form.as_ul()`

`as_ul()` 渲染表单为一系列的`<li>`标签，每个`<li>` 标签包含一个字段。 它*不*包含`</ul>` 和`<ul>`，所以你可以自己指定`<ul>` 的任何HTML 属性：

```shell
>>> f = ContactForm()
>>> f.as_ul()
'<li><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" required /></li>\n<li><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" required /></li>\n<li><label for="id_sender">Sender:</label> <input type="email" name="sender" id="id_sender" required /></li>\n<li><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></li>'
>>> print(f.as_ul())
<li><label for="id_subject">Subject:</label> <input id="id_subject" type="text" name="subject" maxlength="100" required /></li>
<li><label for="id_message">Message:</label> <input type="text" name="message" id="id_message" required /></li>
<li><label for="id_sender">Sender:</label> <input type="email" name="sender" id="id_sender" required /></li>
<li><label for="id_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_cc_myself" /></li>
```

### `as_table()`

`Form.as_table()`

最后，`as_table()`输出表单为一个HTML `<table>`。 它与`print` 完全相同。 事实上，当你`print` 一个表单对象时，在后台调用的就是`as_table()` 方法：

```shell
>>> f = ContactForm()
>>> f.as_table()
'<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" required /></td></tr>\n<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" required /></td></tr>\n<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" required /></td></tr>\n<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" /></td></tr>'
>>> print(f)
<tr><th><label for="id_subject">Subject:</label></th><td><input id="id_subject" type="text" name="subject" maxlength="100" required /></td></tr>
<tr><th><label for="id_message">Message:</label></th><td><input type="text" name="message" id="id_message" required /></td></tr>
<tr><th><label for="id_sender">Sender:</label></th><td><input type="email" name="sender" id="id_sender" required /></td></tr>
<tr><th><label for="id_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_cc_myself" /></td></tr>
```

### 样式要求或错误的表格行

`Form.error_css_class`

`Form.required_css_class`

将必填的表单行和有错误的表单行定义不同的样式特别常见。 例如，你想将必填的表单行以粗体显示、将错误以红色显示。

[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)类具有一对钩子，可以使用它们来添加`class` 属性给必填的行或有错误的行：只需简单地设置[`Form.error_css_class`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.error_css_class) 和/或 [`Form.required_css_class`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.required_css_class) 属性：

```python
from django import forms

class ContactForm(forms.Form):
    error_css_class = 'error'
    required_css_class = 'required'

    # ... and the rest of your fields here
```

一旦你设置好，将根据需要设置行的`"error"` 和/或`"required"` CSS 类型。 其HTML 看上去将类似：

```shell
>>> f = ContactForm(data)
>>> print(f.as_table())
<tr class="required"><th><label class="required" for="id_subject">Subject:</label>    ...
<tr class="required"><th><label class="required" for="id_message">Message:</label>    ...
<tr class="required error"><th><label class="required" for="id_sender">Sender:</label>      ...
<tr><th><label for="id_cc_myself">Cc myself:<label> ...
>>> f['subject'].label_tag()
<label class="required" for="id_subject">Subject:</label>
>>> f['subject'].label_tag(attrs={'class': 'foo'})
<label for="id_subject" class="foo required">Subject:</label>
```

### 配置表单元素的HTML `id`属性和`<label>`标签

- `Form.auto_id` 

默认情况下，表单的渲染方法包含：

- 表单元素的HTML `id` 属性
- 对应的`<label>` 标签。 HTML `<label>` 标签指示标签文本关联的表单元素。 这个小小的改进让表单在辅助设备上具有更高的可用性。 使用`<label>` 标签始终是个好想法。

`id` 属性值通过在表单字段名称的前面加上`id_` 生成。 但是如果你想改变`id` 的生成方式或者完全删除 HTML `id` 属性和`<label>`标签，这个行为是可配置的。

`id` 和label 的行为使用`Form`构造函数的`auto_id` 参数控制。 这个参数必须为`True`、`False` 或者一个字符串。

如果`auto_id` 为`False`，那么表单的输出将不包含`<label>` 标签和`id` 属性：

```shell
>>> f = ContactForm(auto_id=False)
>>> print(f.as_table())
<tr><th>Subject:</th><td><input type="text" name="subject" maxlength="100" required /></td></tr>
<tr><th>Message:</th><td><input type="text" name="message" required /></td></tr>
<tr><th>Sender:</th><td><input type="email" name="sender" required /></td></tr>
<tr><th>Cc myself:</th><td><input type="checkbox" name="cc_myself" /></td></tr>
>>> print(f.as_ul())
<li>Subject: <input type="text" name="subject" maxlength="100" required /></li>
<li>Message: <input type="text" name="message" required /></li>
<li>Sender: <input type="email" name="sender" required /></li>
<li>Cc myself: <input type="checkbox" name="cc_myself" /></li>
>>> print(f.as_p())
<p>Subject: <input type="text" name="subject" maxlength="100" required /></p>
<p>Message: <input type="text" name="message" required /></p>
<p>Sender: <input type="email" name="sender" required /></p>
<p>Cc myself: <input type="checkbox" name="cc_myself" /></p>
```

如果`auto_id` 设置为`True`，那么输出的表示*将* 包含`<label>` 标签并简单地使用字典名称作为每个表单字段的`id`：

```shell
>>> f = ContactForm(auto_id=True)
>>> print(f.as_table())
<tr><th><label for="subject">Subject:</label></th><td><input id="subject" type="text" name="subject" maxlength="100" required /></td></tr>
<tr><th><label for="message">Message:</label></th><td><input type="text" name="message" id="message" required /></td></tr>
<tr><th><label for="sender">Sender:</label></th><td><input type="email" name="sender" id="sender" required /></td></tr>
<tr><th><label for="cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="cc_myself" /></td></tr>
>>> print(f.as_ul())
<li><label for="subject">Subject:</label> <input id="subject" type="text" name="subject" maxlength="100" required /></li>
<li><label for="message">Message:</label> <input type="text" name="message" id="message" required /></li>
<li><label for="sender">Sender:</label> <input type="email" name="sender" id="sender" required /></li>
<li><label for="cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="cc_myself" /></li>
>>> print(f.as_p())
<p><label for="subject">Subject:</label> <input id="subject" type="text" name="subject" maxlength="100" required /></p>
<p><label for="message">Message:</label> <input type="text" name="message" id="message" required /></p>
<p><label for="sender">Sender:</label> <input type="email" name="sender" id="sender" required /></p>
<p><label for="cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="cc_myself" /></p>
```

如果`auto_id` 设置为包含格式字符`'%s'` 的字符串，那么表单的输出将包含`<label>` 标签，并将根据格式字符串生成`id` 属性。 例如，对于格式字符串`'field_%s'`，名为`subject` 的字段的`id` 值将是`'field_subject'`。 继续我们的例子：

```shell
>>> f = ContactForm(auto_id='id_for_%s')
>>> print(f.as_table())
<tr><th><label for="id_for_subject">Subject:</label></th><td><input id="id_for_subject" type="text" name="subject" maxlength="100" required /></td></tr>
<tr><th><label for="id_for_message">Message:</label></th><td><input type="text" name="message" id="id_for_message" required /></td></tr>
<tr><th><label for="id_for_sender">Sender:</label></th><td><input type="email" name="sender" id="id_for_sender" required /></td></tr>
<tr><th><label for="id_for_cc_myself">Cc myself:</label></th><td><input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></td></tr>
>>> print(f.as_ul())
<li><label for="id_for_subject">Subject:</label> <input id="id_for_subject" type="text" name="subject" maxlength="100" required /></li>
<li><label for="id_for_message">Message:</label> <input type="text" name="message" id="id_for_message" required /></li>
<li><label for="id_for_sender">Sender:</label> <input type="email" name="sender" id="id_for_sender" required /></li>
<li><label for="id_for_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></li>
>>> print(f.as_p())
<p><label for="id_for_subject">Subject:</label> <input id="id_for_subject" type="text" name="subject" maxlength="100" required /></p>
<p><label for="id_for_message">Message:</label> <input type="text" name="message" id="id_for_message" required /></p>
<p><label for="id_for_sender">Sender:</label> <input type="email" name="sender" id="id_for_sender" required /></p>
<p><label for="id_for_cc_myself">Cc myself:</label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></p>
```

如果`auto_id` 设置为任何其它的真值 —— 例如不包含`%s` 的字符串 —— 那么其行为将类似`auto_id` 等于`True`。

默认情况下，`auto_id` 设置为`'id_%s'`。

- `Form.label_suffix`

一个可翻译的字符串（默认为冒号`:`）的英文），当表单被呈现时，它将被附加在任何标签名称之后。

使用`label_suffix` 参数可以自定义这个字符，或者完全删除它：

```shell
>>> f = ContactForm(auto_id='id_for_%s', label_suffix='')
>>> print(f.as_ul())
<li><label for="id_for_subject">Subject</label> <input id="id_for_subject" type="text" name="subject" maxlength="100" required /></li>
<li><label for="id_for_message">Message</label> <input type="text" name="message" id="id_for_message" required /></li>
<li><label for="id_for_sender">Sender</label> <input type="email" name="sender" id="id_for_sender" required /></li>
<li><label for="id_for_cc_myself">Cc myself</label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></li>
>>> f = ContactForm(auto_id='id_for_%s', label_suffix=' ->')
>>> print(f.as_ul())
<li><label for="id_for_subject">Subject -></label> <input id="id_for_subject" type="text" name="subject" maxlength="100" required /></li>
<li><label for="id_for_message">Message -></label> <input type="text" name="message" id="id_for_message" required /></li>
<li><label for="id_for_sender">Sender -></label> <input type="email" name="sender" id="id_for_sender" required /></li>
<li><label for="id_for_cc_myself">Cc myself -></label> <input type="checkbox" name="cc_myself" id="id_for_cc_myself" /></li>
```

请注意，仅当标签的最后一个字符不是标点符号（英文版）时才添加标签后缀 `.`, `!`, `?` 要么 `:`）。

字段可以定义自己的[`label_suffix`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.label_suffix)。 而且将优先于[`Form.label_suffix`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.label_suffix)。 在运行时刻，后缀可以使用[`label_tag()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.label_tag) 的`label_suffix` 参数覆盖。

- `Form.use_required_attribute`

当设置为`True`（默认值）时，所需的表单字段将具有`required` HTML属性。

[Formsets](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/formsets.html)使用`use_required_attribute=False`实例化表单，以避免在从表单集添加和删除表单时浏览器验证不正确。

### 配置表单的小部件的呈现

`Form.default_renderer`

**Django中的新功能1.11。**

指定用于表单的[renderer](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/renderers.html)。 默认为`None`，这意味着使用由[`FORM_RENDERER`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-FORM_RENDERER)设置指定的默认渲染器。

在声明表单或使用`renderer`参数`Form.__init__()`时，可以将其设置为类属性。 像这样：

```python
from django import forms

class MyForm(forms.Form):
    default_renderer = MyRenderer()
```

要么：

```python
form = MyForm(renderer=MyRenderer())
```

### 字段顺序注意事项

在`as_table()`、`as_ul()` 和`as_p()` 中，字段以表单类中定义的顺序显示。 例如，在`message` 示例中，字段定义的顺序为`subject`, `ContactForm`, `sender`, `cc_myself`。 若要重新排序HTML 中的输出，只需改变字段在类中列出的顺序。

还有其他几种方式可以自定义顺序：

- `Form.field_order`

默认情况下，`Form.field_order=None`，它保留了您在窗体类中定义字段的顺序。 如果`field_order`是字段名称的列表，则字段按列表指定排序，其余字段将根据默认顺序进行追加。 列表中的未知字段名称将被忽略。 这使得可以通过将子集中的字段设置为`None`来禁用字段，而无需重新定义排序。

您也可以使用`Form.field_order`参数[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)来覆盖字段顺序。 如果[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)定义[`field_order`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.field_order) *和*，则在实例化`Form`时，包含`field_order`后一个`field_order`将具有优先权。

- `Form.order_fields(field_order)`

您可以随时使用`order_fields()`重新排列字段，并显示[`field_order`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.field_order)中的字段名称列表。

### 如何显示错误

如果你渲染一个绑定的`Form`对象，渲染时将自动运行表单的验证，HTML 输出将在出错字段的附近以`<ul class="errorlist">` 形式包含验证的错误。 错误信息的位置与你使用的输出方法有关：

```shell
>>> data = {'subject': '',
...         'message': 'Hi there',
...         'sender': 'invalid email address',
...         'cc_myself': True}
>>> f = ContactForm(data, auto_id=False)
>>> print(f.as_table())
<tr><th>Subject:</th><td><ul class="errorlist"><li>This field is required.</li></ul><input type="text" name="subject" maxlength="100" required /></td></tr>
<tr><th>Message:</th><td><input type="text" name="message" value="Hi there" required /></td></tr>
<tr><th>Sender:</th><td><ul class="errorlist"><li>Enter a valid email address.</li></ul><input type="email" name="sender" value="invalid email address" required /></td></tr>
<tr><th>Cc myself:</th><td><input checked type="checkbox" name="cc_myself" /></td></tr>
>>> print(f.as_ul())
<li><ul class="errorlist"><li>This field is required.</li></ul>Subject: <input type="text" name="subject" maxlength="100" required /></li>
<li>Message: <input type="text" name="message" value="Hi there" required /></li>
<li><ul class="errorlist"><li>Enter a valid email address.</li></ul>Sender: <input type="email" name="sender" value="invalid email address" required /></li>
<li>Cc myself: <input checked type="checkbox" name="cc_myself" /></li>
>>> print(f.as_p())
<p><ul class="errorlist"><li>This field is required.</li></ul></p>
<p>Subject: <input type="text" name="subject" maxlength="100" required /></p>
<p>Message: <input type="text" name="message" value="Hi there" required /></p>
<p><ul class="errorlist"><li>Enter a valid email address.</li></ul></p>
<p>Sender: <input type="email" name="sender" value="invalid email address" required /></p>
<p>Cc myself: <input checked type="checkbox" name="cc_myself" /></p>
```

### 自定义错误列表格式

默认情况下，表单使用`django.forms.utils.ErrorList` 来格式化验证时的错误。 如果你希望使用另外一种类来显示错误，可以在构造时传递（在Python 2 中将 `__str__` 替换为`__unicode__`）：

```shell
>>> from django.forms.utils import ErrorList
>>> class DivErrorList(ErrorList):
...     def __str__(self):              # __unicode__ on Python 2
...         return self.as_divs()
...     def as_divs(self):
...         if not self: return ''
...         return '<div class="errorlist">%s</div>' % ''.join(['<div class="error">%s</div>' % e for e in self])
>>> f = ContactForm(data, auto_id=False, error_class=DivErrorList)
>>> f.as_p()
<div class="errorlist"><div class="error">This field is required.</div></div>
<p>Subject: <input type="text" name="subject" maxlength="100" required /></p>
<p>Message: <input type="text" name="message" value="Hi there" required /></p>
<div class="errorlist"><div class="error">Enter a valid email address.</div></div>
<p>Sender: <input type="email" name="sender" value="invalid email address" required /></p>
<p>Cc myself: <input checked type="checkbox" name="cc_myself" /></p>
```

## 更细粒度的输出

`as_p()`，`as_ul()`和`as_table()`方法只是简单的快捷方式 - 它们不是唯一的表单显示的方式。

```
class BoundField
```

用于显示HTML 表单或者访问[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例的一个属性。其`__str__()`（Python 2 上为`__unicode__`）方法显示该字段的HTML。

以字段的名称为键，用字典查询语法查询表单，可以获取一个 `BoundField`：

```shell
>>> form = ContactForm()
>>> print(form['subject'])
<input id="id_subject" type="text" name="subject" maxlength="100" required />
```

迭代表单可以获取所有的`BoundField`：

```shell
>>> form = ContactForm()
>>> for boundfield in form: print(boundfield)
<input id="id_subject" type="text" name="subject" maxlength="100" required />
<input type="text" name="message" id="id_message" required />
<input type="email" name="sender" id="id_sender" required />
<input type="checkbox" name="cc_myself" id="id_cc_myself" />
```

字段的输出与表单的`auto_id` 设置有关：

```shell
>>> f = ContactForm(auto_id=False)
>>> print(f['message'])
<input type="text" name="message" required />
>>> f = ContactForm(auto_id='id_%s')
>>> print(f['message'])
<input type="text" name="message" id="id_message" required />
```

### `BoundField` 的属性

- `BoundField.auto_id`

  此`BoundField`的HTML ID属性。 如果[`Form.auto_id`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.auto_id)为`False`，则返回一个空字符串。

- `BoundField.data`

  该属性返回小部件的[`value_from_datadict()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.value_from_datadict)方法或`None`提取的[`BoundField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField)的数据，如果没有给出：
  
  ```shell
  >>> unbound_form = ContactForm() 
  >>>  print(unbound_form['subject'].data) None 
  >>>  bound_form = ContactForm(data={'subject': 'My Subject'}) 
  >>>  print(bound_form['subject'].data) My Subject 
  ```

- `BoundField.errors`

  作为HTML `<ul class =“errorlist”>`显示的[list-like object](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#ref-forms-error-list-format)印刷：
  
  ```shell
  >>> data = {'subject': 'hi', 'message': '', 'sender': '', 'cc_myself': ''}
  >>> f = ContactForm(data, auto_id=False)
  >>> print(f['message'])
  <input type="text" name="message" required />
  >>> f['message'].errors
  ['This field is required.']
  >>> print(f['message'].errors)
  <ul class="errorlist"><li>This field is required.</li></ul>
  >>> f['subject'].errors
  []
  >>> print(f['subject'].errors)

  >>> str(f['subject'].errors)
''
  ```

- `BoundField.field`

  来自此[`BoundField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField)的表单类的表单[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field)实例。

- `BoundField.form`

  [`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)实例此[`BoundField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField)被绑定到。

- `BoundField.help_text`

  字段的[`help_text`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.help_text)。

- `BoundField.html_name`

  将在小部件的HTML `name`属性中使用的名称。 它考虑到[`prefix`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.prefix)的形式。

- `BoundField.id_for_label`

  使用这个属性渲染字段的ID。 例如，如果你在模板中手工构造一个`<label>`（尽管 [`label_tag()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.label_tag) 将为你这么做）：
  ```
  <label for="{{ form.my_field.id_for_label }}">...</label>{{ my_field }} 
  ```
  默认情况下，它是在字段名称的前面加上`id_` （上面的例子中将是“`id_my_field`”）。 你可以通过设置字段Widget 的[`attrs`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.attrs) 来修改ID。 例如，像这样声明一个字段：
  ```
  my_field = forms.CharField(widget=forms.TextInput(attrs={'id': 'myFIELD'})) 
  ```
  使用上面的模板，将渲染成：
  ```
  <label for="myFIELD">...</label><input id="myFIELD" type="text" name="my_field" required /> 
  ```

- `BoundField.is_hidden`

  如果[`BoundField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField)的小部件被隐藏，返回`True`。

- `BoundField.label `

  字段的[`label`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.label)。 这在[`label_tag()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.label_tag)中使用。

- `BoundField.name`

  此字段的名称的形式如下：
  ```shell
  >> f = ContactForm()
  >>> print(f['subject'].name)
  subject
  >>> print(f['message'].name)
  message
  ```

### `BoundField` 的方法

- `BoundField.as_hidden(attrs = None，** kwargs)`

返回一个HTML字符串，将其表示为`<input type =“hidden”>`。`**kwargs`传递给[`as_widget()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.as_widget)。这种方法主要在内部使用。 你应该使用一个小部件。

- `BoundField.as_widget(widget=None, attrs=None, only_initial=False)`

通过渲染传递的小部件，添加传递给`attrs`的HTML属性来渲染该字段。 如果没有指定小部件，那么将使用该字段的默认小部件。`only_initial`由Django内部使用，不应显式设置。

- `BoundField.css_classes()`

当你使用Django 的快捷的渲染方法时，习惯使用CSS 类型来表示必填的表单字段和有错误的字段。 如果你是手工渲染一个表单，你可以使用`css_classes` 方法访问这些CSS 类型：

```shell
>>> f = ContactForm(data={'message': ''}) 
>>> f['message'].css_classes() '
'required' 
```

除了错误和必填的类型之外，如果你还想提供额外的类型，你可以用参数传递它们：

```shell
>>> f = ContactForm(data={'message': ''}) 
>>> f['message'].css_classes('foo bar') 
'foo bar required' `
```

- `BoundField.label_tag(contents=None, attrs=None, label_suffix=None)`

要单独呈现表单字段的标签标签，可以调用其`label_tag()`方法：

```shell
>>> f = ContactForm(data={'message': ''})
>>> print(f['message'].label_tag()) 
<label for="id_message">Message:</label> 
```

您可以提供将替换自动生成的标签标签的`contents`参数。 `attrs`字典可能包含`<label>`标记的附加属性。生成的HTML 包含表单的[`label_suffix`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.label_suffix)（默认为一个冒号），或者当前字段的[`label_suffix`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.label_suffix)。 可选的`label_suffix` 参数允许你覆盖之前设置的后缀。 例如，你可以使用一个空字符串来隐藏已选择字段的label。 如果在模板中需要这样做，你可以编写一个自定义的过滤器来允许传递参数给`label_tag`。

- `BoundField.value()`

这个方法用于渲染字段的原始值，与用`Widget` 渲染的值相同：

```shell
>>> initial = {'subject': 'welcome'}
>>> unbound_form = ContactForm(initial=initial)
>>> bound_form = ContactForm(data={'subject': 'hi'}, initial=initial)
>>> print(unbound_form['subject'].value())
welcome
>>> print(bound_form['subject'].value())
hi
```

## 定制`BoundField` 

如果您需要访问有关模板中的表单域的一些附加信息，并且使用[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field)的子类是不够的，还可以自定义[`BoundField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField)。

自定义表单字段可以覆盖`get_bound_field()`：

`Field.get_bound_field(form，field_name)`

  获取[`Form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form)的实例和字段的名称。 访问模板中的字段时将使用返回值。 很可能它将是[`BoundField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField)的子类的一个实例。

例如，如果您有一个`GPSCoordinatesField`，并希望能够访问有关模板中坐标的其他信息，则可以按如下方式实现：

```python
class GPSCoordinatesBoundField(BoundField):
    @property
    def country(self):
        """
        Return the country the coordinates lie in or None if it can't be
        determined.
        """
        value = self.value()
        if value:
            return get_country_from_coordinates(value)
        else:
            return None

class GPSCoordinatesField(Field):
    def get_bound_field(self, form, field_name):
        return GPSCoordinatesBoundField(form, self, field_name)
```

现在，您可以使用`{{ form.coordinates.country }}`访问模板中的国家/地区。

## 将上传的文件绑定到表单

处理带有`FileField` 和`ImageField` 字段的表单比普通的表单要稍微复杂一点。

首先，为了上传文件，你需要确保你的`"multipart/form-data"` 元素正确定义`enctype` 为`<form>`：

```html
<form enctype="multipart/form-data" method="post" action="/foo/">
```

其次，当你使用表单时，你需要绑定文件数据。 文件数据的处理与普通的表单数据是分开的，所以如果表单包含`FileField` 和`ImageField`，绑定表单时你需要指定第二个参数。 所以，如果我们扩展ContactForm 并包含一个名为`ImageField` 的`mugshot`，我们需要绑定包含mugshot 图片的文件数据：

```shell
# Bound form with an image field
>>> from django.core.files.uploadedfile import SimpleUploadedFile
>>> data = {'subject': 'hello',
...         'message': 'Hi there',
...         'sender': 'foo@example.com',
...         'cc_myself': True}
>>> file_data = {'mugshot': SimpleUploadedFile('face.jpg', <file data>)}
>>> f = ContactFormWithMugshot(data, file_data)
```

实际上，你一般将使用`request.FILES` 作为文件数据的源（和使用`request.POST` 作为表单数据的源一样）：

```shell
# Bound form with an image field, data from the request
>>> f = ContactFormWithMugshot(request.POST, request.FILES)
```

构造一个未绑定的表单和往常一样 —— 将表单数据*和*文件数据同时省略：

```shell
# Unbound form with an image field
>>> f = ContactFormWithMugshot()
```

### 测试多部分表单

`Form. is_multipart ()`

如果你正在编写可重用的视图或模板，你可能事先不知道你的表单是否是一个multipart 表单。 `is_multipart()` 方法告诉你表单提交时是否要求multipart：

```shell
>>> f = ContactFormWithMugshot()
>>> f.is_multipart()
True
```

下面是如何在模板中使用它的一个示例：

```html
{% if form.is_multipart %}
    <form enctype="multipart/form-data" method="post" action="/foo/">
{% else %}
    <form method="post" action="/foo/">
{% endif %}
{{ form }}
</form>
```

## 子类表单

如果你有多个`Form`类共享相同的字段，你可以使用子类化来减少冗余。

当你子类化一个自定义的`Form`类时，生成的子类将包含父类中的所有字段，以及在子类中定义的字段。

在下面的例子中，`priority` 包含`ContactForm` 中的所有字段，以及另外一个字段`ContactFormWithPriority`。 排在前面的是`ContactForm` 中的字段：

```shell
>>> class ContactFormWithPriority(ContactForm):
...     priority = forms.CharField()
>>> f = ContactFormWithPriority(auto_id=False)
>>> print(f.as_ul())
<li>Subject: <input type="text" name="subject" maxlength="100" required /></li>
<li>Message: <input type="text" name="message" required /></li>
<li>Sender: <input type="email" name="sender" required /></li>
<li>Cc myself: <input type="checkbox" name="cc_myself" /></li>
<li>Priority: <input type="text" name="priority" required /></li>
```

可以将多个表单进行子类化，将表单视为混合。 在下面的例子中，`InstrumentForm` 子类化`PersonForm` 和 `BeatleForm` ，所以它的字段列表包含两个父类的所有字段：

```shell
>>> from django import forms
>>> class PersonForm(forms.Form):
...     first_name = forms.CharField()
...     last_name = forms.CharField()
>>> class InstrumentForm(forms.Form):
...     instrument = forms.CharField()
>>> class BeatleForm(InstrumentForm, PersonForm):
...     haircut_type = forms.CharField()
>>> b = BeatleForm(auto_id=False)
>>> print(b.as_ul())
<li>First name: <input type="text" name="first_name" required /></li>
<li>Last name: <input type="text" name="last_name" required /></li>
<li>Instrument: <input type="text" name="instrument" required /></li>
<li>Haircut type: <input type="text" name="haircut_type" required /></li>
```

可以声明性地删除从父类继承的`Field`，方法是将该字段的名称设置为`None`子类。 像这样：

```shell
>>> from django import forms

>>> class ParentForm(forms.Form):
...     name = forms.CharField()
...     age = forms.IntegerField()

>>> class ChildForm(ParentForm):
...     name = None

>>> list(ChildForm().fields)
['age']
```

## 表单的前缀

`Form.prefix`

你可以将几个Django 表单放在一个`<form>` 标签中。 为了给每个`Form`一个自己的命名空间，可以使用`prefix` 关键字参数：

```shell
>>> mother = PersonForm(prefix="mother")
>>> father = PersonForm(prefix="father")
>>> print(mother.as_ul())
<li><label for="id_mother-first_name">First name:</label> <input type="text" name="mother-first_name" id="id_mother-first_name" required /></li>
<li><label for="id_mother-last_name">Last name:</label> <input type="text" name="mother-last_name" id="id_mother-last_name" required /></li>
>>> print(father.as_ul())
<li><label for="id_father-first_name">First name:</label> <input type="text" name="father-first_name" id="id_father-first_name" required /></li>
<li><label for="id_father-last_name">Last name:</label> <input type="text" name="father-last_name" id="id_father-last_name" required /></li>
```

前缀也可以在表单类中指定：

```
>>> class PersonForm(forms.Form):
...     ...
...     prefix = 'person'
```
