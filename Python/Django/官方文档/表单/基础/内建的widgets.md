# 窗口小部件

小部件是Django对HTML输入元素的表示。 窗口小部件处理HTML的呈现，以及从与窗口小部件对应的GET / POST字典中提取数据。

内置窗口小部件生成的HTML使用HTML5语法，目标是`<!DOCTYPE html>`。 例如，它使用布尔属性，例如`checked`，而不是`checked ='checked'`的XHTML样式。

> 小贴士
>
> 小部件不应与[表单字段混淆](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/fields.html)。 表单字段处理输入验证的逻辑，并直接在模板中使用。 窗口小部件处理在网页上呈现HTML表单输入元素以及提取原始提交的数据。 但是，窗口小部件需要[指定](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#widget-to-field)才能形成字段。

## 指定小部件

无论何时在表单上指定字段，Django都将使用适合于要显示的数据类型的默认窗口小部件。 要查找在哪个字段上使用哪个窗口小部件，请参阅有关[内置字段类](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/fields.html#built-in-fields)的文档。

但是，如果要为字段使用不同的窗口小部件，则可以在字段定义中使用[`窗口小部件`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/fields.html#django.forms.Field.widget)参数。 例如：

```python
from django import forms

class CommentForm(forms.Form):
    name = forms.CharField()
    url = forms.URLField()
    comment = forms.CharField(widget=forms.Textarea)
```

这将指定一个带有注释的表单，该注释使用较大的[`Textarea`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#django.forms.Textarea)小部件，而不是默认的[`TextInput`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#django.forms.TextInput)小部件。

## 设置小部件的参数

许多小部件具有可选的额外参数；在字段上定义窗口小部件时可以设置它们。 在下面的示例中，设置了[`SelectDateWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SelectDateWidget) 的[`years`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SelectDateWidget.years) 属性：

```python
from django import forms

BIRTH_YEAR_CHOICES = ('1980', '1981', '1982')
FAVORITE_COLORS_CHOICES = (
    ('blue', 'Blue'),
    ('green', 'Green'),
    ('black', 'Black'),
)

class SimpleForm(forms.Form):
    birth_year = forms.DateField(widget=forms.SelectDateWidget(years=BIRTH_YEAR_CHOICES))
    favorite_colors = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=FAVORITE_COLORS_CHOICES,
    )
```

有关哪些窗口小部件可用以及它们接受哪些参数的详细信息，请参阅[内置窗口小部件](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#built-in-widgets)。

## 继承自`Select`小部件的小部件

继承自[`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select) 的Widget 负责处理HTML 选项。 它们呈现给用户一个可以选择的选项列表。 不同的小部件呈现出不同的选择； [`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select)小部件本身使用`<select>` HTML列表表示，而[`RadioSelect`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.RadioSelect)使用单选按钮。

[`ChoiceField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.ChoiceField) 字段默认使用[`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select)。 Widget 上显示的选项来自[`ChoiceField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.ChoiceField)，对[`ChoiceField.choices`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.ChoiceField.choices) 的改变将更新[`Select.choices`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select.choices)。 像这样：

```shell
>>> from django import forms
>>> CHOICES = (('1', 'First',), ('2', 'Second',))
>>> choice_field = forms.ChoiceField(widget=forms.RadioSelect, choices=CHOICES)
>>> choice_field.choices
[('1', 'First'), ('2', 'Second')]
>>> choice_field.widget.choices
[('1', 'First'), ('2', 'Second')]
>>> choice_field.widget.choices = ()
>>> choice_field.choices = (('1', 'First and only',),)
>>> choice_field.widget.choices
[('1', 'First and only')]
```

提供[`choices`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select.choices) 属性的Widget 也可以用于不是基于选项的字段 ， 例如[`CharField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.CharField) —— 当选项与模型有关而不只是Widget 时，建议使用基于[`ChoiceField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.ChoiceField) 的字段。

## 定制小部件实例

当Django 渲染Widget 成HTML 时，它只渲染最少的标记 —— Django 不会添加class 的名称和特定于Widget 的其它属性。 这表示，网页上所有[`TextInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TextInput) 的外观是一样的。

有两种方法可以自定义小部件：[每个小部件实例](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#styling-widget-instances)和[每个小部件类](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#styling-widget-classes)。

### 样式小部件实例

如果你想让某个Widget 实例与其它Widget 看上去不一样，你需要在Widget 对象实例化并赋值给一个表单字段时指定额外的属性（以及可能需要在你的CSS 文件中添加一些规则）。

例如下面这个简单的表单：

```python
from django import forms

class CommentForm(forms.Form):
    name = forms.CharField()
    url = forms.URLField()
    comment = forms.CharField()
```

这个表单包含三个默认的[`TextInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TextInput) Widget，以默认的方式渲染 —— 没有CSS 类、没有额外的属性。 这表示每个Widget 的输入框将渲染得一模一样：

```shell
>>> f = CommentForm(auto_id=False)
>>> f.as_table()
<tr><th>Name:</th><td><input type="text" name="name" required /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url" required /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" required /></td></tr>
```

在真正得网页中，你可能不想让每个Widget 看上去都一样。 你可能想要给comment 一个更大的输入元素，你可能想让‘name’ Widget 具有一些特殊的CSS 类。 可以指定‘type’ 属性使用的是新式的HTML5 输入类型。 在创建Widget 时使用[`Widget.attrs`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.attrs) 参数可以实现：

```python
class CommentForm(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'special'}))
    url = forms.URLField()
    comment = forms.CharField(widget=forms.TextInput(attrs={'size': '40'}))
```

Django 将在渲染的输出中包含额外的属性：

```shell
>>> f = CommentForm(auto_id=False)
>>> f.as_table()
<tr><th>Name:</th><td><input type="text" name="name" class="special" required /></td></tr>
<tr><th>Url:</th><td><input type="url" name="url" required /></td></tr>
<tr><th>Comment:</th><td><input type="text" name="comment" size="40" required /></td></tr>
```

你还可以使用[`attrs`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.attrs) 设置HTML `id`。 参见[`BoundField.id_for_label`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.id_for_label) 示例。

### 样式小部件类

使用小部件，可以添加资源（`css`和`javascript`）并更深入地定制其外观和行为。

简而言之，您需要对窗口小部件进行子类化，并且[定义“Media”内部类](https://yiyibooks.cn/__trs__/qy/django2/topics/forms/media.html#assets-as-a-static-definition)或[创建“media”属性](https://yiyibooks.cn/__trs__/qy/django2/topics/forms/media.html#dynamic-property)。

这些方法涉及一些高级Python编程，并在[表单资产](https://yiyibooks.cn/__trs__/qy/django2/topics/forms/media.html)主题指南中有详细描述。

## 基本小部件类

基本窗口小部件类[`窗口小部件`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#django.forms.Widget)和[`MultiWidget`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#django.forms.MultiWidget)由所有[内置窗口小部件](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#built-in-widgets)子类化，并可作为自定义窗口小部件的基础。

### Widget
```
class Widget(attrs=None)
```

此抽象类无法呈现，但提供基本属性[`attrs`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#django.forms.Widget.attrs)。 您还可以在自定义窗口小部件上实现或覆盖[`render()`](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#django.forms.Widget.render)方法。

- attrs

包含要在呈现的窗口小部件上设置的HTML属性的字典。

```shell
>>> from django import forms
>>> name = forms.TextInput(attrs={'size': 10, 'title': 'Your name',})
>>> name.render('name', 'A name')
'<input title="Your name" type="text" name="name" value="A name" size="10" required />'
```

如果为属性指定值`True`或`False`，则它将呈现为HTML5布尔属性：

```shell
>>> name = forms.TextInput(attrs={'required': True})
>>> name.render('name', 'A name')
'<input name="name" type="text" value="A name" required />'
>>>
>>> name = forms.TextInput(attrs={'required': False})
>>> name.render('name', 'A name')
'<input name="name" type="text" value="A name" />'
```

- `supports_microseconds`

默认为`True`的属性。 如果设置为`False`，则`datetime`和`time`值的微秒部分将设置为`0`。

- `supports_microseconds`

属性默认为`True`。 如果设置为`False`，则`datetime`和`time`值的微秒部分将被设置为`0`。

- `format_value(value)`

清除并返回一个用于小部件模板的值。 `value`不能保证是有效的输入，因此子类的实现应该防御性地编程。

- `get_context(name, value, attrs)`

**Django中的新功能1.11。**

返回在渲染窗口小部件模板时要使用的值的字典。 默认情况下，该字典包含一个单一的键`'widget'`，它是包含以下键的小部件的字典表示形式：

`name`：`name`参数中的字段的名称。

`is_hidden`：一个布尔值，表示该小部件是否被隐藏。

`required`：一个布尔值，表示是否需要此窗口小部件的字段。

`value`：由[`format_value()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.format_value)返回的值。

`attrs`：要在已渲染的小部件上设置HTML属性。 [`attrs`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.attrs)属性和`attrs`参数的组合。

`template_name`：`self.template_name`的值。

`Widget`子类可以通过覆盖此方法来提供自定义上下文值。

- `id_for_label(id_)`

给定该字段的ID，返回此小部件的HTML ID属性，以供`<label>`使用。 如果ID不可用，则返回`None`。
这个钩子是必要的，因为一些小部件具有多个HTML元素，因此具有多个ID。 在这种情况下，该方法应该返回与widget的标签中的第一个ID相对应的ID值。

- `render(name, value, attrs=None, renderer=None)`

使用给定的渲染器将小部件渲染为HTML。 如果`renderer`是`None`，则使用[`FORM_RENDERER`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-FORM_RENDERER)设置中的渲染器。

**在Django更改1.11：**

添加了`renderer`参数。 支持不接受的子类将在Django 2.1中被删除。

- `value_from_datadict(data, files, name)`

根据一个字典和该Widget 的名称，返回该Widget 的值。 `files`可能包含来自[`request.FILES`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest.FILES)的数据。 如果没有提供value，则返回`None`。 在处理表单数据的过程中，`value_from_datadict` 可能调用多次，所以如果你自定义并添加额外的耗时处理时，你应该自己实现一些缓存机制。

- `value_omitted_from_data(data, files, name)`

给定`data`和`files`字典和此小部件的名称，返回是否有数据或文件的小部件。该方法的结果会影响模型窗体[falls back to its default](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#topics-modelform-save)。特殊情况是[`CheckboxInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.CheckboxInput)，[`CheckboxSelectMultiple`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.CheckboxSelectMultiple)和[`SelectMultiple`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SelectMultiple)，它始终返回`False`，因为未选中的复选框并未选择`<select multiple>`不会出现在HTML表单提交的数据中，因此用户是否提交了值是未知的。

- `use_required_attribute(initial)`

给定一个表单域的`initial`值，返回是否可以使用`required` 表单使用此方法与[`Field.required`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.required)和[`Form.use_required_attribute`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.Form.use_required_attribute)一起确定是否显示每个字段的`required`属性。

默认情况下，为隐藏的小部件返回`False`，否则返回`True`。 特殊情况是[`ClearableFileInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.ClearableFileInput)，当`initial`未设置时返回`False`，[`CheckboxSelectMultiple`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.CheckboxSelectMultiple)，它始终返回`False`，因为浏览器验证将需要检查所有复选框，而不是至少一个。

在与浏览器验证不兼容的自定义小部件中覆盖此方法。 例如，由隐藏的`textarea`元素支持的WSYSIWG文本编辑器小部件可能希望始终返回`False`，以避免在隐藏字段上进行浏览器验证。

### MultiWidget

```
class  MultiWidget(widgets, attrs = None）
```

由多个Widget 组合而成的Widget。 [`MultiWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.MultiWidget) 始终与[`MultiValueField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.MultiValueField) 联合使用。

`MultiWidget`具有一个必选参数：

`widgets`

一个包含需要的Widget 的可迭代对象。

以及一个必需的方法：

```
decompress(value)
```

这个方法接受来自字段的一个“压缩”的值，并返回“解压”的值的一个列表。 可以假设输入的值是合法的，但不一定是非空的。

子类**必须实现** 这个方法，而且因为值可能为空，实现必须要防卫这点。“解压”的基本原理是需要“分离”组合的表单字段的值为每个Widget 的值。

有个例子是，[`SplitDateTimeWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SplitDateTimeWidget) 将[`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime) 值分离成两个独立的值分别表示日期和时间：
```python
from django.forms import MultiWidget

class SplitDateTimeWidget(MultiWidget):

    # ...

    def decompress(self, value):
        if value:
            return [value.date(), value.time().replace(microsecond=0)]
        return [None, None]
```

>  小贴士
注意，[`MultiValueField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.MultiValueField) 有一个[`compress()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.MultiValueField.compress) 方法用于相反的工作 —— 将所有字段的值组合成一个值。

它提供一些自定义上下文：

`get_context(name, value, attrs)`

除了[`Widget.get_context()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.get_context)中描述的`'widget'`之外，`MultiValueWidget`添加了一个`widget['subwidgets']`这些可以在窗口小部件模板中循环：

```python
{% for subwidget in widget.subwidgets %}
    {% include widget.template_name with widget=subwidget %}
{% endfor %}
```

下面示例中的Widget 继承[`MultiWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.MultiWidget) 以在不同的选择框中显示年、月、日。 这个Widget 主要想用于[`DateField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.DateField) 而不是[`MultiValueField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.MultiValueField)，所以我们实现了[`value_from_datadict()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.value_from_datadict)：

```python
from datetime import date
from django.forms import widgets

class DateSelectorWidget(widgets.MultiWidget):
    def __init__(self, attrs=None):
        # create choices for days, months, years
        # example below, the rest snipped for brevity.
        years = [(year, year) for year in (2011, 2012, 2013)]
        _widgets = (
            widgets.Select(attrs=attrs, choices=days),
            widgets.Select(attrs=attrs, choices=months),
            widgets.Select(attrs=attrs, choices=years),
        )
        super().__init__(_widgets, attrs)

    def decompress(self, value):
        if value:
            return [value.day, value.month, value.year]
        return [None, None, None]

    def value_from_datadict(self, data, files, name):
        datelist = [
            widget.value_from_datadict(data, files, name + '_%s' % i)
            for i, widget in enumerate(self.widgets)]
        try:
            D = date(
                day=int(datelist[0]),
                month=int(datelist[1]),
                year=int(datelist[2]),
            )
        except ValueError:
            return ''
        else:
            return str(D)
```

构造器在一个元组中创建了多个[`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select) widget。 `super`类使用这个元组来启动widget。

必需的[`decompress()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.MultiWidget.decompress)方法将`datetime.date` 值拆成年、月和日的值，对应每个widget。 注意这个方法如何处理`value`为`None`的情况。

[`value_from_datadict()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.value_from_datadict)的默认实现会返回一个列表，对应每一个`Widget`。 当和[`MultiValueField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.MultiValueField)一起使用`MultiWidget`的时候，这样会非常合理，但是由于我们想要和拥有单一值得[`DateField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.DateField)一起使用这个widget，我们必须覆写这一方法，将所有子widget的数据组装成`datetime.date`。 这个方法从`POST` 字典中获取数据，并且构造和验证日期。 如果日期有效，会返回它的字符串，否则会返回一个空字符串，它会使`form.is_valid`返回`False`。

## 内置小部件

Django提供了所有基本HTML小部件的表示，以及`django.forms.widgets`模块中一些常用的小部件组，包括[文本输入](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#text-widgets)，[各种复选框和选择器](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#selector-widgets)，[上传文件](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#file-upload-widgets)，以及[处理多值输入](https://yiyibooks.cn/__trs__/qy/django2/ref/forms/widgets.html#composite-widgets)。

### 处理文本输入的小部件

这些Widget 使用HTML 元素`input` 和 `textarea`。

#### `TextInput`

```
class TextInput
```

输入类型：`'text'`

模版名字：`django/forms/widgets/text.html`

呈现为：`<input type =“text” ...>`

#### `NumberInput`

```
class NumberInput
```

输入类型：`number`

模版名字：`django/forms/widgets/number.html`

呈现为：`<input type =“number” ...>`

注意，不是所有浏览器的`number`输入类型都支持输入本地化的数字。 Django本身避免将它们用于将[`localize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field.localize)属性设置为`True`的字段。



#### `EmailInput`

```
class EmailInput
```

输入类型：`email`

模版名字：`django/forms/widgets/email.html`

呈现为：`<input type =“email” ...>`

#### `URLInput`

```
class URLInput
```

输入类型：`url`

模版名字：`django/forms/widgets/url.html`

呈现为：`<input type =“url” ...>`

#### `PasswordInput`

```
class PasswordInput
```

输入类型：`password`

模版名字：`django/forms/widgets/password.html`

呈现为：`<input type =“password” ...>`

接收一个可选的参数：

`render_value`

决定在验证错误后重新显示表单时，Widget 是否填充（默认为`False`）。

#### `HiddenInput`

```
class HiddenInput
```

输入类型：`hidden`

模版名字：`django/forms/widgets/hidden.html`

呈现为：`<input type =“hidden” ...>`

注意，还有一个[`MultipleHiddenInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.MultipleHiddenInput) Widget，它封装一组隐藏的输入元素。

#### `DateInput`

```
class DateInput
```

输入类型：`text`

模版名字：`django/forms/widgets/date.html`

呈现为：`<input type =“text” ...>`

接收的参数与[`TextInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TextInput) 相同，但是带有一些可选的参数：

`foramt`

字段的初始值应该显示的格式。

如果没有提供`format` 参数，默认的格式为参考[Format localization](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html)在[`DATE_INPUT_FORMATS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DATE_INPUT_FORMATS) 中找到的第一个格式。



#### `DateTimeInput`

```
class DateTimeInput
```

输入类型：`text`

模版名字：`django/forms/widgets/datetime.html`

呈现为：`<input type =“text” ...>`

接收的参数与[`TextInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TextInput) 相同，但是带有一些可选的参数：

`fromat`

字段的初始值应该显示的格式。

如果没有提供`format` 参数，默认的格式为参考[Format localization](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html)在[`DATETIME_INPUT_FORMATS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DATETIME_INPUT_FORMATS) 中找到的第一个格式。

默认情况下，时间值的微秒部分始终设置为`0`。 如果需要微秒，请使用[`supports_microseconds`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Widget.supports_microseconds)属性设置为`True`的子类。

#### `TimeInput`

```
class TimeInput
```

输入类型：`text`

模版名字：`django/forms/widgets/time.html`

呈现为：`<input type =“text” ...>`

接收的参数与[`TextInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TextInput) 相同，但是带有一些可选的参数：

`format`

字段的初始值应该显示的格式。

如果没有提供`format` 参数，默认的格式为参考[Format localization](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html)在[`TIME_INPUT_FORMATS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-TIME_INPUT_FORMATS) 中找到的第一个格式。

有关微秒的处理，请参阅[`DateTimeInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.DateTimeInput)。

#### `Textarea`

```
class Textarea
```

模版名字：`django/forms/widgets/textarea.html`

呈现为：`<textarea>...</textarea>`

### 选择器和复选框小部件

这些小部件使用HTML元素`<select>`, `<input type="checkbox">`, 和 `<input type="radio">`.

呈现多个选项的窗口小部件具有指定用于呈现每个选项的模板的`option_template_name`属性。 For example, for the [`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select) widget, `select_option.html` renders the `<option>` for a `<select>`.

#### `CheckboxInput`

```
class CheckboxInput
```

输入类型：`checkbox`

模版名字：`django/forms/widgets/checkbox.html`

呈现为：`<input type="checkbox" ...>`

接收一个可选的参数：

`check_test`

一个可调用的对象，接收`CheckboxInput` 的值并如果复选框应该勾上返回`True`。

#### `Select`

```
class Select
```

模版名字：`django/forms/widgets/select.html`

可选模版名字：`django/forms/widgets/select_option.html`

呈现为： `<select><option ...>...</select>`

`choices`

当表单字段没有`choices` 属性时，该属性是随意的。 如果字段有choice 属性，当[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field)的该属性更新时，它将覆盖你在这里的任何设置。

#### `NullBooleanSelect`

```
class NullBooleanSelect
```

模版名字：`django/forms/widgets/select.html`

可选模版名字：`django/forms/widgets/select_option.html`

Select Widget，选项为‘Unknown’、‘Yes’ 和‘No’。

#### `SelectMultiple`

```
class SelectMultiple
```

模版名字：`django/forms/widgets/select.html`

可选模版名字：`django/forms/widgets/select_option.html`

与[`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select)类似，但允许多个选择：`<select multiple="multiple">...</select>`

#### `RadioSelect`

```
class RadioSelect
```

模版名字：`django/forms/widgets/radio.html`

可选模版名字：`django/forms/widgets/radio_option.html`

类似[`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select)，但是渲染成`<li>` 标签中的一个单选按钮列表：

```html
<ul>
  <li><input type="radio" name="..."></li> 
  ... 
</ul> 
```

你可以迭代模板中的单选按钮来更细致地控制生成的HTML。 假设表单`RadioSelect` 具有一个字段`beatles`，它使用`myform` 作为Widget：

```python
{% for radio in myform.beatles %}
<div class="myradio">
    {{ radio }}
</div>
{% endfor %}
```

它将生成以下HTML：

```html
<div class="myradio">
    <label for="id_beatles_0"><input id="id_beatles_0" name="beatles" type="radio" value="john" required /> John</label>
</div>
<div class="myradio">
    <label for="id_beatles_1"><input id="id_beatles_1" name="beatles" type="radio" value="paul" required /> Paul</label>
</div>
<div class="myradio">
    <label for="id_beatles_2"><input id="id_beatles_2" name="beatles" type="radio" value="george" required /> George</label>
</div>
<div class="myradio">
    <label for="id_beatles_3"><input id="id_beatles_3" name="beatles" type="radio" value="ringo" required /> Ringo</label>
</div>
```

这包括`<label>` 标签。 你可以使用单选按钮的`id_for_label`、`choice_label` 和 `tag` 属性进行更细的控制。 例如，这个模板...

```python
{% for radio in myform.beatles %}
    <label for="{{ radio.id_for_label }}">
        {{ radio.choice_label }}
        <span class="radio">{{ radio.tag }}</span>
    </label>
{% endfor %}
```

...将导致以下HTML：

```html
<label for="id_beatles_0">
    John
    <span class="radio"><input id="id_beatles_0" name="beatles" type="radio" value="john" required /></span>
</label>

<label for="id_beatles_1">
    Paul
    <span class="radio"><input id="id_beatles_1" name="beatles" type="radio" value="paul" required /></span>
</label>

<label for="id_beatles_2">
    George
    <span class="radio"><input id="id_beatles_2" name="beatles" type="radio" value="george" required /></span>
</label>

<label for="id_beatles_3">
    Ringo
    <span class="radio"><input id="id_beatles_3" name="beatles" type="radio" value="ringo" required /></span>
</label>
```

如果你不迭代单选按钮 —— 例如，你的模板只是简单地包含`{{ myform.beatles }}` —— 它们将以`<ul>` 中的`<li>` 标签输出，就像上面一样。外部`<ul>`容器接收小部件的`id`属性，如果已定义，否则将接收[`BoundField.auto_id`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.auto_id)。当迭代单选按钮时，`for` 和`input` 标签分别包含`label` 和`id` 属性。 每个单项按钮具有一个`id_for_label` 属性来输出元素的ID。

#### `CheckboxSelectMultiple`
```
class CheckboxSelectMultiple
```

模版名字：`django/forms/widgets/checkbox_select.html`

可选模版名字：`django/forms/widgets/checkbox_option.html`

类似[`SelectMultiple`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SelectMultiple)，但是渲染成一个复选框列表：

```html
<ul>
  <li><input type="checkbox" name="..." ></li>
  ...
</ul>
```

外部`<ul>`容器接收小部件的`id`属性，如果已定义，否则将接收[`BoundField.auto_id`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#django.forms.BoundField.auto_id)。

像[`RadioSelect`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.RadioSelect)一样，您可以循环查看小部件选择的各个复选框。 与[`RadioSelect`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.RadioSelect)不同，复选框将不包含`required` HTML属性，如果该字段是必需的，因为浏览器验证将需要检查所有复选框，而不是至少检查一个。

当迭代单选按钮时，`for` 和`input` 标签分别包含`label` 和`id` 属性。 每个单项按钮具有一个`id_for_label` 属性来输出元素的ID。

### 文件上传小部件

#### `FileInput`

```
class FileInput
```

模版名字：`django/forms/widgets/file.html`

呈现为：`<input type="file" ...>`

#### `ClearableFileInput`

```
class ClearableFileInput
```

模版名字：`django/forms/widgets/clearable_file_input.html`

呈现为：`<input type =“file” ...>` 清除字段的值，如果该字段不是必需的，并具有初始数据。

### 复合小部件

#### `MultipleHiddenInput`

```
class MultipleHiddenInput
```

模版名字：`django/forms/widgets/multiple_hidden.html`

呈现为：多个`<input type =“hidden” ...>`标签

一个处理多个隐藏的Widget 的Widget，用于值为一个列表的字段。

`choices`

当表单字段没有`choices` 属性时，该属性是随意的。 如果字段有choice 属性，当[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field)的该属性更新时，它将覆盖你在这里的任何设置。

#### `SplitDateTimeWidget`

```
class SplitDateTimeWidget
```

模版名字：`django/forms/widgets/splitdatetime.html`

封装（使用[`MultiWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.MultiWidget)）两个Widget：[`DateInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.DateInput) 用于日期，[`TimeInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TimeInput) 用于时间。 必须与[`SplitDateTimeField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.SplitDateTimeField)而不是[`DateTimeField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.DateTimeField)一起使用。

`SplitDateTimeWidget` 有两个可选的属性：

`date_format`

类似[`DateInput.format`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.DateInput.format)

`time_format`

类似[`TimeInput.format`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.TimeInput.format)

#### `SplitHiddenDateTimeWidget`

```
class SplitHiddenDateTimeWidget
```

模版名字：`django/forms/widgets/splithiddendatetime.html`

类似[`SplitDateTimeWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SplitDateTimeWidget)，但是日期和时间都使用[`HiddenInput`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.HiddenInput)。

#### `SelectDateWidget`

```
class SelectDateWidget
```

模版名字：`django/forms/widgets/select_date.html`

封装三个[`Select`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.Select) Widget：分别用于年、月、日。

有几个可选参数：

`years`

一个可选的列表/元组，用于”年“选择框。 默认为包含当前年份和未来9年的一个列表。

`months`

一个可选的字典，用于”月“选择框。字典的键对应于月份的数字（从1开始），值为显示出来的月份：

```python
MONTHS = {
    1:_('jan'), 2:_('feb'), 3:_('mar'), 4:_('apr'),
    5:_('may'), 6:_('jun'), 7:_('jul'), 8:_('aug'),
    9:_('sep'), 10:_('oct'), 11:_('nov'), 12:_('dec')
}
```

`empty_label`

如果[`DateField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.DateField) 不是必选的，[`SelectDateWidget`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/widgets.html#django.forms.SelectDateWidget) 将有一个空的选项位于选项的顶部（默认为`---`）。 你可以通过`empty_label` 属性修改这个文本。 `list` 可以是一个`string`、`empty_label` 或`tuple`。 当使用字符串时，所有的选择框都带有这个空选项。 如果`tuple` 为具有3个字符串元素的`list` 或`empty_label`，每个选择框将具有它们自定义的空选项。 空选项应该按这个顺序`('year_label', 'month_label', 'day_label')`。

```python
# A custom empty label with string
field1 = forms.DateField(widget=SelectDateWidget(empty_label="Nothing"))

# A custom empty label with tuple
field1 = forms.DateField(
    widget=SelectDateWidget(
        empty_label=("Choose Year", "Choose Month", "Choose Day"),
    ),
)
```