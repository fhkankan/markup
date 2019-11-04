## ModelAdmin

```
class ModelAdmin
```

`ModelAdmin`类是模型在管理后台界面中的表示形式。 通常，它们保存在你的应用中的名为`admin.py`的文件里。 让我们来看一个关于`ModelAdmin`类非常简单的例子:

```python
from django.contrib import admin 
from myproject.myapp.models import Author 

class AuthorAdmin(admin.ModelAdmin): 
		pass 

admin.site.register(Author, AuthorAdmin)
```

> 你真的需要一个`ModelAdmin`对象吗?

在上面的例子中，`ModelAdmin`并没有定义任何自定义的值。 因此, 系统将使用默认的管理后台界面。 如果对于默认的管理后台界面足够满意，那你根本不需要自己定义`ModelAdmin`对象 — 你可以直接注册模型类而无需提供`ModelAdmin`的描述。 那么上面的例子可以简化成：

```python
from django.contrib import admin 
from myproject.myapp.models import Author

admin.site.register(Author)
```

### `register`装饰器

```
register(models, site=django.admin.sites.site)
```

还可以用一个装饰器来注册您的`ModelAdmin`类

```python
from django.contrib import admin 
from .models import Author  

@admin.register(Author) 
class AuthorAdmin(admin.ModelAdmin):
  pass 
```

如果你使用的不是默认的`AdminSite`，那么这个装饰器可以接收一些`ModelAdmin`作为参数，以及一个可选的关键字参数 `site` ：（这里使用装饰器来注册需要注册的类和模块的，请特别留意紧跟装饰器后面关于ModelAdmin的声明，前面是Author，后面是PersonAdmin，我的理解是后一种情况 下注册的类都可以用PersonAdmin来作为接口）：

```python
from django.contrib import admin 
from .models import Author, Reader, Editor
from myproject.admin_site import custom_admin_site  

@admin.register(Author, Reader, Editor, site=custom_admin_site) class PersonAdmin(admin.ModelAdmin): 
  pass 
```

在python2中，如果您在类的`__init__())`方法中引用了模型的admin类，则不能使用此装饰器，例如， `super（PersonAdmin， self）.__ init __（*args, **kwargs）`。 但是，在Python3中，通过使用`super().__ init __(*args, **kwargs)`可以避免这个问题； 在python2中，你必须使用`admin.site.register()`而不能使用装饰器方式。

### 探索admin文件

当你将 `'django.contrib.admin'`加入到`INSTALLED_APPS`设置中, Django就会自动搜索每个应用的`admin`模块并将其导入。

```
class apps.AdminConfig
```

这是 admin的默认[`AppConfig`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/applications.html#django.apps.AppConfig) 类. 它在 Django 启动时调用[`autodiscover()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.autodiscover) .

```
class apps.SimpleAdminConfig
```

这个类和 [`AdminConfig`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.apps.AdminConfig)的作用一样,除了它不调用[`autodiscover()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.autodiscover).

```
autodiscover()
```

这个函数尝试导入每个安装的应用中的`admin` 模块。 这些模块用于注册模型到Admin 中。通常，当Django启动时，您将不需要直接调用此函数作为[`AdminConfig`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.apps.AdminConfig)调用该函数。

如果您正在使用自定义 `AdminSite`，则通常会将所有`ModelAdmin`子类导入到代码中，并将其注册到自定义`AdminSite`。 在这种情况下， 为了禁用auto-discovery,在你的[`INSTALLED_APPS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-INSTALLED_APPS) 设置中，应该用 `'django.contrib.admin'`代替`'django.contrib.admin.apps.SimpleAdminConfig'` 。

### `ModelAdmin`的选项

`ModelAdmin` 非常灵活。 它有几个选项来处理自定义界面。 所有的选项都在 `ModelAdmin` 子类中定义：

```python
from django.contrib import admin

class AuthorAdmin(admin.ModelAdmin):
    date_hierarchy = 'pub_date'
```

```
ModelAdmin.actions
```

在修改列表页面可用的操作列表。 详细信息请查看[Admin actions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/actions.html) .

```
ModelAdmin.actions_on_top
ModelAdmin.actions_on_bottom
```

控制actions的下拉框出现在页面的位置。 默认情况下，管理员更改列表显示页面顶部的操作`（actions_on_top = True； actions_on_bottom = False）`。

```
ModelAdmin.actions_selection_counter
```

控制选择计数器是否紧挨着下拉菜单action 默认的admin 更改列表将会显示它 `(actions_selection_counter = True)`.

```
ModelAdmin.date_hierarchy
```

把 date_hierarchy 设置为在你的model 中的DateField或DateTimeField的字段名，然后更改列表页面将包含这个字段基于日期的下拉导航。
例如：

```python
date_hierarchy = 'pub_date' 
```

您也可以使用`__`查找在相关模型上指定一个字段，例如：

```python
date_hierarchy = 'author__pub_date'
```

这将根据现有数据智能地填充自己，例如，如果所有的数据都是一个月里的, 它将只显示天级别的数据.
**在Django更改1.11：**添加了相关模型引用字段的能力。

>注
>`date_hierarchy` 在内部使用[`QuerySet.datetimes()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.datetimes). 当时区支持启用时，请参考它的一些文档说明。([`USE_TZ = True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)).

```
ModelAdmin.empty_value_display
```

此属性将覆盖空的字段（`None`，空字符串等）的默认显示值。 默认值为`-`（破折号）。 像这样：

```python
from django.contrib import admin  

class AuthorAdmin(admin.ModelAdmin):     
  empty_value_display = '-empty-' 
```

您还可以覆盖[`empty_value_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.empty_value_display)的所有管理页面的`AdminSite.empty_value_display`，或者对于特定字段，例如：

```python
from django.contrib import admin 

class AuthorAdmin(admin.ModelAdmin):  
  fields = ('name', 'title', 'view_birth_date') 
  
  def view_birth_date(self, obj):   
    return obj.birth_date     
  
  view_birth_date.empty_value_display = '???' 
```

```
ModelAdmin.exclude
```

如果设置了这个属性，它表示应该从表单中去掉的字段列表。例如，让我们来考虑下面的模型：

```python
from django.db import models  class Author(models.Model):     name = models.CharField(max_length=100)     title = models.CharField(max_length=3)     birth_date = models.DateField(blank=True, null=True)
```

如果你希望`title` 模型的表单只包含`name` 和`Author` 字段, 你应该显式说明`fields` 或`exclude`，像这样：

```python
from django.contrib import admin  

class AuthorAdmin(admin.ModelAdmin):
  fields = ('name', 'title')
  
  class AuthorAdmin(admin.ModelAdmin):  
    exclude = ('birth_date',)
```

由于Author 模型只有三个字段，`birth_date`、`title`和 `name`，上述声明产生的表单将包含完全相同的字段。

```
ModelAdmin.fields
```

使用`fields`选项可以在“添加”和“更改”页面上的表单中进行简单的布局更改，例如仅显示可用字段的一个子集，修改其顺序或将其分组为行。 例如,可以定义一个简单的管理表单的版本使用[`django.contrib.flatpages.models.FlatPage`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/flatpages.html#django.contrib.flatpages.models.FlatPage) 模块像下面这样:

```python
class FlatPageAdmin(admin.ModelAdmin):
  fields = ('url', 'title', 'content')
```

在上面的例子中, 只有字段`content`, `title` 和 `url` 将会在表单中顺序的显示. `fields`能够包含在 [`ModelAdmin.readonly_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.readonly_fields) 中定义的作为只读显示的值对于更复杂的布局需求，请参阅[`fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)选项。不同于 [`list_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_display)，`fields` 选项 只包含model中的字段名或者通过[`form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.form)指定的表单。 只有当它们列在[`readonly_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.readonly_fields)中，它才能包含callables要在同一行显示多个字段， 就把那些字段打包在一个元组里。 在此示例中，`url`和`title`字段将显示在同一行上，`content`字段将在其自己的行下显示：

```python
class FlatPageAdmin(admin.ModelAdmin): 
  fields = (('url', 'title'), 'content') 
```

注此`fields`选项不应与[`fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)选项中的`fields`字典键混淆，如下一节所述。如果`editable=True`和[`fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets) 选项都不存在, Django将会默认显示每一个不是 `fields` 并且 `AutoField`的字段, 在单一的字段集，和在模块中定义的字段有相同的顺序

```
ModelAdmin.fieldsets
```

设置`fieldsets` 控制管理“添加”和 “更改” 页面的布局.`fieldsets` 是一个以二元元组为元素的列表, 每一个二元元组代表一个在管理表单的 `<fieldset>` ( `<fieldset>` 是表单的一部分.)二元元组的格式是 `(name, field_options)`, 其中 `name` 是一个字符串相当于 fieldset的标题， `field_options` 是一个关于 fieldset的字典信息,一个字段列表包含在里面。一个完整的例子, 来自于[`django.contrib.flatpages.models.FlatPage`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/flatpages.html#django.contrib.flatpages.models.FlatPage) 模块:

```python
from django.contrib import admin

class FlatPageAdmin(admin.ModelAdmin):
    fieldsets = (
        (None, {
            'fields': ('url', 'title', 'content', 'sites')
        }),
        ('Advanced options', {
            'classes': ('collapse',),
            'fields': ('registration_required', 'template_name'),
        }),
    )
```

在管理界面的结果看起来像这样:![../../../_images/fieldsets.png](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_images/fieldsets.png)
如果`editable=True`和[`fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields) 选项都不存在, Django将会默认显示每一个不是 `fieldsets` 并且 `AutoField`的字段, 在单一的字段集，和在模块中定义的字段有相同的顺序。`field_options` 字典有以下关键字:

- `fields`字段名元组将显示在该fieldset. 此键必选.例如：

```python
{ 'fields': ('first_name', 'last_name', 'address', 'city', 'state'), }
```

就像[`fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields) 选项, 显示多个字段在同一行, 包裹这些字段在一个元组. 在这个例子中, `first_name` 和 `last_name` 字段将显示在同一行:

```python
{ 'fields': (('first_name', 'last_name'), 'address', 'city', 'state'), } 

```

`fields` 能够包含定义在[`readonly_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.readonly_fields) 中显示的值作为只读.如果添加可调用的名称到`fields`中,相同的规则适用于[`fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields)选项: 可调用的必须在 [`readonly_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.readonly_fields)列表中.

- classes

`classes`包含要应用于字段集的额外CSS类的列表或元组。例如：

```
{ 'classes': ('wide', 'extrapretty'), } 
```

通过默认的管理站点样式表定义的两个有用的classes 是 `collapse` 和 `wide`. Fieldsets 使用 `collapse` 样式将会在初始化时展开并且替换掉一个 “click to expand” 链接. Fieldsets 使用 `wide` 样式将会有额外的水平空格.

- `description`

一个可选择额外文本的字符串显示在每一个fieldset的顶部,在fieldset头部的底下. 字符串没有被[`TabularInline`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.TabularInline) 渲染由于它的布局.记住这个值*不是* HTML-escaped 当它显示在管理接口中时. 如果你愿意，这允许你包括HTML。 另外，你可以使用纯文本和 `django.utils.html.escape()` 避免任何HTML特殊字符。

```
ModelAdmin.filter_horizontal
```

默认的, [`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField) 会在管理站点上显示一个`<select multiple>`.（多选框）． 但是，当选择多个时多选框非常难用. 添加一个 [`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)到该列表将使用一个漂亮的低调的JavaScript中的“过滤器”界面,允许搜索选项。 选和不选选项框并排出现。 参考[`filter_vertical`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.filter_vertical) 使用垂直界面。

```
ModelAdmin.filter_vertical
```

与[`filter_horizontal`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.filter_horizontal)相同，但使用过滤器界面的垂直显示，其中出现在所选选项框上方的未选定选项框。

```
ModelAdmin.form
```

默认情况下， 会根据你的模型动态创建一个`ModelForm`。 它被用来创建呈现在添加/更改页面上的表单。 你可以很容易的提供自己的`ModelForm` 来重写表单默认的添加/修改行为。 或者，你可以使用[`ModelAdmin.get_form()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_form) 方法自定义默认的表单，而不用指定一个全新的表单。

例子见[Adding custom validation to the admin](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#admin-custom-validation)部分。

> 注

如果你在[`ModelForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.ModelForm)中定义 `Meta.exclude`属性，那么也必须定义 `Meta.model`或`Meta.fields`属性。 然而，当admin本身定义了fields，则`Meta.fields`属性将被忽略。如果`ModelAdmin` 仅仅只是给Admin 使用，那么最简单的解决方法就是忽略`Meta.model` 属性，因为`ModelForm` 将自动选择应该使用的模型。 或者，你也可以设置在 `Meta` 类中的 `fields = []` 来满足 `ModelForm` 的合法性。

> 注

如果 `exclude` 和 `ModelAdmin` 同时定义了一个 `ModelForm` 选项，那么 `ModelAdmin` 具有更高的优先级：

```python
from django import forms
from django.contrib import admin
from myapp.models import Person

class PersonForm(forms.ModelForm):

    class Meta:
        model = Person
        exclude = ['name']

class PersonAdmin(admin.ModelAdmin):
    exclude = ['age']
    form = PersonForm
```

在上例中， “age” 字段将被排除而 “name” 字段将被包含在最终产生的表单中。

```
ModelAdmin.formfield_overrides
```

这个属性通过一种临时的方案来覆盖现有的模型中[`Field`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/fields.html#django.forms.Field) （字段）类型在admin site中的显示类型。 `formfield_overrides` 在类初始化的时候通过一个字典类型的变量来对应模型字段类型与实际重载类型的关系。因为概念有点抽象，所以让我们来举一个具体的例子。 `formfield_overrides` 常被用于让一个已有的字段显示为自定义控件。 所以，试想一下我们写了一个 `RichTextEditorWidget` 然后我们想用它来代替`<textarea>`用于输入大段文字。
下面就是我们如何做到这样的替换。

```python
from django.db import models
from django.contrib import admin

# Import our custom widget and our model from where they're defined
from myapp.widgets import RichTextEditorWidget
from myapp.models import MyModel

class MyModelAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.TextField: {'widget': RichTextEditorWidget},
    }
```

注意字典的键是一个实际的字段类型，而*不是*一个具体的字符。 该值是另一个字典；这些参数将被传递给表单域的`__init__()`方法。 有关详细信息，请参见[The Forms API](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html)。

>警告

如果你想用一个关系字段的自定义界面 (即 `ForeignKey` 或者 `ManyToManyField`)， 确保你没有在`raw_id_fields, radio_fields, autocomplete_fields`中包含那个字段名。

`formfield_overrides`不会让您更改`raw_id_fields,radio_fields, autocomplete_fields`设置的关系字段上的窗口小部件。 这是因为`raw_id_fields, radio_fields, autocomplete_fields`暗示自己的自定义小部件。

```
ModelAdmin.inlines
```

请参见下面的[`InlineModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin)对象以及[`ModelAdmin.get_formsets_with_inlines()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_formsets_with_inlines)。

```
ModelAdmin.list_display
```

使用`list_display` 去控制哪些字段会显示在Admin 的修改列表页面中。例如：`

```python
list_display = ('first_name', 'last_name') 
```

如果你没有设置`list_display`，Admin 站点将只显示一列表示每个对象的`__str__()` （Python 2 中是`__unicode__`）。

在`list_display`中，你有4种赋值方式可以使用：

- 模型的字段。

```python
class PersonAdmin(admin.ModelAdmin):

		list_display = ('first_name', 'last_name') 
```

- 一个接受对象实例作为参数的可调用对象。

```python
def upper_case_name(obj):     return ("%s %s" % (obj.first_name, obj.last_name)).upper() upper_case_name.short_description = 'Name'  class PersonAdmin(admin.ModelAdmin):     list_display = (upper_case_name,) 
```

- 一个表示`ModelAdmin` 中某个属性的字符串。 行为与可调用对象相同。

```python
class PersonAdmin(admin.ModelAdmin):
    list_display = ('upper_case_name',)

    def upper_case_name(self, obj):
        return ("%s %s" % (obj.first_name, obj.last_name)).upper()
    upper_case_name.short_description = 'Name'
```

- 表示模型中某个属性的字符串。 它的行为与可调用对象几乎相同，但这时的`self` 是模型实例。 这里是一个完整的模型示例︰

```python
from django.db import models
from django.contrib import admin

class Person(models.Model):
    name = models.CharField(max_length=50)
    birthday = models.DateField()

    def decade_born_in(self):
        return self.birthday.strftime('%Y')[:3] + "0's"
    decade_born_in.short_description = 'Birth decade'

class PersonAdmin(admin.ModelAdmin):
    list_display = ('name', 'decade_born_in')
```

关于`list_display` 要注意的几个特殊情况︰

- 如果字段是一个`ForeignKey()`，Django 将展示相关对象的`__str__()` （Python 2 上是`__unicode__`）。

- 不支持`ManyToManyField` 字段， 因为这将意味着对表中的每一行执行单独的SQL 语句。 如果尽管如此你仍然想要这样做，请给你的模型一个自定义的方法，并将该方法名称添加到 `list_display`。 （`list_display` 的更多自定义方法请参见下文）。

- 如果该字段为`BooleanField` 或`NullBooleanField`，Django 会显示漂亮的"on"或"off"图标而不是`True` 或`False`。

- 如果给出的字符串是模型、`ModelAdmin` 的一个方法或可调用对象，Django 将默认转义HTML输出。 要转义用户输入并允许自己的未转义标签，请使用[`format_html()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/utils.html#django.utils.html.format_html)。
  下面是一个完整的示例模型︰

  ```python
  from django.db import models
  from django.contrib import admin
  from django.utils.html import format_html
  
  class Person(models.Model):
    	first_name = models.CharField(max_length=50)
    	last_name = models.CharField(max_length=50)
    	color_code = models.CharField(max_length=6)
  
    	def colored_name(self):
        	return format_html(
            	'<span style="color: #{};">{} {}</span>',
           		self.color_code,
            	self.first_name,
            	self.last_name,
        	)
  
  class PersonAdmin(admin.ModelAdmin):
    	list_display = ('first_name', 'last_name', 'colored_name')
  ```

- 正如一些例子已经证明，当使用可调用，模型方法或`ModelAdmin`方法时，您可以通过向可调用添加`short_description`属性来自定义列的标题。

- 如果一个字段的值是`None`，一个空字符串，或没有元素的iterable，Django将显示`-`（破折号）。 您可以使用[`AdminSite.empty_value_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.empty_value_display)重写此项：

  ```python
  from django.contrib import admin 
  admin.site.empty_value_display = '(None)'
  ```

  您也可以使用[`ModelAdmin.empty_value_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.empty_value_display)：

  ```python
  class PersonAdmin(admin.ModelAdmin):
  		empty_value_display = 'unknown'
  ```

  或在现场一级：

  ```python
  class PersonAdmin(admin.ModelAdmin): 
  		list_display = ('name', 'birth_date_view') 
      
      def birth_date_view(self, obj): 
      		return obj.birth_date 
      
      birth_date_view.empty_value_display = 'unknown' 
  ```

- 如果给出的字符串是模型的一个方法, `ModelAdmin`或一个返回 True 或False 的可调用的方法，然后赋值给方法一个`boolean`属性为`True`， Django将显示漂亮的"on"或"off"图标。
  下面是一个完整的示例模型︰

```python
from django.db import models
from django.contrib import admin

class Person(models.Model):
    first_name = models.CharField(max_length=50)
    birthday = models.DateField()

    def born_in_fifties(self):
        return self.birthday.strftime('%Y')[:3] == '195'
    born_in_fifties.boolean = True

class PersonAdmin(admin.ModelAdmin):
    list_display = ('name', 'born_in_fifties')
```

- `list_display`（Python 2 上是`__unicode__()`）方法在`__str__()` 中同样合法，就和任何其他模型方法一样，所以下面这样写完全OK︰

```python
list_display = ('__str__', 'some_other_field')
```

- 通常情况下，`list_display` 的元素如果不是实际的数据库字段不能用于排序（因为 Django 所有的排序都在数据库级别）。然而，如果`list_display` 元素表示数据库的一个特定字段，你可以通过设置 元素的`admin_order_field` 属性表示这一事实。

像这样：

```python
from django.db import models
from django.contrib import admin
from django.utils.html import format_html

class Person(models.Model):
    first_name = models.CharField(max_length=50)
    color_code = models.CharField(max_length=6)

    def colored_first_name(self):
        return format_html(
            '<span style="color: #{};">{}</span>',
            self.color_code,
            self.first_name,
        )

    colored_first_name.admin_order_field = 'first_name'

class PersonAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'colored_first_name')
```

上面的示例告诉Django 在Admin 中按照按`first_name` 排序时依据`colored_first_name` 字段。要表示按照`admin_order_field` 降序排序，你可以在该字段名称前面使用一个连字符前缀。 使用上面的示例，这会看起来像︰

```
colored_first_name.admin_order_field = '-first_name' ``admin_order_field
```

支持查询查询，以按相关模型的值进行排序。 此示例包括列表显示中的“作者名字”列，并允许以名字排序：

```python
class Blog(models.Model):
    title = models.CharField(max_length=255)
    author = models.ForeignKey(Person, on_delete=models.CASCADE)

class BlogAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'author_first_name')

    def author_first_name(self, obj):
        return obj.author.first_name

    author_first_name.admin_order_field = 'author__first_name'
```

- `list_display` 的元素也可以是属性。 不过请注意，由于方式属性在Python 中的工作方式，在属性上设置`property()` 只能使用 `short_description` 函数，**不** 能使用`@property` 装饰器。
  像这样：

```python
class Person(models.Model):
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)

    def my_property(self):
        return self.first_name + ' ' + self.last_name
    my_property.short_description = "Full name of the person"

    full_name = property(my_property)

class PersonAdmin(admin.ModelAdmin):
    list_display = ('full_name',)
```

- `<th>` 中的字段名称还将作为HTML 输出的CSS 类， 形式为每个`column-<field_name>` 元素上具有`list_display`。 例如这可以用于在CSS 文件中设置列的宽度。
- Django 会尝试以下面的顺序解释`list_display` 的每个元素︰模型的字段。可调用对象。表示`ModelAdmin` 属性的字符串。表示模型属性的字符串。例如，如果`first_name` 既是模型的一个字段又是`ModelAdmin` 的一个属性，使用的将是模型字段。

```
ModelAdmin.list_display_links
```

使用`list_display_links`可以控制[`list_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_display)中的字段是否应该链接到对象的“更改”页面。默认情况下，更改列表页将链接第一列 - `list_display`中指定的第一个字段 - 到每个项目的更改页面。 但是`list_display_links`可让您更改此设置：

- 将其设置为`None`，根本不会获得任何链接。
- 将其设置为要将其列转换为链接的字段列表或元组（格式与`list_display`相同）。您可以指定一个或多个字段。 只要这些字段出现在`list_display`中，Django不会关心多少（或多少）字段被链接。 唯一的要求是，如果要以这种方式使用`list_display_links`，则必须定义`list_display`。

在此示例中，`first_name`和`last_name`字段将链接到更改列表页面上：

```python
class PersonAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'last_name', 'birthday')
    list_display_links = ('first_name', 'last_name')
```

在此示例中，更改列表页面网格将没有链接：

```python
class AuditEntryAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'message')
    list_display_links = None
```

```
ModelAdmin.list_editable
```

将`list_editable`设置为模型上的字段名称列表，这将允许在更改列表页面上进行编辑。 也就是说，`list_editable`中列出的字段将在更改列表页面上显示为表单小部件，允许用户一次编辑和保存多行。

> 注
> `list_editable`与特定方式与其他选项进行交互；您应该注意以下规则：`list_editable`中的任何字段也必须位于`list_display`中。 您无法编辑未显示的字段！同一字段不能在`list_editable`和`list_display_links`中列出 - 字段不能同时是表单和链接。如果这些规则中的任一个损坏，您将收到验证错误。

```
ModelAdmin.list_filter
```

`list_filter` 设置激活激活Admin 修改列表页面右侧栏中的过滤器，如下面的屏幕快照所示︰![../../../_images/list_filter.png](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_images/list_filter.png)
`list_filter` 应该是一个列表或元组，其每个元素应该是下面类型中的一种：

- 字段名称，其指定的字段应该是`ManyToManyField`、`IntegerField`、`ForeignKey`、`DateField`、`CharField`、`BooleanField` 或`DateTimeField`，例如︰

```python
class PersonAdmin(admin.ModelAdmin): 

		list_filter = ('is_staff', 'company') 
```

`list_filter` 中的字段名称也可以使用`__` 查找跨关联关系，例如︰

```python
class PersonAdmin(admin.UserAdmin): 
		list_filter = ('company__name',) 
```

- 一个继承自`django.contrib.admin.SimpleListFilter` 的类，你需要给它提供`title` 和 `parameter_name` 属性并重写`lookups` 和`queryset` 方法，例如︰

```python
from datetime import date

from django.contrib import admin
from django.utils.translation import gettext_lazy as _

class DecadeBornListFilter(admin.SimpleListFilter):
    # Human-readable title which will be displayed in the
    # right admin sidebar just above the filter options.
    title = _('decade born')

    # Parameter for the filter that will be used in the URL query.
    parameter_name = 'decade'

    def lookups(self, request, model_admin):
        """
        Returns a list of tuples. The first element in each
        tuple is the coded value for the option that will
        appear in the URL query. The second element is the
        human-readable name for the option that will appear
        in the right sidebar.
        """
        return (
            ('80s', _('in the eighties')),
            ('90s', _('in the nineties')),
        )

    def queryset(self, request, queryset):
        """
        Returns the filtered queryset based on the value
        provided in the query string and retrievable via
        `self.value()`.
        """
        # Compare the requested value (either '80s' or '90s')
        # to decide how to filter the queryset.
        if self.value() == '80s':
            return queryset.filter(birthday__gte=date(1980, 1, 1),
                                    birthday__lte=date(1989, 12, 31))
        if self.value() == '90s':
            return queryset.filter(birthday__gte=date(1990, 1, 1),
                                    birthday__lte=date(1999, 12, 31))

class PersonAdmin(admin.ModelAdmin):
    list_filter = (DecadeBornListFilter,)
```

> 注

作为一种方便，`HttpRequest` 对象将传递给`lookups` 和`queryset` 方法，例如︰

```python
class AuthDecadeBornListFilter(DecadeBornListFilter):

    def lookups(self, request, model_admin):
        if request.user.is_superuser:
            return super().lookups(request, model_admin)

    def queryset(self, request, queryset):
        if request.user.is_superuser:
            return super().queryset(request, queryset)
```

也作为一种方便，`ModelAdmin` 对象将传递给`lookups` 方法，例如如果你想要基于现有的数据查找︰

```python
class AdvancedDecadeBornListFilter(DecadeBornListFilter):

    def lookups(self, request, model_admin):
        """
        Only show the lookups if there actually is
        anyone born in the corresponding decades.
        """
        qs = model_admin.get_queryset(request)
        if qs.filter(birthday__gte=date(1980, 1, 1),
                      birthday__lte=date(1989, 12, 31)).exists():
            yield ('80s', _('in the eighties'))
        if qs.filter(birthday__gte=date(1990, 1, 1),
                      birthday__lte=date(1999, 12, 31)).exists():
            yield ('90s', _('in the nineties'))
```

- 一个元组，第一个元素是字段名称，第二个元素是从继承自`django.contrib.admin.FieldListFilter` 的一个类，例如︰

```python
class PersonAdmin(admin.ModelAdmin):
    list_filter = (
        ('is_staff', admin.BooleanFieldListFilter),
    )
```

您可以使用`RelatedOnlyFieldListFilter`将相关模型的选择限制在该关系中涉及的对象中：

```python
class BookAdmin(admin.ModelAdmin):
    list_filter = (
        ('author', admin.RelatedOnlyFieldListFilter),
    )
```

假设`author` 是`User` 模型的一个`ForeignKey`，这将限制`list_filter` 的选项为编写过书籍的用户，而不是所有用户。

> 注
>
> `FieldListFilter` API 被视为内部的，可能会改变。

列表过滤器通常仅在过滤器有多个选择时才会出现。 过滤器的`has_output()`方法控制是否显示。

也可以指定自定义模板用于渲染列表筛选器︰

```python
class FilterWithCustomTemplate(admin.SimpleListFilter):
    template = "custom_template.html"

```

有关具体示例，请参阅Django（`admin/filter.html`）提供的默认模板。

```
ModelAdmin.list_max_show_all

```

设置`list_max_show_all`以控制在“显示所有”管理更改列表页面上可以显示的项目数。 只有当总结果计数小于或等于此设置时，管理员才会在更改列表上显示“显示全部”链接。 默认情况下，设置为`200`。

```
ModelAdmin.list_per_page
```

`list_per_page` 设置控制Admin 修改列表页面每页中显示多少项。 默认设置为`100`。

```
ModelAdmin.list_select_related
```

设置`list_select_related`以告诉Django在检索管理更改列表页面上的对象列表时使用[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)。 这可以节省大量的数据库查询。

该值应该是布尔值，列表或元组。 默认值为`False`。

当值为`True`时，将始终调用`select_related()`。 当值设置为`False`时，如果存在任何`ForeignKey`，Django将查看`list_display`并调用`select_related()` 。

如果您需要更细粒度的控制，请使用元组（或列表）作为`list_select_related`的值。 空元组将阻止Django调用`select_related`。 任何其他元组将直接传递到`select_related`作为参数。 像这样：

```python
class ArticleAdmin(admin.ModelAdmin):
    list_select_related = ('author', 'category')
```

将会调用`select_related('author', 'category')`.如果需要根据请求指定动态值，则可以实现[`get_list_select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_list_select_related)方法。

```
ModelAdmin.ordering
```

  设置`ordering`以指定如何在Django管理视图中对对象列表进行排序。 这应该是与模型的[`ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.ordering)参数格式相同的列表或元组。如果没有提供，Django管理员将使用模型的默认排序。如果您需要指定动态顺序（例如，根据用户或语言），您可以实施[`get_ordering()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_ordering)方法。

```
ModelAdmin.paginator
```

paginator类用于分页。 默认情况下，使用[`django.core.paginator.Paginator`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/pagination.html#django.core.paginator.Paginator)。 如果自定义paginator类没有与[`django.core.paginator.Paginator`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/pagination.html#django.core.paginator.Paginator)相同的构造函数接口，则还需要为[`ModelAdmin.get_paginator()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_paginator) 。

```
ModelAdmin.prepopulated_fields
```

将`prepopulated_fields`设置为将字段名称映射到其应预先填充的字段的字典：

```python
class ArticleAdmin(admin.ModelAdmin): 
		prepopulated_fields = {"slug": ("title",)} 
```

设置时，给定字段将使用一些JavaScript来从分配的字段填充。 此功能的主要用途是自动从一个或多个其他字段生成`SlugField`字段的值。 生成的值是通过连接源字段的值，然后将该结果转换为有效的字节（例如用空格替换破折号）来生成的。`prepopulated_fields` 不能接受 `DateTimeField`, `ForeignKey`, `OneToOneField`, 和 `ManyToManyField` 字段.

```
ModelAdmin.preserve_filters

```

  管理员现在在创建，编辑或删除对象后保留列表视图中的过滤器。 您可以将此属性设置为`False`，以恢复之前清除过滤器的行为。

```
ModelAdmin.radio_fields
```

默认情况下，Django的管理员为`ForeignKey`或者有`choices`集合的字段使用一个下拉菜单(<select>). 如果`radio_fields`中存在字段，Django将使用单选按钮接口。 假设`group`是`Person`模型上的 `ForeignKey`

```python
class PersonAdmin(admin.ModelAdmin):   
		radio_fields = {"group": admin.VERTICAL} 
```

您可以选择使用`django.contrib.admin`模块中的`VERTICAL`或`HORIZONTAL`。除非是`choices`或设置了`ForeignKey`，否则不要在`radio_fields`中包含字段。

```
ModelAdmin.autocomplete_fields
```

Django 2.0的新功能。

autocomplete_fields是要更改为Select2自动完成输入的ForeignKey和/或ManyToManyField字段的列表。

默认情况下，管理员对这些字段使用选择框界面（<select>）。有时，您不想承担选择所有相关实例以显示在下拉菜单中的开销。

Select2输入看起来与默认输入类似，但是具有搜索功能，该功能异步加载选项。如果相关模型具有许多实例，则这将更快，更友好。

您必须在相关对象的ModelAdmin上定义search_fields，因为自动完成搜索会使用它。
结果的排序和分页由相关ModelAdmin的get_ordering（）和get_paginator（）方法控制。

在下面的示例中，ChoiceAdmin为问题的ForeignKey具有一个自动完成字段。结果由question_text字段过滤，并由date_created字段排序：

```python
class QuestionAdmin(admin.ModelAdmin):
    ordering = ['date_created']
    search_fields = ['question_text']

class ChoiceAdmin(admin.ModelAdmin):
    autocomplete_fields = ['question']
```

```
ModelAdmin.raw_id_fields
```

默认情况下，Django的管理员为`ForeignKey`的字段使用一个下拉菜单(<select>). 有时候你不想在下拉菜单中显示所有相关实例产生的开销。`raw_id_fields`是一个字段列表，你希望将`ForeignKey` 或`ManyToManyField` 转换成`Input` 窗口部件：

```python
class ArticleAdmin(admin.ModelAdmin):     
		raw_id_fields = ("newspaper",) 
```

如果该字段是一个`ForeignKey`，`Input` `raw_id_fields` Widget 应该包含一个外键，或者如果字段是一个`ManyToManyField` 则应该是一个逗号分隔的值的列表。 `raw_id_fields` Widget 在字段旁边显示一个放大镜按钮，允许用户搜索并选择一个值︰![../../../_images/raw_id_fields.png](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_images/raw_id_fields.png)

```
ModelAdmin.readonly_fields
```

默认情况下，管理后台将所有字段显示为可编辑。 此选项中的任何字段（应为`list`或`tuple`）将按原样显示其数据，不可编辑；它们也被排除在用于创建和编辑的[`ModelForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.ModelForm)之外。 请注意，指定[`ModelAdmin.fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields)或[`ModelAdmin.fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)时，只读字段必须包含进去才能显示（否则将被忽略）。

如果在未通过[`ModelAdmin.fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields)或[`ModelAdmin.fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)定义显式排序的情况下使用`readonly_fields`，则它们将在所有可编辑字段之后添加。

只读字段不仅可以显示模型字段中的数据，还可以显示模型方法的输出或`ModelAdmin`类本身的方法。 这与[`ModelAdmin.list_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_display)的行为非常相似。 这提供了一种使用管理界面提供对正在编辑的对象的状态的反馈的简单方法，例如：

```python
from django.contrib import admin
from django.utils.html import format_html_join
from django.utils.safestring import mark_safe

class PersonAdmin(admin.ModelAdmin):
    readonly_fields = ('address_report',)

    def address_report(self, instance):
        # assuming get_full_address() returns a list of strings
        # for each line of the address and you want to separate each
        # line by a linebreak
        return format_html_join(
            mark_safe('<br/>'),
            '{}',
            ((line,) for line in instance.get_full_address()),
        ) or mark_safe("<span class='errors'>I can't determine this address.</span>")

    # short_description functions like a model field's verbose_name
    address_report.short_description = "Address"
```

```
ModelAdmin.save_as
```

设置`save_as`以在管理员更改表单上启用“另存为”功能。通常，对象有三个保存选项：“保存”，“保存并继续编辑”和“保存并添加其他”。 如果`save_as`是`True`，“保存并添加另一个”将被替换为创建新对象（使用新ID）而不是更新的“另存为”按钮现有的对象。默认情况下，`save_as` 设置为`False`。

```
ModelAdmin.save_as_continue
```

当[`save_as=True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.save_as)时，保存新对象后的默认重定向是该对象的更改视图。 如果设置`save_as_continue=False`，则重定向将是更改列表视图。默认情况下，`save_as_continue`设置为`True`。

```
ModelAdmin.save_on_top
```

设置`save_on_top`可在表单顶部添加保存按钮。通常，保存按钮仅出现在表单的底部。 如果您设置`save_on_top`，则按钮将同时显示在顶部和底部。默认情况下，`save_on_top`设置为`False`。

```
ModelAdmin.search_fields
```

`search_fields` 设置启用Admin 更改列表页面上的搜索框。 此属性应设置为每当有人在该文本框中提交搜索查询将搜索的字段名称的列表。这些字段应该是某种文本字段，如`CharField` 或`TextField`。 你还可以通过查询API 的"跟随"符号进行`ForeignKey` 或`ManyToManyField` 上的关联查找：

```python
search_fields = ['foreign_key__related_fieldname'] 
```

例如，如果您有一个作者的博客条目，以下定义将允许通过作者的电子邮件地址搜索博客条目：

```
search_fields = ['user__email'] 
```

如果有人在Admin 搜索框中进行搜索，Django 拆分搜索查询为单词并返回包含每个单词的所有对象，不区分大小写，其中每个单词必须在至少一个`search_fields`。 例如，如果`search_fields` 设置为`['first_name', 'last_name']`，用户搜索`john lennon`，Django 的行为将相当于下面的这个`WHERE` SQL 子句︰

```
WHERE (first_name ILIKE '%john%' OR last_name ILIKE '%john%') AND (first_name ILIKE '%lennon%' OR last_name ILIKE '%lennon%') 
```

若要更快和/或更严格的搜索，请在字典名称前面加上前缀︰

`^`

使用'^'运算符来匹配从字段开始的起始位置。 例如，如果`search_fields` 设置为`['^first_name', '^last_name']`，用户搜索`john lennon` 时，Django 的行为将等同于下面这个`WHERE` SQL 字句：

```
WHERE (first_name ILIKE 'john%' OR last_name ILIKE 'john%') AND (first_name ILIKE 'lennon%' OR last_name ILIKE 'lennon%') 
```

此查询比正常`'%john%'` 查询效率高，因为数据库只需要检查某一列数据的开始，而不用寻找整列数据。 另外，如果列上有索引，有些数据库可能能够对于此查询使用索引，即使它是`LIKE` 查询。

`=`

使用'='运算符不区分大小写的精确匹配。 例如，如果`search_fields` 设置为`['=first_name', '=last_name']`，用户搜索`john lennon` 时，Django 的行为将等同于下面这个`WHERE` SQL 字句：

```
WHERE (first_name ILIKE 'john' OR last_name ILIKE 'john') AND (first_name ILIKE 'lennon' OR last_name ILIKE 'lennon')
```

注意，该查询输入通过空格分隔，所以根据这个示例，目前不能够搜索`first_name` 精确匹配`'john winston'`（包含空格）的所有记录。

`@`

使用'@'运算符执行全文匹配。 这就像默认的搜索方法，但使用索引。 目前这只适用于MySQL。如果你需要自定义搜索，你可以使用[`ModelAdmin.get_search_results()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_search_results) 来提供附件的或另外一种搜索行为。

```
ModelAdmin.show_full_result_count
```

设置`show_full_result_count`以控制是否应在过滤的管理页面上显示对象的完整计数（例如`99 结果 103 total）`）。 如果此选项设置为`False`，则像`99 结果 （显示 ）`。默认情况下，`show_full_result_count=True`生成一个查询，对表执行完全计数，如果表包含大量行，这可能很昂贵。

```
ModelAdmin.view_on_site
```

设置`view_on_site`以控制是否显示“在网站上查看”链接。 此链接将带您到一个URL，您可以在其中显示已保存的对象。此值可以是布尔标志或可调用的。 如果`True`（默认值），对象的[`get_absolute_url()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.get_absolute_url)方法将用于生成网址。如果您的模型有[`get_absolute_url()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.get_absolute_url)方法，但您不想显示“在网站上查看”按钮，则只需将`view_on_site`设置为`False`：

```python
from django.contrib import admin  

class PersonAdmin(admin.ModelAdmin):    
		view_on_site = False 
```

如果它是可调用的，它接受模型实例作为参数。 像这样：

```python
from django.contrib import admin
from django.urls import reverse

class PersonAdmin(admin.ModelAdmin):
    def view_on_site(self, obj):
        url = reverse('person-detail', kwargs={'slug': obj.slug})
        return 'https://example.com' + url
```

#### 自定义模板选项

[Overriding admin templates](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#admin-overriding-templates) 一节描述如何重写或扩展默认Admin 模板。 使用以下选项来重写[`ModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin) 视图使用的默认模板︰

```
ModelAdmin.add_form_template
```

`add_view()`使用的自定义模板的路径。

```
ModelAdmin.change_form_template
```

`change_view()`使用的自定义模板的路径。

```
ModelAdmin.change_list_template
```

`changelist_view()` 使用的自定义模板的路径。

```
ModelAdmin.delete_confirmation_template
```

`delete_view()`使用的自定义模板，用于删除一个或多个对象时显示一个确认页。

```
ModelAdmin.delete_selected_confirmation_template
```

`delete_selected` 使用的自定义模板，用于删除一个或多个对象时显示一个确认页。 参见[actions documentation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/actions.html)。

```
ModelAdmin.object_history_template
```

[`history_view()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.history_view) 使用的自定义模板的路径。

```
ModelAdmin.popup_response_template
```

[`response_add()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.response_add)，[`response_change()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.response_change)和[`response_delete()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.response_delete)使用的自定义模板的路径。

### `ModelAdmin`的方法

>警告

当覆盖[`ModelAdmin.save_model()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.save_model)和[`ModelAdmin.delete_model()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.delete_model)时，代码必须保存/删除对象。 它们不是为了否决权，而是允许您执行额外的操作。

```
ModelAdmin.save_model(request, obj, form, change)
```

根据是否要添加或更改对象，为save_model方法提供HttpRequest，一个模型实例，一个ModelForm实例以及一个布尔值。 覆盖此方法允许进行前或后保存操作。 使用[`Model.save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)调用`super().save_model()`来保存对象。例如，在保存之前将`request.user`附加到对象：

```python
from django.contrib import admin 

class ArticleAdmin(admin.ModelAdmin): 
  def save_model(self, request, obj, form, change):  
    obj.user = request.user 
    super(ArticleAdmin, self).save_model(request, obj, form, change) 
```

```python
ModelAdmin.delete_model(request, obj)
```

`delete_model`方法给出了`HttpRequest`和模型实例。 覆盖此方法允许进行前或后删除操作。 使用[`Model.delete()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.delete)调用`super().delete_model()`来删除对象。

```
ModelAdmin.save_formset(request, form, formset, change)
```

`ModelForm`方法是给予`HttpRequest`，父`save_formset`实例和基于是否添加或更改父对象的布尔值。例如，要将`request.user`附加到每个已更改的formset模型实例：
```python
class ArticleAdmin(admin.ModelAdmin):
    def save_formset(self, request, form, formset, change):
        instances = formset.save(commit=False)
        for obj in formset.deleted_objects:
            obj.delete()
        for instance in instances:
            instance.user = request.user
            instance.save()
        formset.save_m2m()
```
另请参见[Saving objects in the formset](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#saving-objects-in-the-formset)。

```python
ModelAdmin.get_ordering(request)
```
`get_ordering`方法将`request`作为参数，并且预期返回`list`或`tuple`，以便类似于[`ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.ordering)属性。 像这样：
```python
class PersonAdmin(admin.ModelAdmin):

    def get_ordering(self, request):
        if request.user.is_superuser:
            return ['name', 'rank']
        else:
            return ['name']
```

```
ModelAdmin.get_search_results(request, queryset, search_term)
```

`get_search_results`方法将显示的对象列表修改为与提供的搜索项匹配的对象列表。 它接受请求，应用当前过滤器的查询集以及用户提供的搜索项。 它返回一个包含被修改以实现搜索的查询集的元组，以及一个指示结果是否可能包含重复项的布尔值。默认实现搜索在[`ModelAdmin.search_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.search_fields)中命名的字段。此方法可以用您自己的自定义搜索方法覆盖。 例如，您可能希望通过整数字段搜索，或使用外部工具（如Solr或Haystack）。 您必须确定通过搜索方法实现的查询集更改是否可能在结果中引入重复项，并在返回值的第二个元素中返回`True`。例如，要通过`name`和`age`搜索，您可以使用：

```python
class PersonAdmin(admin.ModelAdmin):
    list_display = ('name', 'age')
    search_fields = ('name',)

    def get_search_results(self, request, queryset, search_term):
        queryset, use_distinct = super().get_search_results(request, queryset, search_term)
        try:
            search_term_as_int = int(search_term)
        except ValueError:
            pass
        else:
            queryset |= self.model.objects.filter(age=search_term_as_int)
        return queryset, use_distinct
```

在这将导致数字字段的字符串比较上，这个实现比`search_fields = ('name', '= age')有效 ，例如，在PostgreSQL上有` ... OR UPPER("polls_choice"."votes"::text) = UPPER('4')` 

```python
ModelAdmin.save_related(request, form, formsets, change)
```

`ModelForm`方法给出了`HttpRequest`，父`save_related`实例，内联表单列表和一个布尔值，添加或更改。 在这里，您可以对与父级相关的对象执行任何预保存或后保存操作。 请注意，此时父对象及其形式已保存。

```python
ModelAdmin.get_autocomplete_fields(request)
```
new in Django 2.0

`get_autocomplete_fields()`方法被赋予HttpRequest，并期望返回字段名称的列表或元组，该字段或列表将与自动完成小部件一起显示，如上文ModelAdmin.autocomplete_fields部分所述。

```
ModelAdmin.get_readonly_fields(request, obj=None)
```

`list`方法在添加表单上给予`tuple`和`obj`（或`HttpRequest`），希望返回将以只读形式显示的字段名称的`get_readonly_fields`或`None`，如上面在[`ModelAdmin.readonly_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.readonly_fields)部分中所述。

```
ModelAdmin.get_prepopulated_fields(request, obj=None)
```

`dictionary`方法在添加表单上给予`obj`和`HttpRequest`（或`get_prepopulated_fields`），预期返回`None`，如上面在[`ModelAdmin.prepopulated_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.prepopulated_fields)部分中所述。

```
ModelAdmin.get_list_display(request)
```

`list`方法被赋予`HttpRequest`，并且希望返回字段名称的`get_list_display`或`tuple`显示在如上所述的[`ModelAdmin.list_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_display)部分中的changelist视图
上。

```
ModelAdmin.get_list_display_links(request, list_display)
```

get_list_display_links方法被赋予HttpRequest以及[`ModelAdmin.get_list_display()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_list_display)返回的列表或元组。预期将返回更改列表上将链接到更改视图的字段名称的`tuple`或`list`或`None`，如上所述在[`ModelAdmin.list_display_links`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_display_links)部分中。

```
ModelAdmin.get_exclude(request, obj=None)
```

Django中的新功能1.11。

将为get_exclude方法提供HttpRequest和正在编辑的obj（或在添加表单上为obj），并且该方法应返回字段列表，如[`ModelAdmin.exclude`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.exclude)中所述。

```
ModelAdmin.get_fields(request, obj=None)
```

`obj`方法被赋予`HttpRequest`和`get_fields`被编辑（或在添加表单上`None`），希望返回字段列表，如上面在[`ModelAdmin.fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields)部分中所述。

```
ModelAdmin.get_fieldsets(request, obj=None)
```

`<fieldset>`方法是在添加表单上给予`obj`和`HttpRequest`（或`get_fieldsets`），期望返回二元组列表，其中每个二元组在管理表单页面上表示`None`，如上面在[`ModelAdmin.fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)部分。

```
ModelAdmin.get_list_filter(request)
```

`HttpRequest`方法被赋予`get_list_filter`，并且期望返回与[`list_filter`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_filter)属性相同类型的序列类型。

```
ModelAdmin.get_list_select_related(request)
```

`get_list_select_related`方法被赋予HttpRequest，并且应该返回一个布尔值或列表，就像[`ModelAdmin.list_select_related`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_select_related).

```
ModelAdmin.get_search_fields(request)
```

`HttpRequest`方法被赋予`get_search_fields`，并且期望返回与[`search_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.search_fields)属性相同类型的序列类型。

```
ModelAdmin.get_inline_instances(request, obj=None)
```

`list`方法在添加表单上给予`tuple`和`obj`（或`HttpRequest`），预期会返回`get_inline_instances`或`None`的[`InlineModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin)对象，如下面的[`InlineModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin)部分所述。 例如，以下内容将返回内联，而不进行基于添加，更改和删除权限的默认过滤：

```python
class MyModelAdmin(admin.ModelAdmin):
    inlines = (MyInline,)

    def get_inline_instances(self, request, obj=None):
        return [inline(self.model, self.admin_site) for inline in self.inlines]
```

如果覆盖此方法，请确保返回的内联是[`inlines`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.inlines)中定义的类的实例，或者在添加相关对象时可能会遇到“错误请求”错误。

```
ModelAdmin.get_urls()
```

`get_urls` 的`ModelAdmin` 方法返回ModelAdmin 将要用到的URLs，方式与URLconf 相同。 因此，你可以用[URL dispatcher](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/http/urls.html) 中所述的方式扩展它们︰

```python
from django.contrib import admin
from django.template.response import TemplateResponse
from django.urls import path

class MyModelAdmin(admin.ModelAdmin):
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('my_view/', self.my_view),
        ]
        return my_urls + urls

    def my_view(self, request):
        # ...
        context = dict(
           # Include common variables for rendering the admin template.
           self.admin_site.each_context(request),
           # Anything else you want in the context...
           key=value,
        )
        return TemplateResponse(request, "sometemplate.html", context)
```

如果你想要使用Admin 的布局，可以从`admin/base_site.html` 扩展︰
```python
{% extends "admin/base_site.html" %} 
{% block content %} 
...
{% endblock %} 
```

> 注
自定义的模式包含在正常的Admin URLs*之前*：Admin URL 模式非常宽松，将匹配几乎任何内容，因此你通常要追加自定义的URLs 到内置的URLs 前面。在此示例中，`/admin/` 的访问点将是`/admin/myapp/mymodel/my_view/`（假设Admin URLs 包含在`my_view` 下）。

但是, 上述定义的函数`self.my_view` 将遇到两个问题：

它不执行任何权限检查，所以会向一般公众开放。

它不提供任何HTTP头的详细信息以防止缓存。 这意味着，如果页面从数据库检索数据，而且缓存中间件处于活动状态，页面可能显示过时的信息。

因为这通常不是你想要的，Django 提供一个方便的封装函数来检查权限并标记视图为不可缓存的。 这个包装器在`ModelAdmin`实例中是`AdminSite.admin_view()`（即`self.admin_site.admin_view`）；使用它像这样：

```python
class MyModelAdmin(admin.ModelAdmin):
    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            path('my_view/', self.admin_site.admin_view(self.my_view))
        ]
        return my_urls + urls
```

请注意上述第5行中的被封装的视图︰

```
url(r'^my_view/$', self.admin_site.admin_view(self.my_view)) 
```

此包装将保护`self.my_view`未经授权的访问，并将应用[`django.views.decorators.cache.never_cache()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/http/decorators.html#django.views.decorators.cache.never_cache)装饰器，以确保缓存不缓存中间件是活动的。

如果该页面是可缓存的，但你仍然想要执行权限检查，你可以传递`cacheable=True` 的`AdminSite.admin_view()` 参数︰

```python
url(r'^my_view/$', self.admin_site.admin_view(self.my_view, cacheable=True)) 
```

`ModelAdmin`视图具有`model_admin`属性。 其他`AdminSite`视图具有`admin_site`属性。

```
ModelAdmin.get_form(request, obj=None, **kwargs)
```

返回Admin中添加和更改视图使用的[`ModelForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.ModelForm) 类，请参阅[`add_view()`和 [`change_view()`。

其基本的实现是使用[`modelform_factory()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/models.html#django.forms.models.modelform_factory) 来子类化[`form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.form)，修改如[`fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields) 和[`exclude`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.exclude)属性。

所以，举个例子，如果你想要为超级用户提供额外的字段，你可以换成不同的基类表单，就像这样︰

```python
class MyModelAdmin(admin.ModelAdmin):
    def get_form(self, request, obj=None, **kwargs):
        if request.user.is_superuser:
            kwargs['form'] = MySuperuserForm
        return super().get_form(request, obj, **kwargs)
```

你也可以简单地直接返回一个自定义的[`ModelForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.htm
  l#django.forms.ModelForm) 类。
```
ModelAdmin.get_formsets_with_inlines(request, obj=None)
```

产量（`FormSet`，[`InlineModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin)）对用于管理添加和更改视图。例如，如果您只想在更改视图中显示特定的内联，则可以覆盖`get_formsets_with_inlines`，如下所示：

```python
class MyModelAdmin(admin.ModelAdmin):
    inlines = [MyInline, SomeOtherInline]

    def get_formsets_with_inlines(self, request, obj=None):
        for inline in self.get_inline_instances(request, obj):
            # hide MyInline in the add view
            if isinstance(inline, MyInline) and obj is None:
                continue
            yield inline.get_formset(request, obj), inline
```

```python
ModelAdmin.formfield_for_foreignkey(db_field, request, **kwargs)
```

`formfield_for_foreignkey`上的`ModelAdmin`方法允许覆盖外键字段的默认窗体字段。 例如，要根据用户返回此外键字段的对象子集：

```python
class MyModelAdmin(admin.ModelAdmin):
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "car":
            kwargs["queryset"] = Car.objects.filter(owner=request.user)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)
```

这使用`User`实例过滤`Car`外键字段，只显示由`HttpRequest`实例拥有的汽车。

```
ModelAdmin.formfield_for_manytomany(db_field, request, **kwargs)
```

与`formfield_for_foreignkey`方法类似，可以覆盖`formfield_for_manytomany`方法来更改多对多字段的默认窗体字段。 例如，如果所有者可以拥有多个汽车，并且汽车可以属于多个所有者 - 多对多关系，则您可以过滤`Car`外键字段，仅显示由`User`：

```python
class MyModelAdmin(admin.ModelAdmin):
    def formfield_for_manytomany(self, db_field, request, **kwargs):
        if db_field.name == "cars":
            kwargs["queryset"] = Car.objects.filter(owner=request.user)
        return super().formfield_for_manytomany(db_field, request, **kwargs)
```

```
ModelAdmin.formfield_for_choice_field(db_field, request, **kwargs)
```

与`formfield_for_choice_field`和`formfield_for_manytomany`方法类似，可以覆盖`formfield_for_foreignkey`方法更改已声明选择的字段的默认窗体字段。 例如，如果超级用户可用的选择应与正式工作人员可用的选项不同，则可按以下步骤操作：

```python
class MyModelAdmin(admin.ModelAdmin):
    def formfield_for_choice_field(self, db_field, request, **kwargs):
        if db_field.name == "status":
            kwargs['choices'] = (
                ('accepted', 'Accepted'),
                ('denied', 'Denied'),
            )
            if request.user.is_superuser:
                kwargs['choices'] += (('ready', 'Ready for deployment'),)
        return super().formfield_for_choice_field(db_field, request, **kwargs)
```

>注
在表单域中设置的任何`choices`属性将仅限于表单字段。 如果模型上的相应字段有选择集，则提供给表单的选项必须是这些选择的有效子集，否则，在保存模型本身之前验证模型本身时，表单提交将失败并显示[`ValidationError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.ValidationError) 。

```
ModelAdmin.get_changelist(request, **kwargs)
```

返回要用于列表的`Changelist`类。 默认情况下，使用`django.contrib.admin.views.main.ChangeList`。 通过继承此类，您可以更改列表的行为。

```
ModelAdmin.get_changelist_form(request, **kwargs)
```

返回[`ModelForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.ModelForm)类以用于更改列表页面上的`Formset`。 要使用自定义窗体，例如：

```python
from django import forms

class MyForm(forms.ModelForm):
    pass

class MyModelAdmin(admin.ModelAdmin):
    def get_changelist_form(self, request, **kwargs):
        return MyForm
```

> 注
如果你在[`ModelForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.ModelForm)中定义 `Meta.exclude`属性，那么也必须定义 `Meta.model`或`Meta.fields`属性。 但是，`ModelAdmin`会忽略此值，并使用[`ModelAdmin.list_editable`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_editable)属性覆盖该值。 最简单的解决方案是省略`Meta.model`属性，因为`ModelAdmin`将提供要使用的正确模型。

```
ModelAdmin.get_changelist_formset(request, **kwargs)
```

如果使用[`list_editable`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_editable)，则返回[ModelFormSet](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#model-formsets)类以在更改列表页上使用。 要使用自定义表单集，例如：

```python
rom django.forms import BaseModelFormSet

class MyAdminFormSet(BaseModelFormSet):
    pass

class MyModelAdmin(admin.ModelAdmin):
    def get_changelist_formset(self, request, **kwargs):
        kwargs['formset'] = MyAdminFormSet
        return super().get_changelist_formset(request, **kwargs)
```

```
ModelAdmin.lookup_allowed(lookup, value)
```

可以从URL查询字符串中的查找过滤更改列表页面中的对象。 例如，这是[`list_filter`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_filter)的工作原理。 查询与[`QuerySet.filter()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.filter)（例如`user__email=user@example.com`）中使用的查找类似。 由于查询字符串中的查询可以由用户操纵，因此必须对其进行清理，以防止未经授权的数据暴露。给定了`lookup_allowed()`方法，从查询字符串（例如`'user__email'`）和相应的值（例如`'user@example.com'`），并返回一个布尔值，表示是否允许使用参数过滤changelist的`QuerySet`。 如果`lookup_allowed()`返回`False`，则会引发`DisallowedModelAdminLookup`（[`SuspiciousOperation`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.SuspiciousOperation)的子类）。默认情况下，`lookup_allowed()`允许访问模型的本地字段，[`list_filter`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.list_filter)中使用的字段路径（但不是来自[`get_list_filter()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_list_filter)的路径）并且[`limit_choices_to`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey.limit_choices_to)所需的查找在[`raw_id_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.raw_id_fields)中正常运行。覆盖此方法可自定义[`ModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin)子类允许的查找。

```
ModelAdmin.has_add_permission(request)
```

如果允许添加对象，则应返回`True`，否则返回`False`。

```
ModelAdmin.has_change_permission(request, obj=None)
```

如果允许编辑obj，则应返回`True`，否则返回`False`。 如果obj为`False`，则应返回`True`或`None`以指示是否允许对此类对象进行编辑（例如，`False`将被解释为意味着当前用户不允许编辑此类型的任何对象）。

```
ModelAdmin.has_delete_permission(request, obj=None)
```

如果允许删除obj，则应返回`True`，否则返回`False`。 如果obj是`None`，应该返回`True`或`False`以指示是否允许删除此类型的对象（例如，`False`将被解释为意味着当前用户不允许删除此类型的任何对象）。

```
ModelAdmin.has_module_permission(request)
```

如果在管理索引页上显示模块并允许访问模块的索引页，则应返回`True`，否则`False`。 默认情况下使用[`User.has_module_perms()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.has_module_perms)。 覆盖它不会限制对添加，更改或删除视图的访问，[`has_add_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_add_permission)，[`has_change_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_change_permission)和[`has_delete_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_delete_permission)用于那。

```python
ModelAdmin.get_queryset(request)
```

`ModelAdmin`上的`get_queryset`方法会返回管理网站可以编辑的所有模型实例的[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)。 覆盖此方法的一个用例是显示由登录用户拥有的对象：

```python
class MyModelAdmin(admin.ModelAdmin):
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        if request.user.is_superuser:
            return qs
        return qs.filter(author=request.user)
```

```python
ModelAdmin.message_user(request, message, level=messages.INFO, extra_tags='', fail_silently=False)
```

使用[`django.contrib.messages`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/messages.html#module-django.contrib.messages) 向用户发送消息。 参见[custom ModelAdmin example](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/actions.html#custom-admin-action)。关键字参数运行你修改消息的级别、添加CSS 标签，如果`contrib.messages` 框架没有安装则默默的失败。 关键字参数与[`django.contrib.messages.add_message()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/messages.html#django.contrib.messages.add_message) 的参数相匹配，更多细节请参见这个函数的文档。 有一个不同点是级别除了使用整数/常数传递之外还以使用字符串。

```python
ModelAdmin.get_paginator(request, queryset, per_page, orphans=0, allow_empty_first_page=True)
```

返回要用于此视图的分页器的实例。 默认情况下，实例化[`paginator`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.paginator)的实例。

```python
ModelAdmin.response_add(request, obj, post_url_continue=None)
```

为[`add_view()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.add_view)阶段确定[`HttpResponse`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpResponse)。`response_add`在管理表单提交后，在对象和所有相关实例已创建并保存之后调用。 您可以覆盖它以在对象创建后更改默认行为。

```python
ModelAdmin.response_change(request, obj)
```

确定[`change_view()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.change_view) 阶段的[`HttpResponse`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpResponse)。`response_change` 在Admin 表单提交并保存该对象和所有相关的实例之后调用。 您可以重写它来更改对象修改之后的默认行为。

```python
ModelAdmin.response_delete(request, obj_display, obj_id)
```

为[`delete_view()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.delete_view)阶段确定[`HttpResponse`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpResponse)。在对象已删除后调用`response_delete`。 您可以覆盖它以在对象被删除后更改默认行为。`obj_display`是具有已删除对象名称的字符串。`obj_id`是用于检索要删除的对象的序列化标识符。

```python
ModelAdmin.get_changeform_initial_data(request)
```

用于管理员更改表单上的初始数据的挂钩。 默认情况下，字段从`GET`参数给出初始值。 例如，`initial_value`会将`name`字段的初始值设置为`?name=initial_value`。该方法应该返回表单中的字典 `{ '字段名'： 'fieldval'}`:

```python
def get_changeform_initial_data(self, request):
    return {'name': 'custom_initial_value'}
```


#### 其他方法

```python
ModelAdmin.add_view(request, form_url='', extra_context=None)
```

Django视图为模型实例添加页面。 见下面的注释。

```python
ModelAdmin.change_view(request, object_id, form_url='', extra_context=None)
```

模型实例编辑页面的Django视图。 见下面的注释。

```python
ModelAdmin.changelist_view(request, extra_context=None)
```

Django视图为模型实例更改列表/操作页面。 见下面的注释。

```python
ModelAdmin.delete_view(request, object_id, extra_context=None)
```

模型实例删除确认页面的Django 视图。 见下面的注释。

```python
ModelAdmin.history_view(request, object_id, extra_context=None)
```

显示给定模型实例的修改历史的页面的Django视图。

与上一节中详述的钩型`ModelAdmin`方法不同，这五个方法实际上被设计为从管理应用程序URL调度处理程序调用为Django视图，以呈现处理模型实例的页面CRUD操作。 因此，完全覆盖这些方法将显着改变管理应用程序的行为。

覆盖这些方法的一个常见原因是增加提供给呈现视图的模板的上下文数据。 在以下示例中，覆盖更改视图，以便为渲染的模板提供一些额外的映射数据，否则这些数据将不可用：

```python
class MyModelAdmin(admin.ModelAdmin):

    # A template for a very customized change view:
    change_form_template = 'admin/myapp/extras/openstreetmap_change_form.html'
    
    def get_osm_info(self):
        # ...
        pass
    
    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        extra_context['osm_data'] = self.get_osm_info()
        return super(MyModelAdmin, self).change_view(
            request, object_id, form_url, extra_context=extra_context,
        )
```

这些视图返回[`TemplateResponse`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/template-response.html#django.template.response.TemplateResponse)实例，允许您在渲染之前轻松自定义响应数据。 有关详细信息，请参阅[TemplateResponse documentation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/template-response.html)。

### `ModelAdmin`资产定义

有时候你想添加一些CSS和/或JavaScript到添加/更改视图。 这可以通过在`Media`上使用`ModelAdmin`内部类来实现：

```python
class ArticleAdmin(admin.ModelAdmin):
    class Media:
        css = {
            "all": ("my_styles.css",)
        }
        js = ("my_code.js",)
```

[staticfiles app](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/staticfiles.html)将[`STATIC_URL`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-STATIC_URL)（或[`MEDIA_URL`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-MEDIA_URL)如果[`STATIC_URL`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-STATIC_URL)为`None`资产路径。 相同的规则适用于表单上的[regular asset definitions on forms](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/media.html#form-asset-paths)。

#### jQuery

Django管理JavaScript使用[jQuery](https://jquery.com/)库。

为了避免与用户提供的脚本或库冲突，Django的jQuery（版本2.2.3）命名为`django.jQuery`。 如果您想在自己的管理JavaScript中使用jQuery而不包含第二个副本，则可以使用更改列表上的`django.jQuery`对象和添加/编辑视图。

默认情况下，[`ModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin)类需要jQuery，因此除非有特定需要，否则不需要向您的`ModelAdmin`的媒体资源列表添加jQuery。 例如，如果您需要将jQuery库放在全局命名空间中（例如使用第三方jQuery插件时）或者如果您需要更新的jQuery版本，则必须包含自己的副本。

Django提供了jQuery的未压缩和“缩小”版本，分别是`jquery.js`和`jquery.min.js`。

[`ModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin)和[`InlineModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin)具有`media`属性，可返回存储到JavaScript文件的路径的`Media`对象列表形式和/或格式。 如果[`DEBUG`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DEBUG)是`True`，它将返回各种JavaScript文件的未压缩版本，包括`jquery.js`；如果没有，它将返回“最小化”版本。


### 向admin 添加自定义验证

在管理员中添加数据的自定义验证是很容易的。 自动管理界面重用[`django.forms`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/api.html#module-django.forms)，并且`ModelAdmin`类可以定义您自己的形式：

```python
class ArticleAdmin(admin.ModelAdmin):
    form = MyArticleAdminForm
```

`MyArticleAdminForm`可以在任何位置定义，只要在需要的地方导入即可。 现在，您可以在表单中为任何字段添加自己的自定义验证：

```python
class MyArticleAdminForm(forms.ModelForm):
    def clean_name(self):
        # do something that validates your data
        return self.cleaned_data["name"]
```

重要的是你在这里使用`ModelForm`否则会破坏。 有关详细信息，请参阅[custom validation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/validation.html)上的[forms](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/index.html)文档，更具体地说，[model form validation notes](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#overriding-modelform-clean-method)。