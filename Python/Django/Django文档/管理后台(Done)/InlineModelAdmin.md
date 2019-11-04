## InlineModelAdmin
```
class InlineModelAdmin
```

```
class  TabularInline
```

```
class  StackedInline
```
此管理界面能够在一个界面编辑多个Model。 这些称为内联。 假设你有这两个模型：

```python
from django.db import models

class Author(models.Model):
   name = models.CharField(max_length=100)

class Book(models.Model):
   author = models.ForeignKey(Author, on_delete=models.CASCADE)
   title = models.CharField(max_length=100)
```

您可以在作者页面上编辑作者创作的书籍.您可以通过在`ModelAdmin.inlines`中指定模型来为模型添加内联：

```python
from django.contrib import admin

class BookInline(admin.TabularInline):
    model = Book

class AuthorAdmin(admin.ModelAdmin):
    inlines = [
        BookInline,
    ]
```

`Django提供了两个`InlineModelAdmin`的子类如下:
- [`TabularInline`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.TabularInline)
- [`StackedInline`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.StackedInline)这两者之间仅仅是在用于呈现他们的模板上有区别。

###  options 

`InlineModelAdmin`与`ModelAdmin`具有许多相同的功能，并添加了一些自己的功能（共享功能实际上是在`BaseModelAdmin`超类中定义的）。 
#### 共享功能
共享功能包括：

- [`form`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.form)
- [`fieldets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)
- [`fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fields)
- [`formfield_overrides`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.formfield_overrides)
- [`exclude`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.exclude)
- [`filter_horizontal`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.filter_horizontal)
- [`filter_vertical`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.filter_vertical)
- [`ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.ordering)
- [`prepopulated_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.prepopulated_fields)
- [`get_queryset()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_queryset)
- [`radio_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.radio_fields)
- [`readonly_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.readonly_fields)
- [`raw_id_fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.raw_id_fields)
- [`formfield_for_choice_field()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.formfield_for_choice_field)
- [`formfield_for_foreignkey()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.formfield_for_foreignkey)
- [`formfield_for_manytomany()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.formfield_for_manytomany)
- [`has_add_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_add_permission)
- [`has_change_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_change_permission)
- [`has_delete_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_delete_permission)
- [`has_module_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.has_module_permission)

#### 新增功能

`InlineModelAdmin.model`

内联正在使用的模型。 这是必需的。

`InlineModelAdmin.fk_name`

模型上的外键的名称。 在大多数情况下，这将自动处理，但如果同一父模型有多个外键，则必须显式指定`fk_name`。

`InlineModelAdmin.formset`

默认为[`BaseInlineFormSet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.models.BaseInlineFormSet)。 使用自己的表单可以给你很多自定义的可能性。 内联围绕[model formsets](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#model-formsets)构建。

`InlineModelAdmin.form`

`form`的值默认为`ModelForm`。 这是在为此内联创建表单集时传递到[`inlineformset_factory()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/models.html#django.forms.models.inlineformset_factory)的内容。

> 警告

在为`InlineModelAdmin`表单编写自定义验证时，请谨慎编写依赖于父模型功能的验证。 如果父模型无法验证，则可能会处于不一致状态，如[Validation on a ModelForm](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#validation-on-modelform)中的警告中所述。

`InlineModelAdmin.classes`

Django中的新功能1.10。

包含额外CSS类的列表或元组，以应用于为内联呈现的字段集。 默认为`None`。 与[`fieldsets`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.fieldsets)中配置的类一样，带有`collapse`类的内联将最初折叠，并且它们的标题将具有一个小的“show”链接。

`InlineModelAdmin.extra`

这控制除初始形式外，表单集将显示的额外表单的数量。 有关详细信息，请参阅[formsets documentation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/formsets.html)。对于具有启用JavaScript的浏览器的用户，提供了“添加另一个”链接，以允许除了由于`extra`参数提供的内容之外添加任意数量的其他内联。如果当前显示的表单数量超过`max_num`，或者用户未启用JavaScript，则不会显示动态链接。

[`InlineModelAdmin.get_extra()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.get_extra)还允许您自定义额外表单的数量。

`InlineModelAdmin.max_num`

这控制在内联中显示的表单的最大数量。 这不直接与对象的数量相关，但如果值足够小，可以。 有关详细信息，请参阅[Limiting the number of editable objects](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#model-formsets-max-num)。

[`InlineModelAdmin.get_max_num()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.get_max_num)还允许您自定义最大数量的额外表单。

`InlineModelAdmin.min_num`

这控制在内联中显示的表单的最小数量。 有关详细信息，请参阅[`modelformset_factory()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/forms/models.html#django.forms.models.modelformset_factory)。

[`InlineModelAdmin.get_min_num()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.get_min_num)还允许您自定义显示的表单的最小数量。

`InlineModelAdmin.raw_id_fields`

默认情况下，Django的管理员将选择框界面（`<select>`）用于`ForeignKey`字段。. 有时候你不想在下拉菜单中显示所有相关实例产生的开销。`ForeignKey` 是一个字段列表，你希望将`Input` 或`raw_id_fields` 转换成`ManyToManyField` Widget：

```python
class BookInline(admin.TabularInline):
    model = Book
    raw_id_fields = ("pages",)
```

`InlineModelAdmin.template`

用于在页面上呈现内联的模板。

`InlineModelAdmin.verbose_name`

覆盖模型的内部`verbose_name`类中找到的`Meta`。

`InlineModelAdmin.verbose_name_plural`

覆盖模型的内部`verbose_name_plural`类中的`Meta`。

`InlineModelAdmin.can_delete`

指定是否可以在内联中删除内联对象。 默认为`True`。

`InlineModelAdmin. show_change_link`

指定是否可以在admin中更改的内联对象具有指向更改表单的链接。 默认为`False`。

`InlineModelAdmin.get_formset(request, obj=None，** kwargs)`

返回[`BaseInlineFormSet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/forms/modelforms.html#django.forms.models.BaseInlineFormSet)类，以在管理员添加/更改视图中使用。 请参阅[`ModelAdmin.get_formsets_with_inlines`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_formsets_with_inlines)的示例。

`InlineModelAdmin.get_extra(request, obj=None，** kwargs)`

返回要使用的其他内联表单的数量。 默认情况下，返回[`InlineModelAdmin.extra`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.extra)属性。覆盖此方法以编程方式确定额外的内联表单的数量。 例如，这可以基于模型实例（作为关键字参数`obj`传递）：

```python
class BinaryTreeAdmin(admin.TabularInline):
    model = BinaryTree

    def get_extra(self, request, obj=None, **kwargs):
        extra = 2
        if obj:
            return extra - obj.binarytree_set.count()
        return extra
```

`InlineModelAdmin.get_max_num(request, obj=None，** kwargs)`

返回要使用的额外内联表单的最大数量。 默认情况下，返回[`InlineModelAdmin.max_num`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.max_num)属性。覆盖此方法以编程方式确定内联表单的最大数量。 例如，这可以基于模型实例（作为关键字参数`obj`传递）：
```python
class BinaryTreeAdmin(admin.TabularInline):
    model = BinaryTree

    def get_max_num(self, request, obj=None, **kwargs):
        max_num = 10
        if obj and obj.parent:
            return max_num - 5
        return max_num
```

`InlineModelAdmin.get_min_num(request, obj=None，** kwargs)`

返回要使用的内联表单的最小数量。 默认情况下，返回[`InlineModelAdmin.min_num`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.InlineModelAdmin.min_num)属性。

覆盖此方法以编程方式确定最小内联表单数。 例如，这可以基于模型实例（作为关键字参数`obj`传递）。

### 使用具有两个或多个外键的模型与同一个父模型

有时可能有多个外键到同一个模型。 以这个模型为例：

```python
from django.db import models

class Friendship(models.Model):
    to_person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="friends")
    from_person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name="from_friends")
```

如果您想在`Person`管理员添加/更改页面上显示内联，则需要明确定义外键，因为它无法自动执行：

```python
from django.contrib import admin
from myapp.models import Friendship

class FriendshipInline(admin.TabularInline):
    model = Friendship
    fk_name = "to_person"

class PersonAdmin(admin.ModelAdmin):
    inlines = [
        FriendshipInline,
    ]
```

### 使用多对多模型

默认情况下，多对多关系的管理窗口小部件将显示在包含[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)的实际引用的任何模型上。 根据您的`ModelAdmin`定义，模型中的每个多对多字段将由标准HTML`<select multiple>`，水平或垂直过滤器或`raw_id_admin`小部件。 但是，也可以用内联替换这些小部件。

假设我们有以下模型：

```python
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=128)

class Group(models.Model):
    name = models.CharField(max_length=128)
    members = models.ManyToManyField(Person, related_name='groups')
```

如果要使用内联显示多对多关系，可以通过为关系定义`InlineModelAdmin`对象来实现：

```python
from django.contrib import admin

class MembershipInline(admin.TabularInline):
    model = Group.members.through

class PersonAdmin(admin.ModelAdmin):
    inlines = [
        MembershipInline,
    ]

class GroupAdmin(admin.ModelAdmin):
    inlines = [
        MembershipInline,
    ]
    exclude = ('members',)
```

在这个例子中有两个值得注意的特征。

首先 - `MembershipInline`类引用`Group.members.through`。 `through`属性是对管理多对多关系的模型的引用。 在定义多对多字段时，此模型由Django自动创建。

其次，`GroupAdmin`必须手动排除`members`字段。 Django在定义关系（在这种情况下，`Group`）的模型上显示多对多字段的管理窗口小部件。 如果要使用内联模型来表示多对多关系，则必须告知Django的管理员*而不是*显示此窗口小部件 - 否则您最终会在管理页面上看到两个窗口小部件，用于管理关系。

请注意，使用此技术时，不会触发[`m2m_changed`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/signals.html#django.db.models.signals.m2m_changed)信号。 这是因为，就管理而言，`through`只是一个具有两个外键字段而不是多对多关系的模型。

在所有其他方面，`InlineModelAdmin`与任何其他方面完全相同。 您可以使用任何正常的`ModelAdmin`属性自定义外观。

### 使用多对多中介模型

当您使用[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)的`through`参数指定中介模型时，admin将不会默认显示窗口小部件。 这是因为该中间模型的每个实例需要比可以在单个小部件中显示的更多的信息，并且多个小部件所需的布局将根据中间模型而变化。

但是，我们仍然希望能够在内联里编辑该信息。 幸运的是，这用内联管理模型很容易做到 假设我们有以下模型：

```python
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=128)

class Group(models.Model):
    name = models.CharField(max_length=128)
    members = models.ManyToManyField(Person, through='Membership')

class Membership(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE)
    group = models.ForeignKey(Group, on_delete=models.CASCADE)
    date_joined = models.DateField()
    invite_reason = models.CharField(max_length=64)
```

在admin中显示此中间模型的第一步是为`Membership`模型定义一个内联类：

```python
class MembershipInline(admin.TabularInline):
    model = Membership
    extra = 1
```

此简单示例使用`InlineModelAdmin`模型的默认`Membership`值，并将额外添加表单限制为一个。 这可以使用`InlineModelAdmin`类可用的任何选项进行自定义。

现在为`Person`和`Group`模型创建管理视图：

```python
class PersonAdmin(admin.ModelAdmin):
    inlines = (MembershipInline,)

class GroupAdmin(admin.ModelAdmin):
    inlines = (MembershipInline,)
```

最后，向管理网站注册您的`Person`和`Group`模型：

```python
admin.site.register(Person, PersonAdmin)
admin.site.register(Group, GroupAdmin)
```

现在，您的管理网站已设置为从`Group`或`Person`详细信息页面内联编辑`Membership`对象。

### 使用通用关系作为内联

可以使用内联与一般相关的对象。 假设您有以下模型：

```python
from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey

class Image(models.Model):
    image = models.ImageField(upload_to="images")
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey("content_type", "object_id")

class Product(models.Model):
    name = models.CharField(max_length=100)
```

如果要允许在产品上编辑和创建Image实例，请添加/更改视图，可以使用admin提供的`GenericTabularInline`或`GenericStackedInline`（`GenericInlineModelAdmin`的两个子类）。它们分别为表示内联对象的表单分别执行表格和堆叠的视觉布局，就像它们的非通用对象一样。 他们的行为就像任何其他内联一样。 在此示例应用的`admin.py`中：

```python
from django.contrib import admin
from django.contrib.contenttypes.admin import GenericTabularInline

from myproject.myapp.models import Image, Product

class ImageInline(GenericTabularInline):
    model = Image

class ProductAdmin(admin.ModelAdmin):
    inlines = [
        ImageInline,
    ]

admin.site.register(Product, ProductAdmin)
```

有关更多具体信息，请参阅[contenttypes documentation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/contenttypes.html)。