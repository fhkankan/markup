# 格式本地化



## 概述

Django的格式化系统可以在模板中使用当前[locale](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/index.html#term-locale-name)特定的格式，来展示日期、时间和数字。 它还可以处理表单中的本地化输入。

当它被开启时，访问相同内容的两个用户可能会看到以不同方式格式化的日期、时间和数字，这取决于它们的当前地区的格式。

格式化系统默认是禁用的。 需要在你的设置文件中设置[`USE_L10N = True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_L10N)来启用它。

注

为了方便，[`django-admin startproject`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-startproject) 创建的`settings.py` 文件中，[`USE_L10N = True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_L10N)。 但是要注意，要开启千位分隔符的数字格式化，你需要在你的设置文件中设置[`USE_THOUSAND_SEPARATOR = True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_THOUSAND_SEPARATOR)。 或者，你也可以在你的模板中使用[`intcomma`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/humanize.html#std:templatefilter-intcomma)来格式化数字。

注

[`USE_I18N`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_I18N) 是另一个独立的并且相关的设置，它控制着Django是否应该开启翻译。 详见[Translation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/translation.html)。



## 表单中的区域设置感知输入

格式化开启之后，Django可以在表单中使用本地化格式来解析日期、时间和数字。 也就是说，在表单上输入时，它会尝试不同的格式和地区来猜测用户使用的格式。

注

Django对于展示数据，使用和解析数据不同的格式。 尤其是，解析日期的格式不能使用`%p`（星期名称的缩写），`%b` （星期名称的全称），`%A` （月份名称的缩写）， `%B`（月份名称的全称），或者`%a`（上午/下午）。

只是使用`localize`参数，就能开启表单字段的本地化输入和输出：

```
class CashRegisterForm(forms.Form):
   product = forms.CharField()
   revenue = forms.DecimalField(max_digits=4, decimal_places=2, localize=True)
```



## 控制模板中的定位

当你使用[`USE_L10N`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_L10N)来开启格式化的时候，Django会尝试使用地区特定的格式，无论值在模板的什么位置输出。

然而，这对于本地化的值不可能总是十分合适，如果你在输出JavaScript或者机器阅读的XML，你会想要使用去本地化的值。 你也可能想只在特定的模板中使用本地化，而不是任何位置都使用。

DJango提供了`l10n`模板库，包含以下标签和过滤器，来实现对本地化的精细控制。



### 模板标签



#### `localize`

在包含的代码块内开启或关闭模板变量的本地化。

这个标签可以对本地化进行比[`USE_L10N`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_L10N)更加精细的操作。

这样做来为一个模板激活或禁用本地化：

```
{% load l10n %}

{% localize on %}
    {{ value }}
{% endlocalize %}

{% localize off %}
    {{ value }}
{% endlocalize %}
```

注

在 `{% localize %}`代码块内并不遵循 [`USE_L10N`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_L10N)的值。

对于在每个变量基础上执行相同工作的模板过滤器，参见[`localize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html#std:templatefilter-localize) 和 [`unlocalize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html#std:templatefilter-unlocalize)。



### 模板过滤器



#### `localize`

强制单一值的本地化。

像这样：

```
{% load l10n %}

{{ value|localize }}
```

使用[`unlocalize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html#std:templatefilter-unlocalize)来在单一值上禁用本地化。 使用[`localize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html#std:templatetag-localize) 模板标签来在大块的模板区域内控制本地化。



#### `unlocalize`

强制单一值不带本地化输出。

像这样：

```
{% load l10n %}

{{ value|unlocalize }}
```

使用[`localize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html#std:templatefilter-localize)来强制单一值的本地化。 使用[`localize`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/formatting.html#std:templatetag-localize) 模板标签来在大块的模板区域内控制本地化。



## 创建自定义格式文件

Django为许多地区提供了格式定义，但是有时你可能想要创建你自己的格式，因为你的的确并没有现成的格式文件，或者你想要覆写其中的一些值。

指定你首先放置格式文件的位置来使用自定义格式。 把你的[`FORMAT_MODULE_PATH`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-FORMAT_MODULE_PATH)设置设置为格式文件存在的包名来使用它，例如：

```
FORMAT_MODULE_PATH = [
    'mysite.formats',
    'some_app.formats',
]
```

文件并不直接放在这个目录中，而是放在和地区名称相同的目录中，文件也必须名为`formats.py`。 小心不要将敏感信息放在这些文件中，因为如果将字符串传递给`django.utils.formats.get_format()`（由[`date`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/templates/builtins.html#std:templatefilter-date)使用）模板过滤器）。

需要这样一个结构来自定义英文格式：

```
mysite/
    formats/
        __init__.py
        en/
            __init__.py
            formats.py
```

其中`formats.py`包含自定义的格式定义。 像这样：

```
from __future__ import unicode_literals

THOUSAND_SEPARATOR = '\xa0'
```

使用非间断空格(Unicode `00A0`)作为千位分隔符，来代替英语中默认的逗号。



## 提供的区域设置格式的限制

一些地区对数字使用上下文敏感的格式，Django的本地化系统不能自动处理它。



### 瑞士（德语）

瑞士的数字格式化取决于被格式化的数字类型。 对于货币值，使用逗号作为千位分隔符，以及使用小数点作为十进制分隔符。 对于其它数字，逗号用于十进制分隔符，空格用于千位分隔符。 Django提供的本地格式使用通用的分隔符，即逗号用于十进制分隔符，空格用于千位分隔符。