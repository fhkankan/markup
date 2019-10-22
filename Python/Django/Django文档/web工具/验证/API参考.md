# `django.contrib.auth`

这份文档提供Django 认证系统组件的API 参考资料。 对于这些组件的用法以及如何自定义认证和授权请参照[authentication topic guide](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/index.html)。



## `User`模型



### 字段

- *class* `models.``User`

  [`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 对象具有如下字段：`username`必选。 150个字符以内。 用户名可能包含字母数字，`_`，`@`，`+` `.` 和`-`个字符。对于许多用例，`max_length`应该是足够的。 如果您需要较长的长度，请使用[custom user model](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#specifying-custom-user-model)。 如果您使用具有`utf8mb4`编码（推荐用于正确的Unicode支持）的MySQL，请至少指定`max_length=191`，因为MySQL只能创建具有191个字符的唯一索引，默认。用户名和UnicodeDjango最初只接受用户名中的ASCII字母和数字。 虽然这不是一个故意的选择，Unicode字符一直被接受使用Python 3时。 Django 1.10在用户名中正式添加了Unicode支持，在Python 2中保留了仅适用于ASCII的行为，可以使用[`User.username_validator`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.username_validator)自定义行为的选项。**在Django更改1.10：**`max_length`从30个字符增加到150个字符。`first_name`可选（[`blank=True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.Field.blank)）。 少于等于30个字符。`last_name`可选（[`blank=True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.Field.blank)）。 少于等于30个字符。`email`可选（[`blank=True`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.Field.blank)）。 邮箱地址。`password`必选。 密码的哈希及元数据。 （Django 不保存原始密码）。 原始密码可以无限长而且可以包含任意字符。 参见[password documentation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/passwords.html)。`groups`与[`Group`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.Group) 之间的多对多关系。`user_permissions`与[`Permission`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.Permission) 之间的多对多关系。`is_staff`布尔值。 指示用户是否可以访问Admin 站点。`is_active`布尔值。 指示用户的账号是否激活。 我们建议您将此标志设置为`False`而不是删除帐户；这样，如果您的应用程序对用户有任何外键，则外键不会中断。它不是用来控制用户是否能够登录。 不需要验证后端来检查`is_active`标志，而是默认后端（[`ModelBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend)）和[`RemoteUserBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.RemoteUserBackend)。 如果要允许非活动用户登录，您可以使用[`AllowAllUsersModelBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.AllowAllUsersModelBackend)或[`AllowAllUsersRemoteUserBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.AllowAllUsersRemoteUserBackend)。 在这种情况下，您还需要自定义[`LoginView`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.views.LoginView)使用的[`AuthenticationForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.forms.AuthenticationForm)，因为它拒绝了非活动用户。 请注意，诸如[`has_perm()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.has_perm)等权限检查方法，Django管理员中的身份验证全部返回为非活动用户的`False`。**在Django更改1.10：**在旧版本中，[`ModelBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend)和[`RemoteUserBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.RemoteUserBackend)允许非活动用户进行身份验证。`is_superuser`布尔值。 指定这个用户拥有所有的权限而不需要给他们分配明确的权限。`last_login`用户最后一次登录的时间。`date_joined`账户创建的时间。 当账号创建时，默认设置为当前的date/time。



### 属性

- *class* `models.``User`

  `is_authenticated`始终为`True`（与`AnonymousUser.is_authenticated`相对，始终为`False`）的只读属性。 这是区分用户是否已经认证的一种方法。 这并不表示任何权限，也不会检查用户是否处于活动状态或是否具有有效的会话。 即使正常情况下，您将在`request.user`上检查此属性，以了解它是否已由[`AuthenticationMiddleware`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/middleware.html#django.contrib.auth.middleware.AuthenticationMiddleware)填充（表示当前登录的用户），您应该知道对于任何[`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User)实例，此属性为`True`。**在Django更改1.10：**在旧版本中，这是一种方法。 使用它作为方法的向后兼容性支持将在Django 2.0中被删除。不要使用`is`操作符进行比较！为了允许`is_authenticated`和`is_anonymous`属性也可以作为方法，属性是`CallableBool`对象。 因此，直到Django 2.0中的弃用期结束为止，您不能使用`is`运算符来比较这些属性。 也就是说，`request.user.is_authenticated is True`总是求值得`False`。`is_anonymous`始终为`False`的只读属性。 这是区别[`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 和[`AnonymousUser`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.AnonymousUser) 对象的一种方法。 一般来说，您应该优先使用[`is_authenticated`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_authenticated)到此属性。**在Django更改1.10：**在旧版本中，这是一种方法。 使用它作为方法的向后兼容性支持将在Django 2.0中被删除。`username_validator`**Django中的新功能1.10。**指向用于验证用户名的验证器实例。 Python 3上的默认值为[`validators.UnicodeUsernameValidator`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.validators.UnicodeUsernameValidator)和Python 3上的[`validators.ASCIIUsernameValidator`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.validators.ASCIIUsernameValidator)。要更改默认用户名验证器，可以将`User`模型子类化，并将此属性设置为不同的验证器实例。 例如，要在Python 3上使用ASCII用户名：`from django.contrib.auth.models import 用户 from django.contrib.auth.validators import ASCIIUsernameValidator class CustomUser(User):    username_validator = ASCIIUsernameValidator()     class Meta:        proxy = True  # If no new field is added. `



### 方法

- *class* `models.``User`

  `get_username`()返回这个User 的username。 由于可以将`User`模型交换出来，您应该使用此方法，而不是直接引用用户名属性。`get_full_name`()返回[`first_name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.first_name) 和[`last_name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.last_name)，之间带有一个空格。`get_short_name`()返回[`first_name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.first_name)。`set_password`(*raw_password*)设置用户的密码为给定的原始字符串，并负责密码的哈希。 不会保存[`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 对象。当`None` 为`raw_password` 时，密码将设置为一个不可用的密码，和使用[`set_unusable_password()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.set_unusable_password) 的效果一样。`check_password`(*raw_password*)Returns `True` if the given raw string is the correct password for the user. （它负责在比较时密码的哈希）。`set_unusable_password`()标记用户为没有设置密码。 它与密码为空的字符串不一样。 [`check_password()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.check_password) 对这种用户永远不会返回`True`。 不会保存[`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 对象。如果你的认证发生在外部例如LDAP 目录时，可能需要这个函数。`has_usable_password`()如果对这个用户调用过[`set_unusable_password()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.set_unusable_password)，则返回`False`。`get_group_permissions`(*obj=None*)返回一个用户当前拥有的权限的set，通过用户组如果传入`obj`，则仅返回此特定对象的组权限。http://python.usyiyi.cn/translate/django_182/ref/contrib/auth.html#`get_all_permissions`(*obj=None*)通过组和用户权限返回用户拥有的一组权限字符串。如果传入`obj`，则仅返回此特定对象的权限。`has_perm`(*perm*, *obj=None*)如果用户具有指定的权限，则返回`True`，其中perm的格式为`"."`。 （请参阅有关[permissions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#topic-authorization)）。 如果用户没有激活，这个方法将永远返回 `False`。如果传入`obj`，此方法将不会检查模型的权限，而是检查此特定对象。`has_perms`(*perm_list*, *obj=None*)Returns `True` if the user has each of the specified permissions, where each perm is in the format `"."`. 如果用户没有激活，这个方法将永远返回 `False`。如果传入`obj`，此方法将不会检查模型的权限，而是检查特定对象。`has_module_perms`(*package_name*)如果用户具有给出的package_name（Django应用的标签）中的任何一个权限，则返回`True`。 如果用户没有激活，这个方法将永远返回`False`。`email_user`(*subject*, *message*, *from_email=None*, ***kwargs*)发生邮件给这个用户。 如果`None` 为`from_email`，Django 将使用[`DEFAULT_FROM_EMAIL`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DEFAULT_FROM_EMAIL)。 任何`**kwargs` 都将传递给底层的[`send_mail()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/email.html#django.core.mail.send_mail) 调用。



### 管理器方法

- *class* `models.``UserManager`

  [`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 模型有一个自定义的管理器，它具有以下辅助方法（除了[`BaseUserManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.BaseUserManager) 提供的方法之外）：`create_user`(*username*, *email=None*, *password=None*, ***extra_fields*)创建、保存并返回一个[`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User)。[`username`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.username) 和[`password`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.password) 设置为给出的值。 [`email`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.email) 的域名部分将自动转换成小写，返回的[`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 对象将设置[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_active) 为`True`。如果没有提供password，将调用 [`set_unusable_password()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.set_unusable_password)。The `extra_fields` keyword arguments are passed through to the [`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User)’s `__init__` method to allow setting arbitrary fields on a [custom user model](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#auth-custom-user).参见[Creating users](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#topics-auth-creating-users) 中的示例用法。`create_superuser`(*username*, *email*, *password*, ***extra_fields*)与[`create_user()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.UserManager.create_user) 相同，但是设置[`is_staff`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_staff) 和[`is_superuser`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_superuser) 为`True`。



## `AnonymousUser`对象

- *class* `models.``AnonymousUser`

  [`django.contrib.auth.models.AnonymousUser`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.AnonymousUser) 类实现了[`django.contrib.auth.models.User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 接口，但具有下面几个不同点：[id](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#automatic-primary-key-fields) 永远为`None`。[`username`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.username) 永远为空字符串。[`get_username()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.get_username) 永远返回空字符串。[`is_anonymous`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_anonymous)是`True`而不是`False`。[`is_authenticated`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_authenticated)是`False`，而不是`True`。[`is_staff`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_staff) 和[`is_superuser`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_superuser) 永远为`False`。[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_active) 永远为 `False`。[`groups`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.groups) 和[`user_permissions`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.user_permissions) 永远为空。[`set_password()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.set_password)、[`check_password()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.check_password)、[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save) 和[`delete()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.delete) 引发[`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError)。

在实际应用中，你自己可能不需要使用[`AnonymousUser`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.AnonymousUser) 对象，它们用于Web 请求，在下节会讲述。



## `Permission`模型

- *class* `models.``Permission`

  



### 字段

[`Permission`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.Permission) 对象有以下字段:

- *class* `models.``Permission`

  `name`必选。 255个字符或者更少. 例如: `'Can vote'`.`content_type`必选。 对`django_content_type`数据库表的引用，其中包含每个已安装模型的记录。`codename`必选。 小于等于100个字符. 例如: `'can_vote'`.



### 方法

[`Permission`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.Permission)对象具有类似任何其他[Django model](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html)的标准数据访问方法。



## `Group`模型

- *class* `models.``Group`

  



### 字段

[`Group`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.Group) 对象有以下字段:

- *class* `models.``Group`

  `name`必选。 80个字符以内。 允许任何字符. 例如: `'Awesome Users'`.`permissions`多对多字段到[`Permission`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.Permission)：`group.permissions.set([permission_list]) group.permissions.add(permission, permission, ...) group.permissions.remove(permission, permission, ...) group.permissions.clear() `



## 验证

- *class* `validators.``ASCIIUsernameValidator`

  **Django中的新功能1.10。**仅允许使用ASCII字母和数字的字段验证器，除`@`之外， `.`，`+`，`-`和`_`。 Python 2上的`User.username`的默认验证器。

- *class* `validators.``UnicodeUsernameValidator`

  **Django中的新功能1.10。**允许Unicode字符的字段验证器，除`@`之外， `.`，`+`，`-`和`_`。 Python 3上的`User.username`的默认验证器。



## 登入和登出信号

auth框架使用以下[signals](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/signals.html)，可用于在用户登录或注销时通知。

- `user_logged_in`()

  当用户成功登录时发送。与此信号一起发送的参数：`sender`刚刚登录的用户的类。`request`当前的[`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest)实例。`user`刚刚登录的用户实例。

- `user_logged_out`()

  在调用logout方法时发送。`sender`如上所述：刚刚注销的用户的类或`None`，如果用户未通过身份验证。`request`当前的[`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest)实例。`user`如果用户未通过身份验证，刚刚注销的用户实例或`None`。

- `user_login_failed`()

  当用户登录失败时发送`sender`用于认证的模块的名称。`credentials`包含传递给[`authenticate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.authenticate)或您自己的自定义身份验证后端的用户凭据的关键字参数的字典。 匹配一组“敏感”模式（包括密码）的凭证不会作为信号的一部分发送到清除中。`request`[`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest)对象，如果提供给[`authenticate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.authenticate)。**在Django更改1.11：**添加了`request`参数。



## 认证后端

这一节详细讲述Django自带的认证后端。 关于如何使用它们以及如何编写你自己的认证后端，参见[用户认证指南](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/index.html)中的[其它认证源一节](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#authentication-backends)。



### 可用的认证后端

以下是[`django.contrib.auth.backends`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#module-django.contrib.auth.backends)中可以使用的后端：

- *class* `ModelBackend`

  这是Django使用的默认认证后台。 它使用由用户标识和密码组成的凭据进行认证。 对于Django的默认用户模型，用户的标识是用户名，对于自定义的用户模型，它通过USERNAME_FIELD 字段表示（参见[Customizing Users and authentication](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html)）。它还处理 [`User`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User) 和[`PermissionsMixin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.PermissionsMixin) 定义的权限模型。[`has_perm()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend.has_perm), [`get_all_permissions()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend.get_all_permissions), [`get_user_permissions()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend.get_user_permissions), 和[`get_group_permissions()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend.get_group_permissions) 允许一个对象作为特定权限参数来传递, 如果条件是 if `obj is not None`. 后端除了返回一个空的permissions 外，并不会去完成他们。`authenticate`(*request*, *username=None*, *password=None*, ***kwargs*)通过调用[`User.check_password`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.check_password) 验证`password` 和`username`。 如果`kwargs` 没有提供，它会使用[`CustomUser.USERNAME_FIELD`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.USERNAME_FIELD) 关键字从`username` 中获取username。 返回一个认证过的User 或`None`。`request` is an [`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest) and may be `None` if it wasn’t provided to [`authenticate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.authenticate) (which passes it on to the backend).**在Django更改1.11：**添加了`request`参数。`get_user_permissions`(*user_obj*, *obj=None*)返回`user_obj`具有的自己用户权限的权限字符串集合。 如果[`is_anonymous`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.AbstractBaseUser.is_anonymous)或[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.is_active)是`False`，则返回空集。`get_group_permissions`(*user_obj*, *obj=None*)返回`user_obj`从其所属组的权限中获取的权限字符集。 如果[`is_anonymous`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.AbstractBaseUser.is_anonymous)或[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.is_active)是`False`，则返回空集。`get_all_permissions`(*user_obj*, *obj=None*)返回`user_obj`的权限字符串集，包括用户权限和组权限。 如果[`is_anonymous`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.AbstractBaseUser.is_anonymous)或[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.is_active)是`False`，则返回空集。`has_perm`(*user_obj*, *perm*, *obj=None*)使用[`get_all_permissions()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend.get_all_permissions)检查`user_obj`是否具有权限字符串`perm`。 如果用户不是[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.is_active)，则返回`False`。`has_module_perms`(*user_obj*, *app_label*)返回`user_obj`是否对应用`app_label`有任何权限。`user_can_authenticate`()**Django中的新功能1.10。**返回是否允许用户进行身份验证。 To match the behavior of [`AuthenticationForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.forms.AuthenticationForm) which [`prohibits inactive users from logging in`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.forms.AuthenticationForm.confirm_login_allowed), this method returns `False` for users with [`is_active=False`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_active). 不允许使用[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.is_active)字段的自定义用户模型。

- *class* `AllowAllUsersModelBackend`

  **Django 1.10中新增。**与[`ModelBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend)相同，但是不会拒绝非激活的用户，因为[`user_can_authenticate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.ModelBackend.user_can_authenticate)始终返回`True`。使用此后端时，你可能会需要覆盖[`confirm_login_allowed()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.forms.AuthenticationForm.confirm_login_allowed)方法来自定义[`LoginView`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.views.LoginView)使用的[`AuthenticationForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.forms.AuthenticationForm)，因为它拒绝了非激活的用户。

- *class* `RemoteUserBackend`

  使用这个后端来处理Django的外部认证。 它使用在[`request.META['REMOTE_USER'\]`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest.META)中传递的用户名进行身份验证。 请参阅[REMOTE_USER的认证](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/howto/auth-remote-user.html)文档。如果你需要更多的控制，你可以创建你自己的验证后端，继承这个类，并重写这些属性或方法：

- `RemoteUserBackend.``create_unknown_user`

  `True`或`False`。 确定是否创建用户对象（如果尚未在数据库中）默认为`True`。

- `RemoteUserBackend.``authenticate`(*request*, *remote_user*)

  作为`remote_user`传递的用户名被认为是可信的。 此方法只需返回具有给定用户名的用户对象，如果[`create_unknown_user`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.RemoteUserBackend.create_unknown_user)为`True`则创建新的用户对象。如果[`create_unknown_user`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.RemoteUserBackend.create_unknown_user)是`User`，并且在数据库中找不到具有给定用户名的`None`对象，则返回`False`。`request` is an [`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest) and may be `None` if it wasn’t provided to [`authenticate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.authenticate) (which passes it on to the backend).

- `RemoteUserBackend.``clean_username`(*username*)

  在使用它获取或创建用户对象之前，请对`username`执行任何清除（例如剥离LDAP DN信息）。 返回已清除的用户名。

- `RemoteUserBackend.``configure_user`(*user*)

  配置新创建的用户。 此方法在创建新用户后立即调用，并可用于执行自定义设置操作，例如根据LDAP目录中的属性设置用户的组。 返回用户对象。

- `RemoteUserBackend.``user_can_authenticate`()

  **Django中的新功能1.10。**返回是否允许用户进行身份验证。 对于[`is_active=False`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_active)的用户，此方法返回`False`。 不允许使用[`is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.CustomUser.is_active)字段的自定义用户模型。

- *class* `AllowAllUsersRemoteUserBackend`

  **Django 1.10中新增。**与[`ModelBackend`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.RemoteUserBackend)相同，但是不会拒绝非激活的用户，因为[`user_can_authenticate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.backends.RemoteUserBackend.user_can_authenticate)始终返回`True`。



## 实用功能

- `get_user`(*request*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/contrib/auth.html#get_user)

  返回与给定的`request`会话关联的用户模型实例。它检查存储在会话中的身份验证后端是否存在于[`AUTHENTICATION_BACKENDS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-AUTHENTICATION_BACKENDS)中。 如果是这样，它使用后端的`get_user()`方法来检索用户模型实例，然后通过调用用户模型的[`get_session_auth_hash()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/customizing.html#django.contrib.auth.models.AbstractBaseUser.get_session_auth_hash)方法验证会话。如果存储在会话中的身份验证后端不再在[`AUTHENTICATION_BACKENDS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-AUTHENTICATION_BACKENDS)中返回[`AnonymousUser`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.AnonymousUser)的实例，如果后端的`get_user()`