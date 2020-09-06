# drf模式和客户端库

模式是一个机器可读的文档，描述了可用的API端点，它们的URL以及它们支持的操作。模式可以是自动生成文档的有用工具，也可以用于驱动可以与API交互的动态客户端库。

## 核心API

为了提供架构支持，REST框架使用了[Core API](https://www.coreapi.org/)。核心API是用于描述API的文档规范。它用于提供可用端点的内部表示格式以及API公开的可能交互。它既可以用于服务器端，也可以用于客户端。在服务器端使用时，Core API允许API支持呈现各种模式或超媒体格式。在客户端使用时，Core API允许动态驱动的客户端库可以与任何公开支持的模式或超媒体格式的API进行交互。

## 添加模式

REST框架支持显式定义的模式视图或自动生成的模式。由于我们使用的是视图集和路由器，因此我们可以简单地使用自动模式生成。

您需要安装`coreapi` python包以包含API模式，并使用`pyyaml`将模式呈现为常用的基于YAML的OpenAPI格式。

```
pip install coreapi pyyaml
```

我们现在可以通过在URL配置中包含自动生成的模式视图来为我们的API包含模式。

```python
from rest_framework.schemas import get_schema_view

schema_view = get_schema_view(title='Pastebin API')

urlpatterns = [
    path('schema/', schema_view),
    ...
]
```

如果您在浏览器中访问`/schema/endpoint`，您现在应该看到`corejson`表示可用作选项。

我们还可以通过在`Accept`标头中指定所需的内容类型，从命令行请求架构。

```shell
$ http http://127.0.0.1:8000/schema/ Accept:application/coreapi+json
HTTP/1.0 200 OK
Allow: GET, HEAD, OPTIONS
Content-Type: application/coreapi+json

{
    "_meta": {
        "title": "Pastebin API"
    },
    "_type": "document",
    ...
```

默认输出样式是使用Core JSON编码。还支持其他模式格式，例如Open API（以前称为Swagger）。

## 使用命令行客户端

现在我们的API公开了一个模式端点，我们可以使用动态客户端库与API进行交互。为了演示这一点，让我们使用Core API命令行客户端。

命令行客户端可用作`coreapi-cli`包

```shell
pip install coreapi-cli
```

现在检查它是否在命令行上可用...

```shell
$ coreapi
Usage: coreapi [OPTIONS] COMMAND [ARGS]...

  Command line client for interacting with CoreAPI services.

  Visit https://www.coreapi.org/ for more information.

Options:
  --version  Display the package version number.
  --help     Show this message and exit.

Commands:
...
```

首先，我们将使用命令行客户端加载API模式。

```shell
$ coreapi get http://127.0.0.1:8000/schema/
<Pastebin API "http://127.0.0.1:8000/schema/">
    snippets: {
        highlight(id)
        list()
        read(id)
    }
    users: {
        list()
        read(id)
    }
```

我们尚未进行身份验证，因此现在我们只能看到只读端点，这与我们如何设置API权限一致。

让我们尝试使用命令行客户端列出现有的代码段

```shell
$ coreapi action snippets list
[
    {
        "url": "http://127.0.0.1:8000/snippets/1/",
        "id": 1,
        "highlight": "http://127.0.0.1:8000/snippets/1/highlight/",
        "owner": "lucy",
        "title": "Example",
        "code": "print('hello, world!')",
        "linenos": true,
        "language": "python",
        "style": "friendly"
    },
    ...
```

某些API端点需要命名参数。例如，要获取特定代码段的突出显示HTML，我们需要提供ID。

```shell
$ coreapi action snippets highlight --param id=1
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">

<html>
<head>
  <title>Example</title>
  ...
```

## 认证客户端

如果我们希望能够创建，编辑和删除代码段，我们需要作为有效用户进行身份验证。在这种情况下，我们将只使用基本身份验证。

请务必使用您的实际用户名和密码替换下面的`<username>`和`<password>`。

```shell
$ coreapi credentials add 127.0.0.1 <username>:<password> --auth basic
Added credentials
127.0.0.1 "Basic <...>"
```

现在，如果我们再次获取模式，我们应该能够看到完整的可用交互集。

```shell
$ coreapi reload
Pastebin API "http://127.0.0.1:8000/schema/">
    snippets: {
        create(code, [title], [linenos], [language], [style])
        delete(id)
        highlight(id)
        list()
        partial_update(id, [title], [code], [linenos], [language], [style])
        read(id)
        update(id, code, [title], [linenos], [language], [style])
    }
    users: {
        list()
        read(id)
    }
```

我们现在能够与这些端点进行交互。例如，要创建新代码段

```shell
$ coreapi action snippets create --param title="Example" --param code="print('hello, world')"
{
    "url": "http://127.0.0.1:8000/snippets/7/",
    "id": 7,
    "highlight": "http://127.0.0.1:8000/snippets/7/highlight/",
    "owner": "lucy",
    "title": "Example",
    "code": "print('hello, world')",
    "linenos": false,
    "language": "python",
    "style": "friendly"
}
```

并删除一个片段

```shell
$ coreapi action snippets delete --param id=7
```

除了命令行客户端，开发人员还可以使用客户端库与您的API进行交互。Python客户端库是第一个可用的，并且计划很快发布Javascript客户端库。

有关自定义模式生成和使用Core API客户端库的更多详细信息，您需要参考完整的文档。



