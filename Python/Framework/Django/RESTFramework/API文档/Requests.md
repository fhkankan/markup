# Requests

REST框架的`Request`类扩展了标准的`HttpRequest`，增加了对REST框架的灵活请求解析和请求身份验证的支持。

## 请求解析

REST框架的Request对象提供了灵活的请求解析，使您能够以与通常处理表单数据相同的方式来处理JSON数据或其他媒体类型的请求。

### `.data`

`request.data`返回请求正文的解析内容。这类似于标准的`request.POST`和`request.FILES`属性，除了：

- 它包括所有已解析的内容，包括文件和非文件输入。

- 它支持解析`POST`以外的HTTP方法的内容，这意味着您可以访问`PUT`和`PATCH`请求的内容。

- 它支持REST框架的灵活请求解析，而不仅仅是支持表单数据。例如，您可以以处理传入表单数据的相同方式处理传入JSON数据。

有关更多详细信息，请参见[解析器文档](https://www.django-rest-framework.org/api-guide/parsers/).。

### `.query_params` 
`request.query_params`是`request.GET`的更正确命名的同义词。

为了使代码内部更清晰，我们建议使用`request.query_params`而不是Django的标准`request.GET`。这样做将有助于使您的代码库更加正确和明显-任何HTTP方法类型都可以包括查询参数，而不仅仅是`GET`请求。

### `.parsers`

`APIView`类或`@api_view`装饰器将确保根据在视图上设置的`parser_classes`或基于`DEFAULT_PARSER_CLASSES`设置，将此属性自动设置为`Parser`实例的列表。
您通常不需要访问此属性。

>注意：
>
>如果客户端发送格式错误的内容，则访问`request.data`可能会引发`ParseError`。默认情况下，REST框架的`APIView`类或`@api_view`装饰器将捕获该错误并返回`400 Bad Request`响应。
>
>如果客户端发送的请求的内容类型无法解析，则将引发`UnsupportedMediaType`异常，默认情况下将捕获该异常并返回`415 Unsupported Media Type`响应。

## 内容协商

请求提供了一些属性允许你确定内容协商阶段的结果。这允许你实现具体的行为，例如为不同的媒体类型选择不用的序列化方案。

### `.accepted_renderer`

由内容协商阶段选择的render实例。

### `.accepted_media_type`

由内容协商阶段接受的媒体类型的字符串。

## 身份验证

REST framework 提供了灵活的，每次请求的验证，让你能够：

- 对API的不同部分使用不同的身份验证策略。
- 支持使用多个身份验证策略。 
- 提供与传入请求相关联的用户和令牌信息。

### `.user`

`request.user` 通常返回一个 `django.contrib.auth.models.User` 实例, 尽管该行为取决于所使用的的认证策略。

如果请求未认证则 `request.user` 的默认值为 `django.contrib.auth.models.AnonymousUser`的一个实例。

更多详细信息请查阅 [authentication documentation](https://q1mi.github.io/Django-REST-framework-documentation/api-guide/requests_zh/authentication.md).

### `.auth`

`request.auth` 返回任何其他身份验证上下文。 `request.auth` 的确切行为取决于所使用的的认证策略，但它通常可以是请求被认证的token的实例。

如果请求未认证或者没有其他上下文，则 `request.auth` 的默认值为 `None`.

更多详细信息请查阅 [authentication documentation](https://q1mi.github.io/Django-REST-framework-documentation/api-guide/requests_zh/authentication.md).

### `.authenticators`

`APIView` 类或 `@api_view` 装饰器将根据在view中设置的 `authentication_classes` 或基于`DEFAULT_AUTHENTICATORS` 设置，确保此属性自动设置为 `Authentication` 实例的列表。

你通常并不需要访问此属性。

> 注意
>
> 调用`.user`或`.auth`属性时，您可能会看到抛出`WrappedAttributeError`错误。这些错误源自作为标准`AttributeError`的身份验证器，但是有必要将它们重新引发为其他异常类型，以防止外部属性访问抑制它们。Python将不会识别`AttributeError`源自身份验证器，而是会假设请求对象没有`.user`或`.auth`属性。身份验证器将需要固定。

## 浏览器增强

REST framework 支持一些浏览器增强功能，例如基于浏览器的 `PUT`, `PATCH` 和 `DELETE` 表单。

### `.method`

`request.method` 返回请求的HTTP方法的 **大写** 字符串表示形式。

透明地支持基于浏览器的 `PUT`, `PATCH` 和 `DELETE` 表单。

更多详细信息请查阅 [browser enhancements documentation](https://q1mi.github.io/Django-REST-framework-documentation/topics/browser-enhancements/).

### `.content_type`

`request.content_type` 返回表示HTTP请求正文的媒体类型的字符串对象，如果未提供媒体类型，则返回空字符串。

你通常不需要直接访问请求的内容类型，因为你通常将依赖于REST framework的默认请求解析行为。

如果你确实需要访问请求的内容类型，你应该使用 `.content_type` 属性，而不是使用 `request.META.get('HTTP_CONTENT_TYPE')`, 因为它为基于浏览器的非表单内容提供了透明的支持。

更多详细信息请查阅 [browser enhancements documentation](https://q1mi.github.io/Django-REST-framework-documentation/topics/browser-enhancements/).

### `.stream`

`request.stream` 返回一个表示请求主体内容的流。

你通常不需要直接访问请求的内容类型，因为你通常将依赖于REST framework的默认请求解析行为。

## 标准`HttpRequest`属性

由于 REST framework 的 `Request` 扩展了 Django的 `HttpRequest`, 所以所有其他标准属性和方法也是可用的。例如 `request.META` 和 `request.session` 字典正常可用。

请注意，由于实现原因， `Request` 类并不会从 `HttpRequest` 类继承, 而是使用合成扩展类。