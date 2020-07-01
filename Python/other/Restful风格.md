

# Restful

REST:Representational State Transfer的缩写，翻译：“具象状态传输”。一般解释为“表现层状态转换”。

REST是设计风格而不是标准。是指客户端和服务器的交互形式。我们需要关注的重点是如何设计REST风格的网络接口。

## 特点

- 具象的。一般指表现层，要表现的对象就是资源。比如，客户端访问服务器，获取的数据就是资源。比如文字、图片、音视频等。
- 表现：资源的表现形式。txt格式、html格式、json格式、jpg格式等。浏览器通过URL确定资源的位置，但是需要在HTTP请求头中，用Accept和Content-Type字段指定，这两个字段是对资源表现的描述。
- 状态转换：客户端和服务器交互的过程。在这个过程中，一定会有数据和状态的转化，这种转化叫做状态转换。其中，GET表示获取资源，POST表示新建资源，PUT表示更新资源，DELETE表示删除资源。HTTP协议中最常用的就是这四种操作方式。
  - RESTful架构：
  - 每个URL代表一种资源；
  - 客户端和服务器之间，传递这种资源的某种表现层；
  - 客户端通过四个http动词，对服务器资源进行操作，实现表现层状态转换。

## 域名

将api部署在专用域名下：

```
http://api.example.com
```

或者将api放在主域名下：

```
http://www.example.com/api/
```

## 版本

将API的版本号放在url中。

```
http://www.example.com/app/1.0/info
http://www.example.com/app/1.2/info
```

## 路径

路径表示API的具体网址。每个网址代表一种资源。 

- 名词和动词

资源作为网址，网址中不能有动词只能有名词

使用动作动词来以RPC方式公开服务

```
getUser(1234) 
createUser(user) 
deleteAddress(1234)
```

RESTful风格，则建议为

```
GET /users/1234
POST /users
DELETE /addressed/1235
```
- 单数和复数

虽然一些”语法学家”会告诉你使用复数来描述资源的单个实例是错误的，但实际上为了保持URI格式的一致性建议使用复数形式。

```
# 获取单个商品
http://www.example.com/app/goods/1
# 获取所有商品
http://www.example.com/app/goods
```

本着API提供商更容易实施和API使用者更容易操作的原则，可以不必纠结一些奇怪的复数`person/people，goose/geese`。

但是应该怎么处理层级关系呢？如果一个关系只能存在于另一个资源中，RESTful原则就会提供有用的指导。我们来看一下这个例子。学生有一些课程。这些课程在逻辑上映射到学生终端，如下所示：

```
#  检索id为3248234的学生学习的所有课程的清单。
http://api.college.com/students/3248234/courses
#  检索该学生的物理课程
http://api.college.com/students/3248234/courses/physics 
```

- URI结尾不应包含`(/)`

正斜杠`(/)`不会增加语义值，且可能导致混淆。REST API不允许一个尾部的斜杠，不应该将它们包含在提供给客户端的链接的结尾处。
许多Web组件和框架将平等对待以下两个URI：

```
http://api.canvas.com/shapes/
http://api.canvas.com/shapes
```

但是，实际上URI中的每个字符都会计入资源的唯一身份的识别中。

两个不同的URI映射到两个不同的资源。如果URI不同，那么资源也是如此，反之亦然。因此，REST API必须生成和传递精确的URI，不能容忍任何的客户端尝试不精确的资源定位。

有些API碰到这种情况，可能设计为让客户端重定向到相应没有尾斜杠的URI（也有可能会返回301 - 用来资源重定向）。

- 正斜杠分隔符（/）必须用来指示层级关系

```
http://api.canvas.com/shapes/polygons/quadrilaterals/squares 
```

- 应使用连字符来提高URI的可读性

为了使您的URI容易让人们理解，请使用连字符(`-`)字符来提高长路径中名称的可读性。在路径中，应该使用连字符代空格连接两个单词 。
```
http://api.example.com/blogs/guy-levin/posts/this-is-my-first-post
```

- 不得在URI中使用下划线

一些文本查看器为了区分强调URI，常常会在URI下加上下划线。这样下划线`(_)`字符可能被文本查看器中默认的下划线部分地遮蔽或完全隐藏。

为避免这种混淆，请使用连字符`(-)`而不是下划线

- URI路径中首选小写字母

URI路径中首选小写字母，因为大写字母有时会导致一些问题。RFC 3986将URI定义为区分大小写，但scheme 和 host components除外。

```python
http://api.example.com/my-folder/my-doc

HTTP://API.EXAMPLE.COM/my-folder/my-doc
# 这个URI很好。URI格式规范（RFC 3986）认为该URI与URI＃1相同。

http://api.example.com/My-Folder/my-doc
# 而这个URI与URI 1和2不同，这可能会导致不必要的混淆。
```
- 文件扩展名不应包含在URI中

在Web上，`(.)`字符通常用于分隔URI的文件名和扩展名。
REST API不应在URI中包含人造文件扩展名，来指示邮件实体的格式。相反，他们应该依赖通过Content-Type中的header传递media type，来确定如何处理正文的内容。
```
http://api.college.com/students/3248234/courses/2005/fall.json
http://api.college.com/students/3248234/courses/2005/fall
```
如上所示：不应使用文件扩展名来表示格式。

应鼓励REST API客户端使用HTTP提供的格式选择机制Accept request header。

为了是链接和调试更简单，REST API应该支持通过查询参数来支持媒体类型的选择。

## 使用标准HTTP方法

对于资源的具体操作类型，由HTTP动词表示。 常用的HTTP动词有四个。

```
GET     SELECT ：从服务器获取资源。
POST    CREATE ：在服务器新建资源。
PUT     UPDATE ：在服务器更新资源。
DELETE  DELETE ：从服务器删除资源。
```

不常用HTTP动词

```
PATCH(UPDATE):在服务器更新(更新)资源(客户端提供改变的属性)。 
HEAD:获取资源的元数据。
OPTIONS:获取信息，关于资源的哪些属性是客户端可以改变的。
```

示例：

```python
# 获取商品列表
GET http://www.example.com/goods
# 获取指定商品的信息
GET http://www.example.com/goods/ID

# 新建商品的信息
POST http://www.example.com/goods

# 批量修改商品信息
PUT http://www.example.com/goods
# 更新指定商品的信息
PUT http://www.example.com/goods/ID
# 修改成员部分属性
PATCH http://www.example.com/goods/ID  
  
#删除指定商品的信息
DELETE http://www.example.com/goods/ID
```

## 过滤信息

如果资源数据较多，服务器不能将所有数据一次全部返回给客户端。API应该提供参数，过滤返回结果。 实例：

```
#指定返回数据的数量
http://www.example.com/goods?limit=10
#指定返回数据的开始位置
http://www.example.com/goods?offset=10
#指定第几页，以及每页数据的数量
http://www.example.com/goods?page=2&per_page=20
```

参数的设计允许存在冗余，即允许API路径和URL参数偶尔有重复。比如，`GET /zoos/ID/animals` 与` GET /animals?zoo_id=ID` 的含义是相同的。 

## 状态码

服务器向用户返回的状态码和提示信息，常用的有：

```
200 OK  ：服务器成功返回用户请求的数据
201 CREATED ：用户新建或修改数据成功。
202 Accepted：表示请求已进入后台排队。
400 INVALID REQUEST ：用户发出的请求有错误。
401 Unauthorized ：用户没有权限。
403 Forbidden ：访问被禁止。
404 NOT FOUND ：请求针对的是不存在的记录。
406 Not Acceptable ：用户请求的的格式不正确。
500 INTERNAL SERVER ERROR ：服务器发生错误。
```

## 错误信息

一般来说，服务器返回的错误信息，以键值对的形式返回。

```
{
    error:'Invalid API KEY'
}
```

## 响应结果

针对不同结果，服务器向客户端返回的结果应符合以下规范。

```
#返回商品列表
GET    http://www.example.com/goods
#返回单个商品
GET    http://www.example.com/goods/cup
#返回新生成的商品
POST   http://www.example.com/goods
#返回一个空文档
DELETE http://www.example.com/goods
```

## 超媒体

RESTful API最好做到Hypermedia(即返回结果中提供链接，连向其他API方法)，使得用户不查文 

档，也知道下一步应该做什么。 比如，Github的API就是这种设计，访问api.github.com会得到一个所有可用API的网址列表。 

```
{
	"current_user_url":"https://api.github.com/user",
	"authorization_url":"https://api.github.com/authorizations",
	// ...
}
```

从上面可以看到，如果想获取当前用户的信息，应该去访问api.github.com/user，然后就得到了下面结 果。 

```
{
	"message": "Requires authentication",
	"documentation_url": "https://developer.github.com/v3"
}
```

在返回响应结果时提供链接其他API的方法，使客户端很方便的获取相关联的信息。

## 其他

服务器返回的数据格式，应该尽量使用JSON，避免使用XML。