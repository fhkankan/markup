# ApiDoc

[官网](https://apidocjs.com)

## 概述

apidoc是一款可以有源代码中的注释直接自动生成api接口文档的工具，它几乎支持目前主流的所有风格的注释。可在C#,python, Go, Dart, Java, JavaScript, PHP, TypeScript等语言中使用)

## 安装

```
1. 安装node.js
brew install node

2. 安装apiDoc
npm install apidoc -g
```

## 运行

示例

```
apidoc -i ../cms/apps/ -o ../cms/static/apidoc/
```

说明

```
apidoc -i myapp/ -o apidoc/ -t mytemplate/
```

命令参数

|           参数           |                                                                                                                                     描述                                                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-h, —help`             | 查看帮助文档                                                                                                                                                                                                                                                                  |
| `-f, --file-filters`    | 指定读取文件的文件名过滤正则表达式(可指定多个) <br/>例如: `apidoc -f ".*\\.js$" -f ".*\\.ts$"` 意为只读取后缀名为js和ts的文件 默认值:`.clj .cls .coffee .cpp .cs .dart .erl .exs?`  `.go .groovy .ino? .java .js .jsx .kt .litcoffee lua .p .php? .pl .pm .py .rb .scala .ts .vue` |
| `-e, --exclude-filters` | 指定不读取的文件名过滤正则表达式(可指定多个)<br/>例如:`apidoc -e ".*\\.js$"` 意为不读取后缀名为js的文件默认:`''`                                                                                                                                                                      |
| `-i, --input `          | 指定读取源文件的目录<br/>例如：`apidoc -i myapp/` 意为读取`myapp/`目录下面的源文件,默认值:`./`                                                                                                                                                                                      |
| `-o, --output`          | 指定输出文档的目录<br/> 例如：`apidoc -o doc/` 意为输出文档到`doc`目录下, 默认值:`./doc/`                                                                                                                                                                                          |
| `-t, --template`        | 指定输出的模板文件<br/>例如:`apidoc -t mytemplate/`, 默认:`path.join(__dirname, '../template/')(使用默认模板)`                                                                                                                                                                  |
| `-c, --config`          | 指定包含配置文件(apidoc.json)的目录<br/>例如:`apidoc -c config/`, 默认:`./`                                                                                                                                                                                                     |
| `-p, --private`         | 输出的文档中是否包含私有api<br/>例如:`apidoc -p true`, 默认:`false`                                                                                                                                                                                                             |
| `-v, --verbose`         | 是否输出详细的debug信息<br/>例如:`apidoc -v true`, 默认:`false`                                                                                                                                                                                                                |

## 配置

每次导出接口文档都必须要让apidoc读取到`apidoc.json`文件(如果未添加配置文件，导出报错)，你可以在你项目的根目录下添加`apidoc.json`文件，这个文件主要包含一些项目的描述信息，比如标题、简短的描述、版本等，你也可以加入一些可选的配置项，比如页眉、页脚、模板等

apidoc.json

```json
{
  "name": "example",
  "version": "0.1.0",
  "description": "apiDoc basic example",
  "title": "Custom apiDoc browser title",
  "url" : "https://api.github.com/v1"
}
```

如果你的项目中使用了`package.json`文件(例如:node.js工程)，那么你可以将`apidoc.json`文件中的所有配置信息放到`package.json`文件中的*apidoc*参数中

package.json

```json
{
  "name": "example",
  "version": "0.1.0",
  "description": "apiDoc basic example",
  "apidoc": {
    "title": "Custom apiDoc browser title",
    "url" : "https://api.github.com/v1"
  }
}
```

**apidoc.json配置项**

|      参数       | 描述                                                         |
| :-------------: | ------------------------------------------------------------ |
|      name       | 工程名称 如果`apidoc.json`文件中没有配置该参数，`apidoc`会尝试从`pakcage.json`文件中读取 |
|     version     | 版本 如果`apidoc.json`文件中没有配置该参数，`apidoc`会尝试从`pakcage.json`文件中读取 |
|   description   | 工程描述 如果`apidoc.json`文件中没有配置该参数，`apidoc`会尝试从`pakcage.json`文件中读取 |
|      title      | 浏览器标题                                                   |
|       url       | api路径前缀 例如:`https://api.github.com/v1`                 |
|    sampleUrl    | 如果设置了该参数，那么在文档中便可以看到用于测试接口的一个表单(详情可以查看参数@apiSampleReques) |
|  header.title   | 页眉导航标题                                                 |
| header.filename | 页眉文件名(markdown)                                         |
|  footer.title   | 页脚导航标题                                                 |
| footer.filename | 页脚文件名(markdown)                                         |
|      order      | 接口名称或接口组名称的排序列表 如果未定义，那么所有名称会自动排序 "order":[        "Error",       "Define",      "PostTitleAndError",      "PostError" ] |

## 使用

### 样例

```javascript
/**
 *
 * @apiDefine RkNotFoundException
 *
 * @apiError RkNotFoundException 找不到相关数据
 *
 * @apiErrorExample Error-Response:
 *     HTTP/1.1 404 Not Found
 *     {
 *       "error": {
 *           "code": 404,
 *           "msg": "",
 *           "path" ""
 *       }
 *     }
 *
 */

/**
 *
 * @api {get} /v3.1/ues/:sn/rt-info 获取设备上报实时信息
 * @apiVersion 3.1.0
 * @apiName GetUeRealTimeInfo
 * @apiGroup UE
 *
 * @apiHeader {String} Authorization 用户授权token
 * @apiHeader {String} firm 厂商编码
 * @apiHeaderExample {json} Header-Example:
 *     {
 *       "Authorization": "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOjM2NzgsImF1ZGllbmNlIjoid2ViIiwib3BlbkFJZCI6MTM2NywiY3JlYXRlZCI6MTUzMzg3OTM2ODA0Nywicm9sZXMiOiJVU0VSIiwiZXhwIjoxNTM0NDg0MTY4fQ.Gl5L-NpuwhjuPXFuhPax8ak5c64skjDTCBC64N_QdKQ2VT-zZeceuzXB9TqaYJuhkwNYEhrV3pUx1zhMWG7Org",
 *       "firm": "cnE="
 *     }
 *
 * @apiParam {String} sn 设备序列号
 *
 * @apiSuccess {String} sn 设备序列号
 * @apiSuccess {Number} status 设备状态
 * @apiSuccess {Number} soc 电池电量百分比
 * @apiSuccess {Number} voltage 电池电压
 * @apiSuccess {Number} current 电池电流
 * @apiSuccess {Number} temperature 电池温度
 * @apiSuccess {String} reportTime 上报时间(yyyy-MM-dd HH:mm:ss)
 *
 * @apiSuccessExample Success-Response:
 *     HTTP/1.1 200 OK
 *     {
 *       "sn": "P000000000",
 *       "status": 0,
 *       "soc": 80,
 *       "voltage": 60.0,
 *       "current": 10.0,
 *       "temperature": null,
 *       "reportTime": "2018-08-13 18:11:00"
 *     }
 *
 * @apiUse RkNotFoundException
 *
 */
@RequestMapping(value = "/{sn}/rt-info", method = RequestMethod.GET)
public UeRealTimeInfo getUeRealTimeInfo(@RequestHeader(Constants.HEADER_LOGIN_USER_KEY) long userId, @PathVariable("sn") String sn) {

    return ueService.getRealTimeInfo(sn);
}
```

## 参数

### @api

【必填字段】否则，`apidoc`会忽略该条注释

```
@api {method} path [title]
```

参数列表:

|  参数  | 必填 | 描述                                                         |
| :----: | ---- | ------------------------------------------------------------ |
| method | yes  | 请求类型:DELETE, GET, POST, PUT, ...[更多](https://link.jianshu.com?t=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FHypertext_Transfer_Protocol%23Request_methods) |
|  path  | yes  | 请求路径                                                     |
| title  | no   | 接口标题                                                     |

例：

```
/**
 * @api {get} /user/:id
 */
```

### @apiDefine

```
@apiDefine name [title]
                     [description]
```

定义注释模块(类似于代码中定义一个常量)，对于一些通用可复用的注释模块(例如:接口错误响应模块),只需要在源代码中定义一次，便可以在其他注释模块中随便引用，最后在文档导出时会自动替换所引用的注释模块，定义之后您可以通过`@apiUse`来引入所定义的注释模块。(注:可以同时使用`@apiVersion`来定义注释模块的版本)
 参数列表：

|    参数     | 必填 | 描述                                                         |
| :---------: | ---- | ------------------------------------------------------------ |
|    name     | yes  | 注释模块名称(唯一)，不同`@apiVersion`可以定义相同名称的注释模块 |
|    title    | no   | 注释模块标题                                                 |
| description | no   | 注释模块详细描述(详细描述另起一行，可包含多行)               |

例:

```
/**
 * @apiDefine MyError
 * @apiError UserNotFound The <code>id</code> of the User was not found.
 */

/**
 * @api {get} /user/:id
 * @apiUse MyError
 */
/**
 * @apiDefine admin User access only
 * This optional description belong to to the group admin.
 */

/**
 * @api {get} /user/:id
 * @apiPermission admin
 */
```

### @apiDeprecated

```
@apiDeprecated [text]
```

标注一个接口已经被弃用
 参数列表:

| 参数 | 必填 | 描述         |
| :--: | ---- | ------------ |
| text | yes  | 多行文字描述 |

例：

```
**
 * @apiDeprecated
 */

/**
 * @apiDeprecated use now (#Group:Name).
 *
 * Example: to set a link to the GetDetails method of your group User
 * write (#User:GetDetails)
 */
```

### @apiDescription

```
@apiDescription text
```

api接口的详细描述
 参数列表:

| 参数 | 必填 | 描述         |
| :--: | ---- | ------------ |
| text | yes  | 多行文字描述 |

```
/**
 * @apiDescription This is the Description.
 * It is multiline capable.
 *
 * Last line of Description.
 */
```

### @apiError

```
@apiError [(group)] [{type}] field [description]
```

错误返回参数
 参数列表:

|    参数     | 必填 | 描述                                                         |
| :---------: | ---- | ------------------------------------------------------------ |
|   (group)   | no   | 所有的参数都会通过这个参数进行分组，如果未设置，默认值为`Error 4xx` |
|   {type}    | no   | 返回类型(例如:`{Boolean}, {Number}, {String}, {Object}, {String[]}`) |
|    field    | yes  | 返回id                                                       |
| description | no   | 参数描述                                                     |

例:

```
/**
 * @api {get} /user/:id
 * @apiError UserNotFound The <code>id</code> of the User was not found.
 */
```

### @apiErrorExample

```
@apiErrorExample [{type}] [title]
                 example
```

接口错误返回示例(格式化输出)
 参数列表:

|  参数   | 必填 | 描述               |
| :-----: | ---- | ------------------ |
|  type   | no   | 响应类型           |
|  title  | no   | 示例标题           |
| example | yes  | 示例详情(兼容多行) |

例:

```
/**
 * @api {get} /user/:id
 * @apiErrorExample {json} Error-Response:
 *     HTTP/1.1 404 Not Found
 *     {
 *       "error": "UserNotFound"
 *     }
 */
```

### @apiExample

```
@apiExample [{type}] title
            example
```

接口方式请求示例
 参数列表:

|  参数   | 必填 | 描述               |
| :-----: | ---- | ------------------ |
|  type   | no   | 请求内容格式       |
|  title  | yes  | 示例标题           |
| example | yes  | 示例详情(兼容多行) |

```
/**
 * @api {get} /user/:id
 * @apiExample {curl} Example usage:
 *     curl -i http://localhost/user/4711
 */
```

### @apiGroup

```
@apiGroup name
```

定义接口所属的接口组，虽然接口定义里不需要这个参数，但是您应该在每个接口注释里都添加这个参数，因为导出的接口文档会以接口组的形式导航展示。
 参数列表:

| 参数 | 必填 | 描述                            |
| :--: | ---- | ------------------------------- |
| name | yes  | 接口组名称(用于导航,不支持中文) |

例：

```
/**
 * @api {get} /user/:id
 * @apiGroup User
 */
```

### @apiHeader

```
@apiHeader [(group)] [{type}] [field=defaultValue] [description]
```

描述接口请求头部需要的参数(功能类似`@apiParam`)
 参数列表:

|     参数      | 必填 | 描述                                                         |
| :-----------: | ---- | ------------------------------------------------------------ |
|    (group)    | no   | 所有的参数都会以该参数值进行分组(默认`Parameter`)            |
|    {type}     | no   | 返回类型(例如:`{Boolean}, {Number}, {String}, {Object}, {String[]}`) |
|     field     | yes  | 参数名称(定义该头部参数为必填)                               |
|    [field]    | yes  | 参数名称(定义该头部参数为可选)                               |
| =defaultValue | no   | 参数默认值                                                   |
|  description  | no   | 参数描述                                                     |

例:

```
/**
 * @api {get} /user/:id
 * @apiHeader {String} access-key Users unique access-key.
 */
```

### @apiHeaderExample

```
@apiHeaderExample [{type}] [title]
                   example
```

请求头部参数示例
 参数列表:

|  参数   | 必填 | 描述                   |
| :-----: | ---- | ---------------------- |
|  type   | no   | 请求内容格式           |
|  title  | no   | 请求示例标题           |
| example | yes  | 请求示例详情(兼容多行) |

例：

```
/**
 * @api {get} /user/:id
 * @apiHeaderExample {json} Header-Example:
 *     {
 *       "Accept-Encoding": "Accept-Encoding: gzip, deflate"
 *     }
 */
```

### @apiIgnore

```
@apiIgnore [hint]
```

如果你需要使用该参数，请把它放到注释块的最前面。如果设置了该参数，那么该注释模块将不会被解析(当有些接口还未完成或未投入使用时，可以使用该字段)
 参数列表:

| 参数 | 必填 | 描述               |
| :--: | ---- | ------------------ |
| hint | no   | 描接口忽略原因描述 |

例：

```
/**
 * @apiIgnore Not finished Method
 * @api {get} /user/:id
 */
```

### @apiName

```
@apiName name
```

接口名称，每一个接口注释里都应该添加该字段，在导出的接口文档里会已该字段值作为导航子标题，如果两个接口的`@apiVersion`和`@apiName`一样，那么有一个接口的注释将会被覆盖(接口文档里不会展示)
 参数列表:

| 参数 | 必填 | 描述                                             |
| :--: | ---- | ------------------------------------------------ |
| name | yes  | 接口名称(相同接口版本下所有接口名称应该是唯一的) |

例：

```
/**
 * @api {get} /user/:id
 * @apiName GetUser
 */
```

### @apiParam

```
@apiParam [(group)] [{type}] [field=defaultValue] [description]
```

接口请求体参数
 参数列表:

|         参数         | 必填 | 描述                                                         |
| :------------------: | ---- | ------------------------------------------------------------ |
|       (group)        | no   | 所有的参数都会以该参数值进行分组(默认`Parameter`)            |
|        {type}        | no   | 返回类型(例如:`{Boolean}, {Number}, {String}, {Object}, {String[]}`) |
|     {type{size}}     | no   | 返回类型,同时定义参数的范围 `{string{..5}}`意为字符串长度不超过5 `{string{2..5}}`意为字符串长度介于25之间<br/>`{number{100-999}}`意为数值介于100999之间 |
| {type=allowedValues} | no   | 参数可选值 `{string="small"}`意为字符串仅允许值为"small" `{string="small","huge"}`意为字符串允许值为"small"、"huge" `{number=1,2,3,99}`意为数值允许值为1、2、3、99 `{string {..5}="small","huge"`意为字符串最大长度为5并且值允许为:"small"、"huge" |
|        field         | yes  | 参数名称(定义该请求体参数为必填)                             |
|       [field]        | yes  | 参数名称(定义该请求体参数为可选)                             |
|    =defaultValue     | no   | 参数默认值                                                   |
|     description      | no   | 参数描述                                                     |

例:

```
/**
 * @api {get} /user/:id
 * @apiParam {Number} id Users unique ID.
 */

/**
 * @api {post} /user/
 * @apiParam {String} [firstname]  Optional Firstname of the User.
 * @apiParam {String} lastname     Mandatory Lastname.
 * @apiParam {String} country="DE" Mandatory with default value "DE".
 * @apiParam {Number} [age=18]     Optional Age with default 18.
 *
 * @apiParam (Login) {String} pass Only logged in users can post this.
 *                                 In generated documentation a separate
 *                                 "Login" Block will be generated.
 */
```

### @apiParamExample

```
@apiParamExample [{type}] [title]
                   example
```

请求体参数示例
 参数列表:

|  参数   | 必填 | 描述                   |
| :-----: | ---- | ---------------------- |
|  type   | no   | 请求内容格式           |
|  title  | no   | 请求示例标题           |
| example | yes  | 请求示例详情(兼容多行) |

例：

```
/**
 * @api {get} /user/:id
 * @apiParamExample {json} Request-Example:
 *     {
 *       "id": 4711
 *     }
 */
```

### @apiPermission

允许访问该接口的角色名称

```
@apiPermission name
```

参数列表:

| 参数 | 必填 | 描述                     |
| :--: | ---- | ------------------------ |
| name | yes  | 允许访问的角色名称(唯一) |

例：

```
/**
 * @api {get} /user/:id
 * @apiPermission none
 */
```

### @apiPrivate

```
@apiPrivate
```

定义私有接口，对于定义为私有的接口，可以在生成接口文档的时候，通过在命令行中设置参数 `--private false|true`来决定导出的文档中是否包含私有接口
 例：

```
/**
 * @api {get} /user/:id
 * @apiPrivate
 */
```

### @apiSampleRequest

```
@apiSampleRequest url
```

设置了该参数后，导出的html接口文档中会包含模拟接口请求的form表单；如果在配置文件`apidoc.json`中设置了参数`sampleUrl`,那么导出的文档中每一个接口都会包含模拟接口请求的form表单，如果既设置了`sampleUrl`参数，同时也不希望当前这个接口不包含模拟接口请求的form表单，可以使用`@apiSampleRequest off`来关闭。
 参数列表:

| 参数 | 必填 | 描述                                                         |
| :--: | ---- | ------------------------------------------------------------ |
| url  | yes  | 模拟接口请求的url `@apiSampleRequest http://www.example.com`意为覆盖`apidoc.json`中的`sampleUrl`参数 `@apiSampleRequest off`意为关闭接口测试功能 |

例：
 发送测试请求到:`http://api.github.com/user/:id`

```
Configuration parameter sampleUrl: "http://api.github.com"
/**
 * @api {get} /user/:id
 */
```

发送测试请求到:`http://test.github.com/some_path/user/:id`(覆盖`apidoc.json`中的`sampleUrl`参数)

```
Configuration parameter sampleUrl: "http://api.github.com"
/**
 * @api {get} /user/:id
 * @apiSampleRequest http://test.github.com/some_path/
 */
```

关闭接口测试功能

```
Configuration parameter sampleUrl: "http://api.github.com"
/**
 * @api {get} /user/:id
 * @apiSampleRequest off
 */
```

发送测试请求到`http://api.github.com/some_path/user/:id`(由于没有设置`apidoc.json`中的`sampleUrl`参数，所以只有当前接口有模拟测试功能)

```
Configuration parameter sampleUrl is not set
/**
 * @api {get} /user/:id
 * @apiSampleRequest http://api.github.com/some_path/
 */
```

### @apiSuccess

```
@apiSuccess [(group)] [{type}] field [description]
```

接口成功返回参数
 参数列表:

|     参数      | 必填 | 描述                                                         |
| :-----------: | ---- | ------------------------------------------------------------ |
|    (group)    | no   | 所有的参数都会以该参数值进行分组,默认值:`Success 200`        |
|    {type}     | no   | 返回类型(例如:`{Boolean}, {Number}, {String}, {Object}, {String[]}`) |
|     field     | yes  | 返回值(返回成功码)                                           |
| =defaultValue | no   | 参数默认值                                                   |
|  description  | no   | 参数描述                                                     |

例:

```
/**
 * @api {get} /user/:id
 * @apiSuccess {String} firstname Firstname of the User.
 * @apiSuccess {String} lastname  Lastname of the User.
 */
```

包含`(group)`:

```
/**
 * @api {get} /user/:id
 * @apiSuccess (200) {String} firstname Firstname of the User.
 * @apiSuccess (200) {String} lastname  Lastname of the User.
 */
```

返回参数中有对象:

```
/**
 * @api {get} /user/:id
 * @apiSuccess {Boolean} active        Specify if the account is active.
 * @apiSuccess {Object}  profile       User profile information.
 * @apiSuccess {Number}  profile.age   Users age.
 * @apiSuccess {String}  profile.image Avatar-Image.
 */
```

返回参数中有数组：

```
/**
 * @api {get} /users
 * @apiSuccess {Object[]} profiles       List of user profiles.
 * @apiSuccess {Number}   profiles.age   Users age.
 * @apiSuccess {String}   profiles.image Avatar-Image.
 */
```

### @apiSuccessExample

```
@apiSuccessExample [{type}] [title]
                   example
```

返回成功示例
 参数列表:

|  参数   | 必填 | 描述                   |
| :-----: | ---- | ---------------------- |
|  type   | no   | 返回内容格式           |
|  title  | no   | 返回示例标题           |
| example | yes  | 返回示例详情(兼容多行) |

例：

```
/**
 * @api {get} /user/:id
 * @apiSuccessExample {json} Success-Response:
 *     HTTP/1.1 200 OK
 *     {
 *       "firstname": "John",
 *       "lastname": "Doe"
 *     }
 */
```

### @apiUse

```
@apiUse name
```

引入注释模块，如果当前模块定义了`@apiVersion`,那么版本相同或版本最近的注释模块会被引入
 参数列表:

| 参数 | 必填 | 描述               |
| :--: | ---- | ------------------ |
| name | yes  | 引入注释模块的名称 |

例:

```
/**
 * @apiDefine MySuccess
 * @apiSuccess {string} firstname The users firstname.
 * @apiSuccess {number} age The users age.
 */

/**
 * @api {get} /user/:id
 * @apiUse MySuccess
 */
```

### @apiVersion

```
@apiVersion version
```

定义接口/注释模块版本
 参数列表:

|  参数   | 必填 | 描述                                      |
| :-----: | ---- | ----------------------------------------- |
| version | yes  | 版本号(支持APR版本规则:major.minor.patch) |

例:

```
/**
 * @api {get} /user/:id
 * @apiVersion 1.6.2
 */
```