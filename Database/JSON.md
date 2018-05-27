# JSON

json是 JavaScript Object Notation 的首字母缩写，单词的意思是javascript对象表示法，是基于JavaScript对象字面量中表示属性的语法，但并不包括与JavaScript对象字面量的函数相关的部分

## 概念

两种结构：

```
1、对象结构
对象结构是使用大括号“{}”括起来的，大括号内是由0个或多个用英文逗号分隔的“关键字:值”对（key:value）构成的。
语法：
{
    "键名1":值1,
    "键名2":值2,
    "键名n":值n
}
说明：
对象结构是以“{”开始，到“}”结束。其中“键名”和“值”之间用英文冒号构成对，两个“键名:值”之间用英文逗号分隔。
注意，这里的键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。

2、JSON数组结构
JSON数组结构是用中括号“[]”括起来，中括号内部由0个或多个以英文逗号“,”分隔的值列表组成。
语法：
[
    {
        "键名1":值1,
        "键名2":值2
    },
    {
        "键名3":值3,
        "键名4":值4
    },
    ……
]
说明：
arr指的是json数组。数组结构是以“[”开始，到“]”结束，这一点跟JSON对象不同。 在JSON数组中，每一对“{}”相当于一个JSON对象。 
注意，这里的键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。
```

注意：

```
在JSON的名称-值对中，名称始终被双引号包裹
在JSON的名称-值对中，值可以是字符串、数字、布尔值、null、对象、数组
JSOn中名称-值对列表被花括号包裹
在JSON中，多个名称-值对使用逗号分隔
JSON文件使用.json扩展名
JSON的媒体类型是application/json
```

## 数据类型

- 对象

```
{
    "person":{
        "name": "LiLei",
        "age": 25,
        "head": {
            "hair":{
				"color": "black",
				"length": "short"
			},
			"eyes": "black"
        }
    }
}
```

- 字符串

```
{
    "title": "title",
    "body": "body",
    "location": "c:\\Programe Files"
}
```

需要转义的字符

```
\/	正斜线
\b	退格符
\f	换页符
\t	制表符
\n	换行符
\r	回车符
\u	后面跟十六进制制字符
```

- 数字

```
{
    "width": 256,
    "height": 138
}
```

- 布尔

```
{
    "toastWithBreakfast": false,
    "breadWithLunch": true
}
```

- null

```
{
    "hairy": false,
    "watchColor": null
}
```

- 数组

```
{
	"students":[
        "Jane",
        "Tom"
	],
	"scores":[
        93.5,
        87.6
	],
	"answers":[
        true,
        false
	]
	"test":[
        {
            "question": "this is blue",
            "answer": true
        },
        {
            "question": "the earth is flat",
            "answer": false
        }
	]
}
```

## 验证

JSON验证器负责验证语法错误，JSON Schema负责提供一致性检验。

JSON Schema指的是数据交换中的一种虚拟的“合同”，可以解决下列有关一致性验证的问题。

1、  值的数据类型是否正确：可以具体规定一个值是数字、字符串等类型；

2、  是否包含所需的数据：可以规定哪些数据是需要的，哪些是不需要的；

3、  值的形式是不是我需要的：可以指定范围、最小值和最大值。

example:

第一行：声明其为一个schema文件

第二行：文件的标题

第三行：需要JSON中包含的属性

第四行：必需字段



```
{
    "$schema": "http://json-schema.org/draf-04/schema#",
    "title": "Cat",
    "properties": {
        "name": {
            "type": "string",
            "minLength":3,
            "maxLength": 20
        },
        "age": {
            "type": "number",
            "description": "your cat's age in years."
            "minimum": 0
        },
        "declawed": {
            "type": "boolean"
        }
        "description": {
            "type": "string"
        }
    },
    "required": {
        "name",
        ""age"
    }
}
```

## 安全问题

- 跨站请求伪造

CSRF是一种利用站点对用户浏览器新人而发起攻击的方式。

浏览器对于不同域名的站点之间进行资源分享有一定的限制规则，但是`<script>`标签不受规则限制

当处于与银行的会话状态时，json数据采用JSON数组形式，若点击了黑客的带有`<script>`标签的网站，则黑客可窃取敏感的json数据

解决：

不要使用顶级数组，顶级数组是合法的JavaScript脚本，可以用`<script>`标签链接并使用

对于不想公开的资源，仅允许使用HTTP POST方法请求，而不是GET方法。GET方法可以通过URL来请求，甚至可以放到`<script>`中

- 注入攻击

> 跨站脚本攻击

安全漏洞常发生于JavaScript从服务器获取到一段儿JSON字符串并将其转化为JavaScript对象时

```
var jsonString = '{"animal": "cat"}'
var myObject = eval("( + jsonString + )");
alert(myObject)
```

eval()函数会将传入的字符串无差别地编译执行，若从第三方服务器中获取的JSON字符串被替换了恶意脚本，则会在浏览器中编译执行

解决：

JSON.parse()函数会解析JSON，并不会执行脚本

```
var jsonString = '{"animal": "cat"}'
var myObject = JSON.parse(jsonString);
alert(myObject)
```

> 决策上的失误

```
{
    "message": "<div onmouse=\"alert('gotcha!')\">hover here.</div>>"
}
```

这段JavaScript在网站的消息页面中输出时，黑客可以通过该脚本获取你在这一页面上的所有私人信息，并发送到自己的网站上保存

解决：

一方面：采取一些手段使得消息汇总不包含HTML，可在客户端和服务端都加上这一认证；

另一方面：将消息中的所有的HTML进行转义，`<div>`转为`&lt;div&gt;`,然后再插入页面

## 跨域资源请求

- CROS

服务端在响应头额外加上一些带有字段

```
Access-Contorl-Allow-Credentials: true	# 证书可用
Access-Contorl-Allow-Method: GET, POST	# 允许GET和POST方式访问
Access-Contorl-Allow-Origin: *			# 允许任意域名访问
```

- JOSN-P

若想让不同站点共享JSON文件，需要使用JSON-P，运用了`<script>`标签不受同源策略影响来向不同域名的站点请求JSON

```
// 客户端
function getTheAnimal(data){
    var myAnimal = data.animal;
}
// 配置
var script = document.createElement("script");
script.type = "text/javascript";
script.src = "http://notarealdomain.com/animal.json";
document.getElementsByTagName('head')[0].appendChild(script)
// 通过queryString告知服务器函数名字
script.src = "http://notarealdomain.com/animal.json?callback=getThing"
// 服务器
getThing(
	{
        "animal": "cat"
	}
);
```

使用ajax

```
$.ajax({
    url:'js/data.js',
    type:'get',
    dataType:'jsonp',
    jsonpCallback:'fnBack'
})
.done(function(data){
    alert(data.name);
})
.fail(function() {
    alert('服务器超时，请重试！');
});

// data.js里面的数据： fnBack({"name":"tom","age":18});
```

# 前端数据请求

## JavaScript

```javascript
// 创建XMLHttpRequest对象
var myXMLHttpRequest = new XMLHttpRequest();
// 保存JSON资源
var url = "http://api.openweathermap.org/data/2.5/weather?lat=35&lon=139";
// 函数会在readyState属性发生变化时执行
myXMLHttpRequest.onreadystatechange = function(){
    // readyState返回一个0~4的值，0表示未发送(open()未执行)，1表示已发送(open()已执行，send()未执行)，2表示接收到头部(send()已执行切头部和状态码可以获取了)，3表示解析中(头部已经收到，但相应体在解析中)，4表示完成(请求完成，包括相应头和相应体的内容都已经接收到了)
    // status返回HTTP状态码，200表示成功
    if (myXMLHttpRequest.readyState == 4 && myXMLHttpRequest.status == 200){
        // responseText：当请求成功时，该属性会包含作为文本的响应体
        // JSON.parse:解析json文本，转换为JavaScript对象
        var myObject = JSON.parse(myXMLHttpRequest.responseText);
        // 对象的序列化
        var myJSON = JSON.stringify(myObject)
    }
}
// 建立一个JSON请求并发送
myXMLHttpRequest.open("GET", url, true)
myXMLHttpRequest.send()
```

## JQuery

```javascript
var url = "http://api.openweathermap.org/data/2.5/weather?lat=35&lon=139";
$.getJSON(url,function(data)){
          // 解析JSON
          var myObject = jQuery.parseJSON(dat)
		 // 对天气数据执行操作
		...
          });
```





# python解析

从python2.6开始，python标准库中添加了对json的支持，操作json时，只需要`import json`即可

Python3 中可以使用 json 模块来对 JSON 数据进行编解码，它包含了两个函数：

- **json.dumps():** 对数据进行编码。
- **json.loads():** 对数据进行解码。

在json的编解码过程中，python 的原始类型与json类型会相互转换，具体的转化对照如下：

##转换表

Python 编码为 JSON 类型转换对应表：

| Python                                 | JSON   |
| -------------------------------------- | ------ |
| dict                                   | object |
| list, tuple                            | array  |
| str                                    | string |
| int, float, int- & float-derived Enums | number |
| True                                   | true   |
| False                                  | false  |
| None                                   | null   |

JSON 解码为 Python 类型转换对应表：

| JSON          | Python |
| ------------- | ------ |
| object        | dict   |
| array         | list   |
| string        | str    |
| number (int)  | int    |
| number (real) | float  |
| true          | True   |
| false         | False  |
| null          | None   |

##实例

Python 数据结构 ---> JSON字符串：

```
#!/usr/bin/python3

import json

# Python 字典类型转换为 JSON 对象
data = {
    'no' : 1,
    'name' : 'Runoob',
    'url' : 'http://www.runoob.com'
}

json_str = json.dumps(data)
print ("Python 原始数据：", repr(data))
print ("JSON 对象：", json_str)
```

执行以上代码输出结果为：

```
Python 原始数据： {'url': 'http://www.runoob.com', 'no': 1, 'name': 'Runoob'}
JSON 对象： {"url": "http://www.runoob.com", "no": 1, "name": "Runoob"}
```

通过输出的结果可以看出，简单类型通过编码后跟其原始的repr()输出结果非常相似。

JSON字符串  --->  Python数据结构：

```
#!/usr/bin/python3

import json

# Python 字典类型转换为 JSON 对象
data1 = {
    'no' : 1,
    'name' : 'Runoob',
    'url' : 'http://www.runoob.com'
}

json_str = json.dumps(data1)
print ("Python 原始数据：", repr(data1))
print ("JSON 对象：", json_str)

# 将 JSON 对象转换为 Python 字典
data2 = json.loads(json_str)
print ("data2['name']: ", data2['name'])
print ("data2['url']: ", data2['url'])
```

执行以上代码输出结果为：

```
Python 原始数据： {'name': 'Runoob', 'no': 1, 'url': 'http://www.runoob.com'}
JSON 对象： {"name": "Runoob", "no": 1, "url": "http://www.runoob.com"}
data2['name']:  Runoob
data2['url']:  http://www.runoob.com
```

如果你要处理的是文件而不是字符串，你可以使用 **json.dump()** 和 **json.load()** 来编码和解码JSON数据。例如：

```
# 写入 JSON 数据
with open('data.json', 'w') as f:
    json.dump(data, f)

# 读取数据
with open('data.json', 'r') as f:
    data = json.load(f)
```