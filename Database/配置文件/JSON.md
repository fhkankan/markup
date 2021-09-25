# JSON

json是 JavaScript Object Notation 的首字母缩写，单词的意思是javascript对象表示法，是基于JavaScript对象字面量中表示属性的语法，但并不包括与JavaScript对象字面量的函数相关的部分

优点

```
1.数据格式比较简单，易于读写，格式都是压缩的，占用宽带较小
2.易于解析，客户端JavaScript可以简单地通过eval()进行JSON数据的读取
3.支持多种语言，包括C，C#，Java，JavaScript，PHP，Python，Ruby等服务器语言，便于服务器端的解析
4.因为JSON格式可以为服务器端代码使用，简化了服务器端和客户端的代码开发量，且完成任务不变，易于维护
```

缺点

```
没有xml格式推广广泛，也没有xml通用性强
```

## 概念

- 两种结构

对象结构

```javascript
{
    "key1":value1,
    ...
    "keyn":valuen 
}

// 对象结构是使用大括号“{}”括起来的，大括号内是由0个或多个用英文逗号分隔的“关键字:值”对（key:value）构成的。键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。

// 说明：对象结构是以“{”开始，到“}”结束。其中“键名”和“值”之间用英文冒号构成对，两个“键名:值”之间用英文逗号分隔。
```


JSON数组结构
```javascript
[
    {
        "key1":value1,
        "key2":value2
    },
    {
        "key3":value3,
        "key4":value4
    },
    ……
]

// JSON数组结构是用中括号“[]”括起来，中括号内部由0个或多个以英文逗号“,”分隔的值列表组成。键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。

// 说明：arr指的是json数组。数组结构是以“[”开始，到“]”结束，这一点跟JSON对象不同。 在JSON数组中，每一对“{}”相当于一个JSON对象。 
```

>  注意
JSON文件使用.json扩展名
JSON的媒体类型是application/json


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

