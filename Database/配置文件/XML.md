# XML

[XML教程](http://www.runoob.com/xml/xml-tutorial.html)

## 概述

XML 指可扩展标记语言（e**X**tensible **M**arkup **L**anguage），标准通用标记语言的子集，是一种用于标记电子文件使其具有结构性的标记语言。 

XML 被设计用来传输和存储数据。

XML是一套定义语义标记的规则，这些标记将文档分成许多部件并对这些部件加以标识。

它也是元标记语言，即定义了用于定义其他与特定领域有关的、语义的、结构化的标记语言的句法语言。

优点

```
1.格式统一，符合标准
2.容易与其它系统进行远程交互，数据共享比较方便
```

缺点

```
1.文件庞大，文件格式复杂，传输占宽带
2.服务器端和客户端都需要大量代码来解析xml，导致服务器端和客户端代码变得异常复杂且不易维护
3.客户端不通浏览器之间解析xml的方式不一致，需要重复编写代码
4.服务器端和客户端解析xml花费较多资源和时间
```

基本语法

```
1.文档后缀名.xml
2.第一行必须为文档声明
3.有且仅有一个根标签
4.属性值必须用引号（单双均可）
5.标签必须正确关闭
6.标签名称区分大小写
```

## 实例

`demo.xml`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<note>
  <to>Tove</to>
  <from>Jani</from>
  <heading>Reminder</heading>
  <body>Don't forget me this weekend!</body>
  <user id="1">
    <name>zhangsan</name>
  </user>
  <user id="2">
    <name>lisi</name>
  </user>
  <code>
  	<![CDATA[ if(a>b){} ]]>
  </code>
</note>
```

组成部分

```xml
# 文档声明
version：版本号，必需
encoding：编码
standlone：是否独立，yes/no
# 自定义标签
# 属性id
# CDATA原样输出区域
```

