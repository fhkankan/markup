[TOC]

# XML

XML 指可扩展标记语言（e**X**tensible **M**arkup **L**anguage），标准通用标记语言的子集，是一种用于标记电子文件使其具有结构性的标记语言。 你可以通过本站学习[XML教程](http://www.runoob.com/xml/xml-tutorial.html)

XML 被设计用来传输和存储数据。

XML是一套定义语义标记的规则，这些标记将文档分成许多部件并对这些部件加以标识。

它也是元标记语言，即定义了用于定义其他与特定领域有关的、语义的、结构化的标记语言的句法语言。

# Python解析

常见的XML编程接口有DOM和SAX，这两种接口处理XML文件的方式不同，当然使用场合也不同。

python有三种方法解析XML，SAX，DOM，以及ElementTree:

```
- SAX (simple API for XML )
python 标准库包含SAX解析器，SAX用事件驱动模型，通过在解析XML的过程中触发一个个的事件并调用用户定义的回调函数来处理XML文件。
是流模式，边读边解析，占用内存小，解析快，但是需要自己处理事件

- DOM(Document Object Model)
将XML数据在内存中解析成一个树，通过对树的操作来操作XML。
会把整个XML读入内存并解析为树，占用内存大且解析慢。优点是可以任意遍历树的节点

- ElementTree
提供了轻量级python式API，相对于DOM来说，快了很多，性能与SAX差不多
```

XML实例文件movies.xml内容如下:

```xml
<collection shelf="New Arrivals">
<movie title="Enemy Behind">
   <type>War, Thriller</type>
   <format>DVD</format>
   <year>2003</year>
   <rating>PG</rating>
   <stars>10</stars>
   <description>Talk about a US-Japan war</description>
</movie>
<movie title="Transformers">
   <type>Anime, Science Fiction</type>
   <format>DVD</format>
   <year>1989</year>
   <rating>R</rating>
   <stars>8</stars>
   <description>A schientific fiction</description>
</movie>
   <movie title="Trigun">
   <type>Anime, Action</type>
   <format>DVD</format>
   <episodes>4</episodes>
   <rating>PG</rating>
   <stars>10</stars>
   <description>Vash the Stampede!</description>
</movie>
<movie title="Ishtar">
   <type>Comedy</type>
   <format>VHS</format>
   <rating>PG</rating>
   <stars>2</stars>
   <description>Viewable boredom</description>
</movie>
</collection>
```

## SAX

SAX是一种基于事件驱动的API。

利用SAX解析XML文档牵涉到两个部分:解析器和事件处理器。

解析器负责读取XML文档,并向事件处理器发送事件,如元素开始跟元素结束事件;

而事件处理器则负责对事件作出相应,对传递的XML数据进行处理。

1、对大型文件进行处理；

2、只需要文件的部分内容，或者只需从文件中得到特定信息。

3、想建立自己的对象模型的时候。

在python中使用sax方式处理xml要先引入xml.sax中的parse函数，还有xml.sax.handler中的ContentHandler。

> 方法

ContentHandler

```python
characters(content)方法

调用时机：
从行开始，遇到标签之前，存在字符，content的值为这些字符串。
从一个标签，遇到下一个标签之前， 存在字符，content的值为这些字符串。
从一个标签，遇到行结束符之前，存在字符，content的值为这些字符串。标签可以是开始标签，也可以是结束标签。

startDocument()方法
文档启动的时候调用。

endDocument()方法
解析器到达文档结尾时调用。

startElement(name, attrs)方法
遇到XML开始标签时调用，name是标签的名字，attrs是标签的属性值字典。

endElement(name)方法
遇到XML结束标签时调用。
```

make_parse

```python
# 创建一个新的解析器对象并返回。
xml.sax.make_parser( [parser_list] )

# 参数说明:
parser_list - 可选参数，解析器列表
```

parse

```python
# 创建一个 SAX 解析器并解析xml文档：
xml.sax.parse( xmlfile, contenthandler[, errorhandler])

# 参数说明:
xmlfile - xml文件名
contenthandler - 必须是一个ContentHandler的对象
errorhandler - 如果指定该参数，errorhandler必须是一个SAX ErrorHandler对象
```

parseString

```python
# 创建一个XML解析器并解析xml字符串：
xml.sax.parseString(xmlstring, contenthandler[, errorhandler])

# 参数说明:
xmlstring - xml字符串
contenthandler - 必须是一个ContentHandler的对象
errorhandler - 如果指定该参数，errorhandler必须是一个SAX ErrorHandler对象
```

> 实例

```python
#!/usr/bin/python3

import xml.sax

class MovieHandler( xml.sax.ContentHandler ):
   def __init__(self):
      self.CurrentData = ""
      self.type = ""
      self.format = ""
      self.year = ""
      self.rating = ""
      self.stars = ""
      self.description = ""

   # 元素开始调用
   def startElement(self, tag, attributes):
      self.CurrentData = tag
      if tag == "movie":
         print ("*****Movie*****")
         title = attributes["title"]
         print ("Title:", title)

   # 元素结束调用
   def endElement(self, tag):
      if self.CurrentData == "type":
         print ("Type:", self.type)
      elif self.CurrentData == "format":
         print ("Format:", self.format)
      elif self.CurrentData == "year":
         print ("Year:", self.year)
      elif self.CurrentData == "rating":
         print ("Rating:", self.rating)
      elif self.CurrentData == "stars":
         print ("Stars:", self.stars)
      elif self.CurrentData == "description":
         print ("Description:", self.description)
      self.CurrentData = ""

   # 读取字符时调用
   def characters(self, content):
      if self.CurrentData == "type":
         self.type = content
      elif self.CurrentData == "format":
         self.format = content
      elif self.CurrentData == "year":
         self.year = content
      elif self.CurrentData == "rating":
         self.rating = content
      elif self.CurrentData == "stars":
         self.stars = content
      elif self.CurrentData == "description":
         self.description = content
  
if ( __name__ == "__main__"):
   
   # 创建一个 XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)

   # 重写 ContextHandler
   Handler = MovieHandler()
   parser.setContentHandler( Handler )
   
   parser.parse("movies.xml")
```

## DOM

文件对象模型（Document Object Model，简称DOM），是W3C组织推荐的处理可扩展置标语言的标准编程接口。

一个 DOM 的解析器在解析一个 XML 文档时，一次性读取整个文档，把文档中所有元素保存在内存中的一个树结构里，之后你可以利用DOM 提供的不同的函数来读取或修改文档的内容和结构，也可以把修改过的内容写入xml文件。

python中用xml.dom.minidom来解析xml文件，实例如下：

```python
#!/usr/bin/python3

from xml.dom.minidom import parse
import xml.dom.minidom

# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse("movies.xml")
collection = DOMTree.documentElement
if collection.hasAttribute("shelf"):
   print ("Root element : %s" % collection.getAttribute("shelf"))

# 在集合中获取所有电影
movies = collection.getElementsByTagName("movie")

# 打印每部电影的详细信息
for movie in movies:
   print ("*****Movie*****")
   if movie.hasAttribute("title"):
      print ("Title: %s" % movie.getAttribute("title"))

   type = movie.getElementsByTagName('type')[0]
   print ("Type: %s" % type.childNodes[0].data)
   format = movie.getElementsByTagName('format')[0]
   print ("Format: %s" % format.childNodes[0].data)
   rating = movie.getElementsByTagName('rating')[0]
   print ("Rating: %s" % rating.childNodes[0].data)
   description = movie.getElementsByTagName('description')[0]
   print ("Description: %s" % description.childNodes[0].data)
```

## ET

> 解析

```xml
<? xml version="1.0" encoding="utf-8"?>
<zoo>
	<animal id='1'>
		<name>dog</name>
		<age>2</age>
	</animal>
	<animal id='2'>
		<name>tiget</name>
		<age>3</age>
	</animal>
</zoo>
```

解析代码

```python
# 导入模块
import xml.etree.ElementTree as ET

# 加载文件
root = ET.parse('zoo.xml')
# 获取指定节点
animal_node = root.getiterator("animal")
# 获取子节点
for node in animal_node:
	animal_node_child = node.getchildren()[0]
	print(animal_node_child.tag + ':' + animal_node_child.text)
```

element对象的属性

```python
element.tag 	标签
element.attrib	属性
element.text	值
```

查找方法

```python
# find用于查找指定的第一个节点
zoo_find = root.find('animal')
# findall用于查找指定的所有节点
zoo = root.findall('animal')
```

> 创建

```python
import xml.etree.ElementTree as ET

# 创建根节点
root = ET.Element('root')
# 创建子节点，添加属性
sub1 = ET.SubElement(root, 'sub1')
sub1.attrib = {"attribute":"sub1 attribute"}
# 创建子节点，添加数据
sub2 = ET.SubElement(root, 'sub2')
sub2.text = "new xml"
# 创建子节点，添加数据
sub3 = ET.SubElement(root, 'sub3')
sub3.text = "new xml"
# 创建对象，写入文件
tree = ET.elementTree(root)
tree.write("new.xml")
```

> 修改

```python
ElementTree.write("xmlfile")  更新xml文件
Element.append()	为当前的element对象添加子元素(element)
Element.set(key, value)	为当前element的key属性设置value值
Element.remove(element)	删除为element的节点
```

实例

```python
import xml.etree.ElementTree as ET

# 读取文件
tree = ET.parse("new.xml")
root = tree.getroot()
# 创建新节点，添加属性和数据，并将其设置为root的子节点
sub_new = ET.Element("sub_new")
sub_new.attrib = {"name":"a", "age":30}
sub_new.text = "new element"
root.append(sub_new)
# 修改sub1的属性
sub1 = root.find("sub1")
sub1.set("attribute","new attribute")
# 修改sub2的数据
sub2 = root.find("sub2")
sub2.text = "new value"
# 删除sub3
sub3 = root.find("sub3")
sub3.remove(sub3)
# 写入文件
tree.write("new.xml")
```









