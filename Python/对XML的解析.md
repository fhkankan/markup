# XML

XML 指可扩展标记语言（e**X**tensible **M**arkup **L**anguage），标准通用标记语言的子集，是一种用于标记电子文件使其具有结构性的标记语言。 你可以通过本站学习[XML教程](http://www.runoob.com/xml/xml-tutorial.html)

XML 被设计用来传输和存储数据。

XML是一套定义语义标记的规则，这些标记将文档分成许多部件并对这些部件加以标识。

它也是元标记语言，即定义了用于定义其他与特定领域有关的、语义的、结构化的标记语言的句法语言。

# Python解析

常见的XML编程接口有DOM和SAX，这两种接口处理XML文件的方式不同，当然使用场合也不同。

python有三种方法解析XML，SAX，DOM，以及ElementTree:

- SAX (simple API for XML )

python 标准库包含SAX解析器，SAX用事件驱动模型，通过在解析XML的过程中触发一个个的事件并调用用户定义的回调函数来处理XML文件。

- DOM(Document Object Model)

将XML数据在内存中解析成一个树，通过对树的操作来操作XML。

XML实例文件movies.xml内容如下:

```
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

###ContentHandler

```
- characters(content)方法

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

