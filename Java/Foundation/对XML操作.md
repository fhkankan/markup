#  对XML操作

## 概述

操作xml文档
```
1. 解析(读取)：将文档中的数据读取到内存中
2. 写入：将内存中的数据保存到xml文档中。持久化的存储
```
解析xml的方式：
```
1. DOM：将标记语言文档一次性加载进内存，在内存中形成一颗dom树
优点：操作方便，可以对文档进行CRUD的所有操作
缺点：占内存
2. SAX：逐行读取，基于事件驱动的。
优点：不占内存。
缺点：只能读取，不能增删改
```

常见解析器

```
1. JAXP：sun公司提供的解析器，支持dom和sax两种思想
2. DOM4J：一款非常优秀的解析器
3. Jsoup：jsoup 是一款Java 的HTML解析器，可直接解析某个URL地址、HTML文本内容。它提供了一套非常省力的API，可通过DOM，CSS以及类似于jQuery的操作方法来取出和操作数据。
4. PULL：Android操作系统内置的解析器，sax方式的。
```

## Jsoup

jsoup 是一款Java的HTML解析器，可直接解析某个URL地址、HTML文本内容。它提供了一套非常省力的API，可通过DOM，CSS以及类似于jQuery的操作方法来取出和操作数据。

步骤

```
1. 导入jar包
2. 获取Document对象
3. 获取对应的标签Element对象
4. 获取数据
```

### 快速入门

目录

```
- Test
	- libs
		jsoup-1.11.2.jar
	- src
		- cn
			- itcast
				- xml
					- jsoup
						Demo.java
		student.xml
```

`student.xml`

```xml
<?xml version="1.0" encoding="UTF-8" ?>

<students>
	<student number="heima_0001">
		<name id="itcast">
			<xing>张</xing>
			<ming>三</ming>
		</name>
		<age>18</age>
		<sex>male</sex>
	</student>
	<student number="heima_0002">
		<name>jack</name>
		<age>18</age>
		<sex>female</sex>
	</student>

</students>
```

快速入门

```java
package cn.itcast.xml.jsoup;


import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;


public class Demo {
    public static void main(String[] args) throws IOException {
        //2.获取Document对象，根据xml文档获取
        //2.1获取student.xml的path
        String path = JsoupDemo1.class.getClassLoader().getResource("student.xml").getPath();
        //2.2解析xml文档，加载文档进内存，获取dom树--->Document
        Document document = Jsoup.parse(new File(path), "utf-8");
        //3.获取元素对象 Element
        Elements elements = document.getElementsByTag("name");

        System.out.println(elements.size());
        //3.1获取第一个name的Element对象
        Element element = elements.get(0);
        //3.2获取数据
        String name = element.text();
        System.out.println(name);
    }

}
```

### Jsoup

```java
Jsoup // 工具类，可以解析html或xml文档，返回Document

// parse：解析html或xml文档，返回Document
parse(File in, String charsetName)：解析xml或html文件的。
parse(String html)：解析xml或html字符串
parse(URL url, int timeoutMillis)：通过网络路径获取指定的html或xml的文档对象
```

代码

```java
package cn.itcast.xml.jsoup;


import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;
import java.net.URL;


public class JsoupDemo2 {
    public static void main(String[] args) throws IOException {
        //2.1获取student.xml的path
        String path = JsoupDemo2.class.getClassLoader().getResource("student.xml").getPath();
        //2.2解析xml文档，加载文档进内存，获取dom树--->Document
      	Document document = Jsoup.parse(new File(path), "utf-8");
        System.out.println(document);

        //2.parse​(String html)：解析xml或html字符串
      	String str = "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n" +
                "\n" +
                "<students>\n" +
                "\t<student number=\"heima_0001\">\n" +
                "\t\t<name>tom</name>\n" +
                "\t\t<age>18</age>\n" +
                "\t\t<sex>male</sex>\n" +
                "\t</student>\n" +
                "\t<student number=\"heima_0002\">\n" +
                "\t\t<name>jack</name>\n" +
                "\t\t<age>18</age>\n" +
                "\t\t<sex>female</sex>\n" +
                "\t</student>\n" +
                "\n" +
                "</students>";
        Document document = Jsoup.parse(str);
        System.out.println(document);

       //3.parse​(URL url, int timeoutMillis)：通过网络路径获取指定的html或xml的文档对象
        URL url = new URL("https://baike.baidu.com/item/jsoup/9012509?fr=aladdin");//代表网络中的一个资源路径
        Document document = Jsoup.parse(url, 10000);
        System.out.println(document);

    }
}
```

### Document

```java
Document // 文档对象。代表内存中的dom树

// 获取Element对象
getElementById(String id)：根据id属性值获取唯一的element对象
getElementsByTag(String tagName)：根据标签名称获取元素对象集合
getElementsByAttribute(String key)：根据属性名称获取元素对象集合
getElementsByAttributeValue(String key, String value)：根据对应的属性名和属性值获取元素对象集合
```

代码

```java
package cn.itcast.xml.jsoup;


import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Document/Element对象功能
 */
public class JsoupDemo3 {
    public static void main(String[] args) throws IOException {
        //1.获取student.xml的path
        String path = JsoupDemo3.class.getClassLoader().getResource("student.xml").getPath();
        //2.获取Document对象
        Document document = Jsoup.parse(new File(path), "utf-8");

        //3.获取元素对象了。
        //3.1获取所有student对象
        Elements elements = document.getElementsByTag("student");
        System.out.println(elements);

        System.out.println("-----------");


        //3.2 获取属性名为id的元素对象们
        Elements elements1 = document.getElementsByAttribute("id");
        System.out.println(elements1);
        System.out.println("-----------");
        //3.2获取 number属性值为heima_0001的元素对象
        Elements elements2 = document.getElementsByAttributeValue("number", "heima_0001");
        System.out.println(elements2);

        System.out.println("-----------");
        //3.3获取id属性值的元素对象
        Element itcast = document.getElementById("itcast");
        System.out.println(itcast);
    }

}
```

### Element

```java
Elements
// 元素Element对象的集合。可以当做 ArrayList<Element>来使用

Element
// 元素对象
// 获取子元素对象
getElementById(String id)：根据id属性值获取唯一的element对象
getElementsByTag(String tagName)：根据标签名称获取元素对象集合
getElementsByAttribute(String key)：根据属性名称获取元素对象集合
getElementsByAttributeValue(String key, String value)：根据对应的属性名和属性值获取元素对象集合
// 获取属性值
String attr(String key)：根据属性名称获取属性值
// 获取文本内容
String text():获取文本内容
String html():获取标签体的所有内容(包括字标签的字符串内容)

Node
// 节点对象,是Document和Element的父类
```

代码

```java
package cn.itcast.xml.jsoup;


import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;

/**
 *Element对象功能
 */
public class JsoupDemo4 {
    public static void main(String[] args) throws IOException {
        //1.获取student.xml的path
        String path = JsoupDemo4.class.getClassLoader().getResource("student.xml").getPath();
        //2.获取Document对象
        Document document = Jsoup.parse(new File(path), "utf-8");
        //通过Document对象获取name标签，获取所有的name标签，可以获取到两个
        Elements elements = document.getElementsByTag("name");
        System.out.println(elements.size());
        System.out.println("----------------");
        //通过Element对象获取子标签对象
        Element element_student = document.getElementsByTag("student").get(0);
        Elements ele_name = element_student.getElementsByTag("name");
        System.out.println(ele_name.size());

        //获取student对象的属性值
        String number = element_student.attr("NUMBER");
        System.out.println(number);
        System.out.println("------------");
        //获取文本内容
        String text = ele_name.text();
        String html = ele_name.html();
        System.out.println(text);
        System.out.println(html);
    }

}
```

### 选择器

```java
Elements	select(String cssQuery)
// 语法：参考Selector类中定义的语法
```

代码

```java
package cn.itcast.xml.jsoup;


import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;


public class JsoupDemo5 {
    public static void main(String[] args) throws IOException {
        //1.获取student.xml的path
        String path = JsoupDemo5.class.getClassLoader().getResource("student.xml").getPath();
        //2.获取Document对象
        Document document = Jsoup.parse(new File(path), "utf-8");

        //3.查询name标签
        /*
            div{

            }
         */
        Elements elements = document.select("name");
        System.out.println(elements);
        System.out.println("=----------------");
        //4.查询id值为itcast的元素
        Elements elements1 = document.select("#itcast");
        System.out.println(elements1);
        System.out.println("----------------");
        //5.获取student标签并且number属性值为heima_0001的age子标签
        //5.1.获取student标签并且number属性值为heima_0001
        Elements elements2 = document.select("student[number=\"heima_0001\"]");
        System.out.println(elements2);
        System.out.println("----------------");

        //5.2获取student标签并且number属性值为heima_0001的age子标签
        Elements elements3 = document.select("student[number=\"heima_0001\"] > age");
        System.out.println(elements3);
    }
}
```

### Xpath

XPath即为XML路径语言，它是一种用来确定XML（标准通用标记语言的子集）文档中某部分位置的语言

```
使用Jsoup的Xpath需要额外导入jar包JsoupXpath-0.3.2.jar.
查询w3cshool参考手册，使用xpath的语法完成查询
```

代码

```java
package cn.itcast.xml.jsoup;


import cn.wanghaomiao.xpath.exception.XpathSyntaxErrorException;
import cn.wanghaomiao.xpath.model.JXDocument;
import cn.wanghaomiao.xpath.model.JXNode;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import java.io.File;
import java.io.IOException;
import java.util.List;


public class JsoupDemo6 {
    public static void main(String[] args) throws IOException, XpathSyntaxErrorException {
        //1.获取student.xml的path
        String path = JsoupDemo6.class.getClassLoader().getResource("student.xml").getPath();
        //2.获取Document对象
        Document document = Jsoup.parse(new File(path), "utf-8");

        //3.根据document对象，创建JXDocument对象
        JXDocument jxDocument = new JXDocument(document);

        //4.结合xpath语法查询
        //4.1查询所有student标签
        List<JXNode> jxNodes = jxDocument.selN("//student");
        for (JXNode jxNode : jxNodes) {
            System.out.println(jxNode);
        }

        System.out.println("--------------------");

        //4.2查询所有student标签下的name标签
        List<JXNode> jxNodes2 = jxDocument.selN("//student/name");
        for (JXNode jxNode : jxNodes2) {
            System.out.println(jxNode);
        }

        System.out.println("--------------------");

        //4.3查询student标签下带有id属性的name标签
        List<JXNode> jxNodes3 = jxDocument.selN("//student/name[@id]");
        for (JXNode jxNode : jxNodes3) {
            System.out.println(jxNode);
        }
        System.out.println("--------------------");
        //4.4查询student标签下带有id属性的name标签 并且id属性值为itcast

        List<JXNode> jxNodes4 = jxDocument.selN("//student/name[@id='itcast']");
        for (JXNode jxNode : jxNodes4) {
            System.out.println(jxNode);
        }
    }
}
```



