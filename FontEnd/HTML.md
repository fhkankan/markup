[TOC]
# HTML

## 文档

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>
<body>
    hello world!
</body>
</html>
```

快速创建

```
! + tab键
html:5 + tab键
```

### head

title

```
<title>Document</title>

此元素指定了页面的标题
```

meta

```
<meta name="author" content="Mark">
<meta name="description" content="Sample HTML document"/>
<meta http-equiv="refresh" content="45">

元数据，描述其他数据的数据
每个meta远古三使用名称/值对结构提供了单个数据点
```

script

```
<script type="text/javascript">
    function doSomething(){
        alert("hello world!")
    }
</script>
 
javascript在html中有三种嵌入方式：行间，页面script标签，外部引入

在浏览器解析一个html文档的过程中，遇到script元素时会加载并执行脚本，之后再继续解析文档的其他部分。
针对外部文件，可以使用async或defer特性来更改此解析行为。两个都是boolean特性，
当指定了async特性，会以并行方式加载和执行文件，同时解析过程继续进行；
若使用了defer特性，脚本会在页面完全解析之后执行
<script src="../scripts/demo.js" defer></script>
```

link

```
<link rel="stylesheet" href="style.css">
<link rel="shortcut icon" href="favicon.ico" type="image/x-icon">
<link rel="alternate" type="text/plain" href="TextPage.txt">

用于引用额外的外部资源，可分两类：
使用链接来加载呈现源文档所需的资源，常见级联样式表
对其他相关文档的链接，导航到这些文档，同时无需呈现当前页面
```

style

```
<style>
	html {
        color: red;
	}
</style>

可以在元素上显式地指定特定的样式特性。
script：元素包含JavaScript
style：定义样式，可在head和body中
link：加载一个外部的样式文件， 只能在head中
```

base

```
<base href="www.thecreativepeople.com/html5" target="_self">

用于定义文档中所有其他引用所使用的基本URL。就可以在后面使用相对URL。

target特性值：
_blank:在新窗口或选项卡中打开
_self:在当前的窗口或者选项卡找那个打开(默认)
_parent:在父框架中打开
_top:在最顶层的框架中打开
```

## 标签

### 概览

- 按内容分类：

根据内容类型并按照一定的规则对元素进行分类，这些规则定义了元素可以使用的地方以及可以包含的内容。有些元素可以属于多个类别，有些则不属于任何类别

```
元数据元素
没有实际内容，只是为哪些用来处理HTML文档的应用程序提供关于文档的元数据及相关信息

节元素
用来将一个页面组织成若干节，同时也用于构建文档大纲

标题元素
用来在节中定义标题及子标题

嵌入元素
用来在文档中插入非HTML内容，比如图片

交互元素
提供了用户交互。常见一个按钮或输入字段

表单元素
用来捕获用户输入

短语元素
用来标记可合并成段落的文本或短语

流式元素
绝大多数元素都属于此类别。包括实际内容或者嵌入内容
```

- 按占位分类

标签在页面上会显示成一个方块。
块元素：在布局中默认会独占一行，块元素后的元素需换行排列，块元素默认宽度等于父元素的宽度，即使设置了很小宽度，也占用一行。
内联元素：元素之间可以排列在一行，设置宽高无效，它的宽高由内容撑开，内联元素之间默认会有小间距。

- 标签的的使用方法：

```
<!-- 1、成对出现的标签：-->

<h1>h1标题</h1>
<div>这是一个div标签</div>
<p>这个一个段落标签</p>


<!-- 2、单个出现的标签： -->
<br>
<img src="images/pic.jpg" alt="图片">

<!-- 3、带属性的标签，如src、alt 和 href等都是属性 -->
<img src="images/pic.jpg" alt="图片">
<a href="http://www.baidu.com">百度网</a>

<!-- 4、标签的嵌套 -->
<div>
    <img src="images/pic.jpg" alt="图片">
    <a href="http://www.baidu.com">百度网</a>
</div>
```

### 常用块元素标签

- 标题标签，表示文档的标题，除了具有块元素基本特性外，还含有默认的外边距和字体大小

```
<h1>一级标题</h1>
<h2>二级标题</h2>
<h3>三级标题</h3>
<h4>四级标题</h4>
<h5>五级标题</h5>
<h6>六级标题</h6>
```

- 段落标签，表示文档中的一个文字段落，除了具有块元素基本特性外，还含有默认的外边距

```
<p>本人叫张山，毕业于某大学计算机科学与技术专业，今年23岁，本人性格开朗、
稳重、待人真诚、热情。有较强的组织能力和团队协作精神，良好的沟通能力和社
交能力，善于处理各种人际关系。能迅速适应环境，并融入其中。</p>
<p>本人热爱研究技术，热爱编程，希望能在努力为企业服务的过程中实现自身价值。</p>
```

- 通用块容器标签，表示文档中一块内容，具有块元素基本特性，没有其他默认样式

```
<div>这是一个div元素</div>
<div>这是第二个div元素</div>
<div>
    <h3>自我介绍</h3>
    <p>本人叫张山，毕业于某大学计算机科学与技术专业，今年23岁，本人性格开朗、
稳重、待人真诚、热情。有较强的组织能力和团队协作精神，良好的沟通能力和社
交能力，善于处理各种人际关系。能迅速适应环境，并融入其中。</p>
</div>
```

### 常用内联元素标签

- 超链接标签，链接到另外一个网页，具有内联元素基本特性，默认文字蓝色，有下划线

```
<a href="02.html">第二个网页</a>
<a href="http://www.baidu.com">百度网</a>
<a href="http://www.baidu.com"><img src="images/logo.png" alt="logo"></a>
```

- 通用内联容器标签，具有内联元素基本特性，没有其他默认样式

```
<p>这是一个段落文字，段落文字中有<span>特殊标志或样式</span>的文字</p>
```

- 图片标签，在网页中插入图片，具有内联元素基本特性，但是它支持宽高设置。

```
<img src="images/pic.jpg" alt="图片" />
```

### 其他常用功能标签

- 换行标签

```
<p>这是一行文字，<br>这是一行文字</p>
```

- html注释：

html文档代码中可以插入注释，注释是对代码的说明和解释，注释的内容不会显示在页面上，html代码中插入注释的方法是：

```
<!-- 这是一段注释  -->
```

- 常用html字符实体

代码中成段的文字，如果文字间想空多个空格，在代码中空多个空格，在渲染成网页时只会显示一个空格，如果想显示多个空格，可以使用空格的字符实体,代码如下：

```
<!--  在段落前想缩进两个文字的空格，使用空格的字符实体：&nbsp;   -->
<p>
&nbsp;&nbsp;一个html文件就是一个网页，html文件用编辑器打开显示的是文本，可以用<br />
文本的方式编辑它，如果用浏览器打开，浏览器会按照标签描述内容将文件<br />
渲染成网页，显示的网页可以从一个网页链接跳转到另外一个网页。</p>
```

在网页上显示 “<” 和 “>” 会误认为是标签，想在网页上显示“<”和“>”可以使用它们的字符实体，比如：

```
<!-- “<” 和 “>” 的字符实体为 &lt; 和 &gt;  -->
<p>
    &lt;div&gt;是一个html的一个标签<br>
    3 &lt; 5 <br>
    10 &gt; 5
</p>
```

## 布局

### 原理

标签在网页中会显示成一个个的方块，先按照行的方式，把网页划分成多个行，再到行里面划分列，也就是在表示行的标签中再嵌套标签来表示列，标签的嵌套产生叠加效果。

### 标签语义化

在布局中需要尽量使用带语义的标签，使用带语义的标签的目的首先是为了让搜索引擎能更好地理解网页的结构，提高网站在搜索中的排名(也叫做SEO)，其次是方便代码的阅读和维护。

- 带语义的标签 

````
h1~h6：表示标题
p：表示段落
img：表示图片
a：表示链接
````

- 不带语义的标签 

```
div：表示一块内容
span：表示行内的一块内容
```

## 地址

网页上引入或链接到外部文件，需要定义文件的地址，常见引入或链接外部文件包括以下几种：

```
<!-- 引入外部图片   -->
<img src="images/001.jpg" alt="图片" />

<!-- 链接到另外一个网页   -->
<a href="002.html">链接到网页2</a>

<!-- 外链一个css文件   -->
<link rel="stylesheet" type="text/css" href="css/main.css" />

<!-- 外链一个js文件   -->
<script type="text/javascript" src="js/jquery.js"></script>
```

### 相对地址 

相对于引用文件本身去定位被引用的文件地址，以上的例子都是相对地址，相对地址的定义技巧：

- “ ./ ” 表示当前文件所在目录下，比如：“./pic.jpg” 表示当前目录下的pic.jpg的图片，这个使用时可以省略。
- “ ../ ” 表示当前文件所在目录下的上一级目录，比如：“../images/pic.jpg” 表示当前目录下的上一级目录下的images文件夹中的pic.jpg的图片。

### 绝对地址 

相对于磁盘的位置去定位文件的地址，比如：<img src="C:\course5\03day\images\001.jpg" alt="图片" /> 绝对地址在整体文件迁移时会因为磁盘和顶层目录的改变而找不到文件，相对地址就没有这个问题。

## 结构化HTML

### 节内容

将HTML文档组织成不同逻辑节便于管理。可以使用`<div>`进行内容分组并可嵌套，在HTML5中也可以使用具体特定语义的元素

- section

用来将内容组织成逻辑节

指导原则：节应该有主题，应该线性流动

- article

用来对可以独立存在的内容进行分组

一般用于被重复使用的独立存在内容

- aside

用来对那些不属于正常文档流的内容进行分组

通常用作侧边栏显示，如参考目、作者信息、广告等

- nav

用来组织一组链接。

如菜单导航跳转、相关资料的链接

- address

不属于节内容，常用于整个文档或特定文章提供联系信息。

若用于单个文章，放置于article中； 若用于整个文档，放置于body中；通常将其放置于footer中

### 大纲

- 显性节`h1~h6`

```
<body>
   <h1>My Sample Page</h1> 
   <nav>
       <h1>Navigation</h1>
   </nav>
   <section>
       <h1>Top-level</h1>
       <section>
           <h1>Main content</h1>
           <section>
               <h1>Featured content</h1>
           </section>
           <section>
               <h1>Articles</h1>
               <article>
                   <h1>Artile 1</h1>
               </article>>
           </section>
       </section>
       <aside>
           <h1>Related content</h1>
           <section>
               <h1>HTML Reference</h1>
           </section>
           <section>
               <h1>Book list</h1>
               <article>
                   <h1>Book 1</h1>
               </article>
           </section>
       </aside>
   </section>
</body>
```

- header/footer

header和footer元素并不会在文档大纲中创建新的节，而只是对它们所处的节内容进行分组。通常在body元素中使用一个header和一个footer元素来定义页面的页眉和页脚。此外，放在session子节中，将只针对该节中的介绍性内容进行分组

- 页面布局

在页面顶部使用header和nav

在底部使用footer

中间主区域使用一个section，并且包括两个并排区域(每个区域包含一系列的article标签)。其中左边较大部分被包含在另一个section元素中并提供主要内容(这些内容被组织成article元素)。右边较小区域则使用一个aside元素，并包含一个session元素。该区域包含了一系列显示相关信息的article元素。

### 节根

它们拥有自己的大纲，并且与文档剩余部分的大纲没有任何关联

- body

特例，body的大纲就是文档的大纲

- blockquote

用于文档中包括较长的引文。

cite确定引文来源的URL或包含引文信息的资源

```
 <blockquote cite="www.apress.com">
       <h1>Quotation</h1>
       <p>This is a qutotation</p>
 </blockquote>
```

- details

允许创建可折叠的内容节

可以包含一个可选的summary元素，其中包含了details元素被折叠时所显示的内容。

```
   <details open>
       <summary>This is the collapsed text</summary>
       <h1>Details</h1>
       <p>These are collapsable details</p>
   </details>
```

- figure

用于对自包含的内容进行分组，从逻辑上讲，该元素可以移动到不同位置而不会影响主文档流

通常将图像或其他嵌入内容与标题组合在一起。此外，还可以用来组合文本，如带有标题的代码清单

```
   <figure>
       <h1>Figure</h1>
       <img src="HTML5Badge.png" alt="HTML5">
       <figcaption>Offcial HTML5 Logo</figcaption>
   </figure>
```



- fieldset
- td

### 分组

主要用于语义目的，不会对大纲产生影响

- `<p>`

```
用来定义一个段落
```

- `<hr>`

```
4中表示创建一个水平规则，5中表示主题的变化。当主体发生变化，常将其放置在段落之间

   <p>paragraph 1</p>
   <hr / >
   <p>paragraph 2</p>
```

- `<pre>`

```
放置在此标签中的内容将按照输入时的样式显示，包括空格
```

- `<main>`

```
用来表明其内容是文档的主要目的或主题。它确定了文档的核心主题。
main元素不能包含子啊article,aside,footer,header,nav元素中
每个文档中只能有一个main元素，不能被其他的文档所共享
```

- `<div>`

```
5之前，用于所有类型的分组，现在由于针对具体目的可以使用新元素：局部(section),独立或重复(article)，正常文档流之外(aside),导航(nav)等，div用于所有其他的分组原因。
```

### 列表

- ul

若列表的顺序是无意义的，使用无需列表

一般应用在布局中的新闻标题列表和链接列表，它是含有语义的，标签结构如下：

```
<ul>
    <li>列表标题一</li>
    <li>列表标题二</li>
    <li>列表标题三</li>
</ul>
```

- ol

若列表的顺序是有意义的，则使用有序列表

```
<ol start="4">
    <li>列表标题一</li>
    <li>列表标题二</li>
    <li>列表标题三</li>
</ol>

start特性：
指定了第一个li元素所使用的数字，默认1
reversed特性：
布尔特性，用于按相反顺序分配数字，只是影响了项目的编号
type特性：
指定了所使用的编号类型，有
1	数字
A	大写字母
a	小写字母
I	大写罗马数字
i	小写罗马数字
```

- dl

描述列表，名称/值，通常用来创建一个词汇表，名称为所定义的术语，值为术语的定义或描述信息。可以使用一系列的dt和dd

```
   <dl>
       <dt>Terml</dt>
       <dd>Definition</dd>
       <dt>Terml2</dt>
       <dd>Definition</dd>
   </dl>
```

## 短语HTML元素

### 突出显示文本

- 重要性`<strong>`： 不应该忽略的关键字或者重要概念
- 强调`<em>`： 重点阅读，在发音上应该有不同
- 关联`<mark>`： 处于参考目的而突出显示的文本
- 交替声音`<i>`：外文单词，技术术语等
- 细则`<small>`：主文档流之外的尖端法律细节信息
- 删除线`<s>`：不在准确或相关
- 文体突出`<b>`：需要突出的关键字或其他短语
- 无法明确表达`<u>`：表明拼写或语法错误、专有名词或姓氏

### 其他语义短语

- 代码`<code>`
- 变量`<var>`
- 缩写`<abbr>`
- 下标`<sub>`
- 上标`<sup>`
- 时间`<time>`

### 编辑

若想对某一文档进行修改，可以在插入(ins)和删除(del)元素中包含所做的更改

除了全局特性外，还支持：

cite：使用该特性表示更改源

datetime：表示更改所发生的日期、时间

### 引用

内联引用(q)和引用(cite)元素，当然，也可以使用blockquote元素

### span

是一个没有提供任何语义汉仪的通用容器。然而，可以使用span元素中的特性来表明语义信息。常用class特性

### 添加回车

- 换行`<br />`
- 单词换行时机`<wbr />`
- 连字符`&shy;`

## 表格HTML元素

### 简单表格

```
<table>
	<tr>
		<td>One</td>
		<td>Two</td>
	</tr>
	<tr>
		<td>Three</td>
		<td>Four</td>
	</tr>
</table>
```

### 列和行标题

- `<tr>`

```
定义表格中的一行
```

- `<td>`和`<th>`

```
定义一行中的一个单元格
td代表普通单元格，th表示表头单元格


- colspan 设置单元格水平合并，设置值是数值
- rowspan 设置单元格垂直合并，设置值是数值
```

### 列组

`<colgroup>`

列组元素包含这些列，并对列组元素应用不同的样式

方法一：span指定列组包含列数

```
<table>
    <caption>Squares and Cubes</caption>
    <colgroup span="1"></colgroup>
    <colgroup span="1"></colgroup>
    <colgroup span="2" style="background-color: red"></colgroup>
    <tr></tr>
</table>
```

方法二：在列组元素中使用col元素

```
   <table>
        <caption>Squares and Cubes</caption>
        <colgroup>
            <col style="background-color: red"><col/>
        </colgroup>
        <colgroup style="background-color: pink;">
            <col> <span="2"/>
        </colgroup>
        <tr></tr>
   </table>
```

### 表标题和页脚

列标题应该位于表头(thead)元素中，汇总信息应位于表页脚(tfoot)元素中，构成表主要部分的其余行应位于表体(tbody)元素中。

表头、表体和表页脚元素本身不会影响表格的显示方式。

它们元素中的行都被视为一个行组(row group)。当定义表头单元格元素的scope特性时，可以使用rowgroup值来表明标题文本适用于当前行组

```
<table>
	<caption>Squares and Cubes</caption>
	<thead>
		<tr>
			<th>Sample</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td>test</td>
		</tr>
	</tbody>
	<tfoot>
		<tr>
			<td>final</td>
		</tr>
    </tfoot>
</table>
```

### 跨越单元格

在`<th>`和`<td>`中使用特性`colsapn="n"`和`rowspan="m"`特性可以合并单元格。

- `colspan`：合并列单元格
- `rowspan`：合并行单元格



## 嵌入式HTML元素

通过使用嵌入元素，可以由外部文件提供图像或视频等资源。

常用嵌入元素：图像(img), audio, video,embed,object,svg,canvas

### 锚

锚元素用来将内容转换为一个超链接。内容可以是任何类型的流式内容或者段落内容(交互内容除外)，该元素支持href特性，定义了链接将要导航到的URL

```
<a href="http://www.apress.com" target="_self"> Apress </a>

资源类型：
http:web资源
ftp:文本传输
mailto：发送电子邮件
tel:拨打电话号码
file:打开文件

target特性值：
_blank:在新窗口或选项卡中打开
_self:在当前的窗口或者选项卡找那个打开(默认)

download特性：
文件保存的默认名称，如download="name.png"
由于无法使用/\，无法指定默认文件路径

hreflang 表明链接资源额语言
rel 定义了与链接资源的关系
type 表明了资源的MIME类型
```

### 图像

src指定了所引用图像文件的URL

alt提供了文本描述信息，当图像无法下载或格式不支持时显示

```
<img src="Media/HTML5.jpg" alt="The HTML5 Bage logo">
```

- 图片格式

图片是网页制作中很重要的素材，图片有不同的格式，每种格式都有自己的特性，了解这些特效，可以方便我们在制作网页时选取适合的图片格式，图片格式及特性如下：

> psd 

photoshop的专用格式。
优点：完整保存图像的信息，包括未压缩的图像数据、图层、透明等信息，方便图像的编辑。
缺点：应用范围窄，图片容量相对比较大。

> jpg

网页制作及日常使用最普遍的图像格式。
优点：图像压缩效率高，图像容量相对最小。
缺点：有损压缩，图像会丢失数据而失真，不支持透明背景，不能制作成动画。

> gif

制作网页小动画的常用图像格式。
优点：无损压缩，图像容量小、可以制作成动画、支持透明背景。
缺点：图像色彩范围最多只有256色，不能保存色彩丰富的图像，不支持半透明，透明图像边缘有锯齿。

> png

网页制作及日常使用比较普遍的图像格式。
优点：无损压缩，图像容量小、支持透明背景和半透明色彩、透明图像的边缘光滑。
缺点：不能制作成动画

**总结** 
在网页制作中，如何选择合适的图片格式呢？
1、使用大幅面图片时，如果要使用不透明背景的图片，就使用jpg图片；如果要使用透明或者半透明背景的图片，就使用png图片；
2、使用小幅面图片或者图标图片时，使用png图片；如果图片是动画，可以使用gif。

- 多个来源

图像元素包含srcset和sizes特性，可以指定一组带有相关信息的图像文件

> 像素比选择
>
> 需要根据像素密度提供统一图像的一个或多个缩小版本

```html
<img src="Media/HTML5.jpg" alt="The HTML5 Badge logo" srcset="Media/HTML5_2.jpg 2x, Media/HTML5_3.jpg 3x" />
```

> 视口选择
>
> 根据设备或窗口大小选择图片

```html
# 宽度大小
srcset="Media/HTML5.jpg 300w, Media/HTML5_2.jpg 150w"
# 视口的个数
sizes="(max-width: 600px) 25vw, (max-width: 400px), (max-width: 200px) 50vw, 100vw"
```

- 图像映射

将图像放置于锚标签，图像称为超链接

```html
<a href="www.baidu.com"><img src="Media/BAIDU.jpg" alt="The Baidu log" /></a>
```

在单个图像上定义多个区域，每个区域导航到不同的链接，需要图像映射

```html
    <img src="Media/Shapes.png" alt="Shapes" width="150" height="50" usemap="#shapeMap">
    <map name="shapeMap">
        <area shape="rect" coords="0,0,50,50" href="www.baidu.com" alt="squrare" title="Square">
        <area shape="circle" coords="75,25,25" href="www.sina.com" alt="circle" title="Circle">
        <area shape="poly" coords="101,50,126,0,150,50" href="www.taobao.com" alt="triangle" title="Triangle">
    </map>
```

### 音频

src指定音频剪辑资源的位置

当元素不被支持或音频文件无法加载时，使用开始和结束标签中的内容

```
<audio src="Media/Tom.mp3">
    <p>The audio is not supported on your browser</p>
</audio>
```

audio支持很多布尔特性

```
preload 当贤惠页面时预加载音频内容
autoplay 一旦内容加载，音频剪辑会播放
muted 音频静音
loop 当音频剪辑结束时自动从开始出重新播放
controls 用户可以使用本机控件与音频剪辑进行交互
```

- 使用本机控件

```
<audio src="Media/Topm.mp3" autoplay controls></audio>
```

在用户界面方面，主要有三个选项

> 没有控件

音频播放，但用户没有控件可用，若使用了autoplay，页面被加载时自动开始播放音频，但可以痛殴javascript开始、暂停和停止

> 本机控件

浏览器为用户提供了播放、暂停和停止音频以及控制音量所需的本机控制

> 自定义控件

页面提供了通过JavaScript与audio元素进行交互的自定义控件

- 多文件格式

```
<audio autoplay controls>
	<source src="Media/Tom.ogg" />
    <source src="Media/Tom.mp3" />
    <p>The audio is not supported on your browser</p>
</audio>
```

### 视频

```
<video src="Media/Tom.mp4" controls poster="Media/Poster_1.png" width="852" height="480">
    <p>The video is not supported on your browser</p>
</video>
```

支持特性

```
autoplay 在页面加载时启动视频，但无本机控件，但在视频右击，可现实交互选项菜单
controls 提供控件，但需鼠标悬停在视频上，否则会隐藏
source 指定多个源
poster 在视频播放前，指定所显示的图像
```

### 轨道

track元素用于提供与媒体剪辑时间同步的基于文本的内容，如音频歌词，视频字幕

track是一个空元素，使用自结束标签，仅通过其特性进行配置，只能在audio和video中使用。若使用了source元素提供的多个文件类型，则track元素应位于source之后

- kind

指定了轨道细节信息的作用，包括如下几个值

> captions

用于隐藏字幕，该轨道提供了对白的转录以及相关的声音效果，主要用于听力受损的用户或音频静音时

> chapters

用于较长的视频剪辑，当用户通过媒体文件导航是，该轨道提供了章节标题

> descriptions

提供了音频或视频文件内容的文本描述。主要用于视力受损的用户或当视频不可用时

> matedata

提供了脚本所使用的相关数据，通常不想用户展示

> subtitles

若无指定kindl，此为默认值。提供了不同语言之间的对白文本转换，提供额外的信息，如描述事件的日期和地点

```html
<video src="Media/Tom.mp4" controls 
            poster="Media/Poster_1.png" width="852" height="480">
        <track kind="captions" src="bbb.vtt" srclang="en" label="English"/>
        <p>The video is supported on your browser</p>
</video>
```

### HTML5插件

object是任何外部内容的通用容器，对象的实际类型由type特性所定义

有两个必须特性：data定义了外部资源，type指定了应是使用什么插件来显示内容

可以指定height/width特性，为该内容分配所需的空间

允许通过param元素向插件传递参数

```
<object data="some file" type="application/some plug in">
    <param name="paramName" value="paramValue" />
    <p>fallback content</p>
</object>
```

## 表单HTML元素

### 表单元素

```
   <form action="" method="get">
       <label for="iFirstName">First Name</label>
       <input id="iFirstName" type="text" />
       <label for="iLastName">Last Name</label>
       <input id="iLastName" type="text" />
       <input type="submit" value="Submit" />
   </form>
```

#### 表单动作

在示例字段中输入数据并点击提交，就会发送与下方类似的HTML请求

```
http://localhost:5266/?FirstName=Mark&LastName=Collins
```

- action属性 定义表单数据提交地址

#### 表单方法

在HTML5中只支持GET和POST方法

get动词不支持消息体，所有的表单数据必须经过URL传递

post动词，表单数据位于请求体中

- method属性 定义表单提交的方式

#### 附件特性

- enctype特性用来控制如何对数据进行格式化
  - text/plain时，空格将被转换为+
  - multipart/from-data时，上传1个或多个文件
- accept-charset特性指定服务器所支持的字符集
- novalidate特性可禁用客户端验证

### 输入元素

#### 文本数据

- 文本值

```
input中有几种输入类型是基于type特性提供数据验证的文本框
text		未指定type时是默认值
email		内置验证文本格式是否符合标准电子邮件地址，与普通文本类似
password	以星号或其他掩盖输入的方法来显示所输入的字符，以明文传输
search		与普通文本类似，但是可添加autosave
tel			用于输入电话号码，仅用于语义目的，没有内置的验证
url			与email类似，验证了输入的文本是否是格式良好的URL，单并不验证资源是否存在
```

- textarea

```
定义多行文本输入框，是一个单独元素，并不是input的元素类型，与类型为text的input元素功能类似

支持特性
cols		指定单行应该显示的字符数量
rows		指定可见的行数
wrap		指定了文本换行的方式，允许hard/soft(默认)
inputmode	针对emial/password/text/url等类型的input，作为一个提示来表明显示哪种键盘
maxlength	指定字段可以输入的最大字符数
minlength	指定字段可以输入的最小字符数
pattern		指定了验证输入数据所使用的正则表达式
placeholder	占位符文本放置在实际输入内容的元素内，表示了字段中的期望数据
size		以输入的字符数指定输入元素的物理大小
spellcheck	boolean特性，指示输入数据是否应进行拼写好语法检查
```

- 自动填充

在form元素上将autocomplete特性设置为off/on,将可针对表单所有字段关闭或开启自动填充功能

可以使用datalist元素提供自定义的自动填充列表

```
<datalist id="sports">
    <option value="Baseball"></option>
    <option value="Basketball"></option>
    <option value="Hockey"></option>
    <option value="Football"></option>
</datalist>
   <label for="iSport">Favorite Sport:</label>
   <input type="text" id="iSport" name="Sport" list="sports" />
```

- 其他共有特性

```
name		表单数据的key
value		表单数局的value
disabled	boolean特性，true时禁止任何用户与input元素进行交互，禁用字段不提交
readonly	boolean特性，true时阻止用户更改input元素的值，可交互与提交
required	boolean特性，true时用户必须为元素输入或选择一个值
autofocus	boolean特性，true时表示当加载页面时input元素应该拥有焦点
```

#### 选择元素

- 复选框

选中表示true，未选中表示false

```
   <p>
       Toppings:
       <input type="checkbox" name="Topping" value="Mushrooms" />Mushroom?
       <input type="checkbox" name="Topping" value="Sausage" />Sausage?
       <input type="checkbox" name="Topping" value="Olives" />Olives?
   </p>
```

- 单选按钮

```
   <p>
       Crust
       <input type="radio" name="Crust" value="Thin" />Thin
       <input type="radio" name="Crust" value="Thick" />Thick
       <input type="radio" name="Crust" value="DeepDish" />Thick
   </p>
```

- 下拉列表

```
   <p>
       Addons:
       <select name="Addons" required>
           <option value="">Please select...</option>
           <option value="None">Pizza only</option>
           <optgroup label="Addons">
                <option value="Wings">Side of Buffalo Wings</option>
                <option value="GarlicBread">Add Garlic Bread</option>
           </optgroup>
       </select>
   </p>
```

- 多选列表

```
       <select name="Topping" multiple size="4">
           <option label="Mushroom?" value="Mushrooms"></option>
           <option label="Sausage?" value="Sausage"></option>
       </select>
```

#### 其他类型

- number		数字

```
<input type="number" min="1" max="4" value="1" name="Utensils">
```

- color		颜色
	 file	                文件

```
<input type="file" name="music" accept="audio/*" />
<input type="file" name="pictures" multiple accept=".jpg, .png" />
```

- range		不精确的数字输入控制

```
<datalist id="SurveyStops">
    <option value="0"></option>
    <option value="20"></option>
    <option value="40"></option>
    <option value="60"></option>
</datalist>
<p>
	<input type="range" name="Survey" min="0" max="50" step="10" list="SurveyStops"/>
</p>
```

#### 日期时间

```
date			不带有时间部分的日期
datetime-local	日期和时间，浏览器的本地时间
time             本地时间
month			一个特定年份和月份
week			一个特定的星期，表示为年和周数
```

eg

```
<p>
   Date:
   <input type="date" name="Date" min="2016-08-06" max="2026-08-06" placeholder="mm/dd/yy" />
   Date/Time:
   <input type="datetime-local" name="DateTime" step="30" placeholder="mm/dd/yy hh:mm:ss Am" />
   Time:
   <input type="time" name="Time" min="10:00:00" max="17:00:00" step="15" placeholder="hh:mm:ss AM" />
   Month:
   <input type="month" name="month" name="Month" min="2016-01-01" max="2026-12-31" placeholder="yyyy-mm" />
   Week:
   <input type="week" name="Week" min="2016-01-01" max="2026-12-31" placeholder="yyyy-W##" />
   </p>
```

### 可视元素

- label

label元素与其描述的input元素相关联，两种方式:input嵌套进label或for指定

```
   <p>Deliver to: <br />
        <label>
            Address:
            <input type="text" size="30" name="Address" />
        </label>
        <label for="telephone">Phone #:</label>
        <input type="tel" id="telephone" name="Phone" />
   </p>
```

- output

HTML5之后有了output元素，用作基于用户输入数据的计算输出数据

支持for特性，通过该特性，可以指定用来计算结果值得input元素，每个input元素的id应包含在for特性中，并由空格分隔

```
<p>
    <label>
        Total due:
        <output id="total" name="Total">$0.00</output>
    </label>
</p>
```

- meter

meter元素与前面的range元素相类似，显示的都是一个值，但不同的是meter元素显示的不是一个特定的数字，而是沿着刻度的位置值。meter元素的值不能被用户修改

```
<p>
        Meter example:<br/>
        <meter min="0" max="100" low="33" high="65" optimum="66" value="25"></meter>
        <meter min="0" max="100" low="33" high="65" optimum="66" value="50"></meter>
        <meter min="0" max="100" low="33" high="65" optimum="66" value="75"></meter>
        Optimal: high<br/>
        <meter min="0" max="100" low="34" high="66" optimum="33" value="25"></meter>
        <meter min="0" max="100" low="34" high="66" optimum="33" value="50"></meter>
        <meter min="0" max="100" low="34" high="66" optimum="33" value="75"></meter>
        Optimal: low<br/>
        <meter min="0" max="100" low="32" high="66" optimum="33" value="25"></meter>
        <meter min="0" max="100" low="32" high="66" optimum="33" value="50"></meter>
        <meter min="0" max="100" low="32" high="66" optimum="33" value="75"></meter>
        Optimal: media<br/>
</p>
```

- progress

与meter类似，但无low、high和optimum特性，只是显示进度

```
<p>
    Progress example:
    <progress min="0" max="10" value="3">
            Your browser does not support the progress element! Value is 3 to 7.
    </progress>
</p>
```

### 按钮类型

input的四种类型与button的三种类型本质上是等同的。

input没有内容，文本是通过value特性设置，src特性用于指定一个图像(image)

button较新，可以使用任何类型的短语内容，所有元素都包括图像，使用了正则内容，拥有更多样式，包括::before和::after伪元素

- input元素支持四种type值

```
- submit 单击时提交订单
- reset 单击时清楚input元素，并将表单返回到原始值
- image 功能与submit类似，但支持使用src特性指定一个按钮所使用的图象
- button 创建一个没有默认动作的按钮，可以用它实现自定义动作(使用JavaScript)
```

- button元素支持三种type值

```
- submit 提交表单
- reset 重设表单
- button 不带有默认动作的按钮
```

eg

```
    <input type="submit" value="Submit" />
    <button type="submit">Submit</button>
```

### 组织表单

fieldset元素用于提供input元素的可视分组。会在input元素周围绘制一个框。

fieldset元素中包含一个legend元素，用来定义框所显示的文本

支持disable特性，则fieldset中的所有input元素会被禁用

```
<fieldset>
        <legend>Topping:</legend>
                <input type="checkbox" name="Topping" value="Mushrooms" />Mushroom?
                <input type="checkbox" name="Topping" value="Sausage" />Sausage?
                <input type="checkbox" name="Topping" value="Olives" />Olives?
</fieldset> 
```

### 验证

maxlength 设定最大值

pattern 确保数据有效

placeholder 提供暗示，以便用户知道期望的内容

required 指定必须被填充的字段