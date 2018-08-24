[TOC]

# CSS

```
页面表现：元素大小、颜色、位置、隐藏或显示、部分动画效果
```

```
css的定义方法是：
选择器 { 属性：值； 属性：值； 属性：值；}
```

## 载入方式

```
1、内联式：通过标签的style属性，在标签上直接写样式。
<div style="width:100px; height:100px; background:red ">......</div>
2、嵌入式：通过style标签，在网页上创建嵌入的样式表。
<style type="text/css">
    div{ width:100px; height:100px; background:red }
    ......
</style>
3、外链式：通过link标签，链接外部样式文件到页面中。
<link rel="stylesheet" type="text/css" href="css/main.css">
```

## 选择器

- 单一

> 元素选择器

标签选择器，此种选择器影响范围大，一般用来做一些通用设置，或用在层级选择器中。
` div{}`

> 类选择器

通过类名来选择元素，一个类可应用于多个元素，一个元素上也可以使用多个类，应用灵活，可复用，是css中应用最多的一种选择器。
` .className{}`

> ID选择器

通过id名来选择元素，元素的id名称不能重复，所以一个样式设置项只能对应于页面上一个元素，不能复用，id名一般给程序使用，所以不推荐使用id作为选择器。
` #Submit{}`

> 特性选择器

根据元素的任何特性来选择元素，通常以`[attribute=value]`格式

允许使用特性值得部分内容来进行匹配，需要使用以下符号作为等号的前缀

```
~	[class~="book"]	特性值必须包含选择器值所指定的单词
|	[class|="book"] 特性值必须以与选择器值匹配的单词开头
^	[class^="book"]	特性值必须以选择器值开头
$	[class$="book"] 特性值必须以选择器值结尾
*	[class*="book"]	特性值必须包含选择器值
```

> 伪类选择器

伪类选择与常规类型特性相似，不过他们由浏览器自动添加，不在HTML标记中设置，以`:`做前缀

```
.box1:hover{color:red}
```

常用的有

```
:focus	选择当前拥有焦点的元素
:hover	选择鼠标当前悬停的元素
:invalid 选择没有有效值得输入元素
:valid  选择具有有效值得输入元素
:link	选择所有未访问的链接
:visited 选择所有访问的链接
:active 选择刚刚被单击的链接
```

> 伪元素选择器

伪元素返回了新的虚拟元素，这些元素不属于DOM的一部分，可以是空元素或现有元素的一部分

```
.box2:before{content:'行首文字';}
.box3:after{content:'行尾文字';}
```

可用的有

```
::after			在所选元素后面创建一个空元素
::before		在所选元素的前面创建一个空元素
::first-letter	选择每个选定元素的第一个字符
::first-line	选择每个选定元素的第一行
::selection		返回用户所选择元素的一部分
```

- 组合

> 组合元素和类选择器

元素和类选择器以`.`组合

`p.featured{}`

> 伪类选择器

元素和伪类组合

`a:visited{}`

> not选择器

在任何选择器之前使用`:not`，范湖所有没有被选择的元素

`body:not(header){}`

> 组合运算符

组合选择器来指定某种元素层次结构

```
1. Group,
一个逻辑OR运算符，多个选择器，如果有同样的样式设置，可以使用组选择器。
eg: .box1,.box2,.box3{width:30px;height:30px}
2. Descendant 空格
当第二个元素位于第一个元素内时(不一定父节点)，选择第二个元素
eg: .className a{}
3. Child>
当第一个节点是直接父节点时，选择第二个元素
eg: header>p
4. Adjacent Sibling+
当第一个元素是第二个元素的前一个兄弟节点时，选择第二个元素
eg: header+p
5. Follows~
当第二个元素跟在第一个元素之后(不一定直接相邻)，选择第二个元素
eg：p~header
```

## 权重

```
CSS权重指的是样式的优先级，有两条或多条样式作用于一个元素，权重高的那条样式对元素起作用,权重相同的，后写的样式会覆盖前面写的样式。

权重的等级
1、!important，加在样式属性值后，权重值为 10000
2、内联样式，如：style=””，权重值为1000
3、ID选择器，如：#content，权重值为100
4、类，伪类，如：.content、:hover 权重值为10
5、标签选择器，如：div、p 权重值为1
```

## 定位内容

### 文档流

文档流，是指盒子按照html标签编写的顺序依次从上到下，从左到右排列，块元素占一行，行内元素在一行之内从左到右排列，先写的先排列，后写的排在后面，每个盒子都占据自己的位置。

### 块元素类型

- 块元素特性

块元素，也可以称为行元素

布局中常用的标签如：div、p、ul、li、h1~h6等等都是块元素

它在布局中的行为：

```
支持全部的样式
如果没有设置宽度，默认的宽度为父级宽度100%
盒子占据一行、即使设置了宽度
```

- 包含默认样式的块元素

上面讲的块标签中，有些标签是包含默认的样式的，这个含默认样式的有

```
- p标签：含有默认外边距
- ul：含有默认外边距和内边距，以及条目符号
- h1~h6标签：含有默认的外边距和字体大小
- body标签：含有默认的外边距
```

实际开发中，我们会把这些默认的样式在样式定义开头清除掉，清除掉这些默认样式，方便我们写自己的定义的样式，这种做法叫样式重置。

### 内联元素

- 内联元素特性

内联元素，也可以称为行内元素

布局中常用的标签如：a、span等等都是内联元素

它们在布局中的行为：

```
不支持宽、高、margin上下、padding上下
宽高由内容决定
盒子并在一行
代码换行，盒子之间会产生间距
子元素是内联元素，父元素可以用text-align属性设置子元素水平对齐方式
```

- 解决内联元素间隙的方法

1、去掉内联元素之间的换行
2、将内联元素的父级设置font-size为0，内联元素自身再设置font-size

- 其他内联元素

```
- em 标签 行内元素，表示语气中的强调词
- i 标签 行内元素，表示专业词汇
- b 标签 行内元素，表示文档中的关键字或者产品名
- strong 标签 行内元素，表示非常重要的内容
```

- 包含默认样式的内联元素

```
- a标签：含有的下划线以及文字颜色
- em、i标签：文字默认为斜体
- b、strong标签：文字默认加粗
```

### 内联块元素

内联块元素，也叫行内块元素，是新增的元素类型，现有元素没有归于此类别的，img和input元素的行为类似这种元素，但是也归类于内联元素，我们可以用display属性将块元素或者内联元素转化成这种元素。它们在布局中表现的行为：

```
支持全部样式
如果没有设置宽高，宽高由内容决定
盒子并在一行
代码换行，盒子会产生间距
子元素是内联块元素，父元素可以用text-align属性设置子元素水平对齐方式。
```

### Display

display属性是用来设置元素的类型及隐藏的，常用的属性有：

```
- none 元素隐藏且不占位置
- block 元素以块元素显示
- inline 元素以内联元素显示
- inline-block 元素以内联块元素显示
```

### 定义大小

width 设置元素(标签)的宽度

height 设置元素(标签)的高度

- 绝对大小

```
.container{
    width: 150px;
    height: 100px;
}
```

- 相对大小

```
.container{
    width: 70%;
    height: 50%;
}
```

- 最大值

```
.container{
    max-width: 300px;
}
```

- 基于内容

min-content 在了解了所有换行机会后，使用可适应内容的最小可能区域

max-content 使用不需要对任何内容进行换行的最小空间

```
.container{
    width: max-content;
    height: max-content;
}
```

- 盒子大小调整

```
padding 设置元素包含的内容和元素边框的距离，也叫内边距，如padding:20px;

margin 设置元素和外界的距离，也叫外边距，如margin:20px;

border，padding，margin均可分别设置四个边top,right,bottom，left

z-index 设置优先级

/* 设置元素圆角,将元素四个角设置4px半径的圆角 */
border-radius:4px;

/* 设置元素透明度,将元素透明度设置为0.3，此属性需要加一个兼容IE的写法 */
opacity:0.3;
/* 兼容IE */
filter:alpha(opacity=30);
```

### 浮动

float 设置元素浮动，浮动可以让块元素排列在一行

- 浮动特性

```
浮动元素有左浮动(float:left)和右浮动(float:right)两种

浮动的元素会向左或向右浮动，碰到父元素边界、其他元素才停下来

相邻浮动的块元素可以并在一行，超出父级宽度就换行

浮动让行内元素或块元素转化为有浮动特性的行内块元素(此时不会有行内块元素间隙问题)

父元素如果没有设置尺寸(一般是高度不设置)，父元素内整体浮动的子元素无法撑开父元素，父元素需要清除浮动
```

- 清除浮动

> clear

在最后一个子元素的后面加一个空的div，给它样式属性 clear:both（不推荐）

> overflow

父级上增加属性overflow：hidden

```
当子元素的尺寸超过父元素的尺寸时，需要设置父元素显示溢出的子元素的方式，设置的方法是通过overflow属性来设置。

overflow的设置项： 
1、visible 默认值。内容不会被修剪，会呈现在元素框之外。
2、hidden 内容会被修剪，并且其余内容是不可见的，此属性还有清除浮动、清除margin-top塌陷的功能。
3、scroll 内容会被修剪，但是浏览器会显示滚动条以便查看其余的内容。
4、auto 如果内容被修剪，则浏览器会显示滚动条以便查看其余的内容。
```

> 伪元素

使用成熟的清浮动样式类，clearfix

```
.clearfix:after,.clearfix:before{ content: "";display: table;}
.clearfix:after{ clear:both;}
.clearfix{zoom:1;}
```

清除浮动的使用方法：

```
.con2{... overflow:hidden}
或者
<div class="con2 clearfix">
```

### 定位

我们可以使用css的position属性来设置元素的定位类型，postion的设置项如下：

- relative 生成相对定位元素，元素所占据的文档流的位置保留，元素本身相对自身原位置进行偏移。
- absolute 生成绝对定位元素，元素脱离文档流，不占据文档流的位置，可以理解为漂浮在文档流的上方，相对于上一个设置了定位的父级元素来进行定位，如果找不到，则相对于body元素进行定位。
- fixed 生成固定定位元素，元素脱离文档流，不占据文档流的位置，可以理解为漂浮在文档流的上方，相对于浏览器窗口进行定位。
- static 默认值，没有定位，元素出现在正常的文档流中，相当于取消定位属性或者不设置定位属性。

**定位元素的偏移** 
定位的元素还需要用left、right、top或者bottom来设置相对于参照元素的偏移值。

**定位元素层级** 
定位元素是浮动的正常的文档流之上的，可以用z-index属性来设置元素的层级

伪代码如下:

```
.box01{
    ......
    position:absolute;  /* 设置了绝对定位 */
    left:200px;            /* 相对于参照元素左边向右偏移200px */
    top:100px;          /* 相对于参照元素顶部向下偏移100px */
    z-index:10          /* 将元素层级设置为10 */
}
```

**定位元素特性** 
绝对定位和固定定位的块元素和行内元素会自动转化为行内块元素

### 水平居中

margin设置为0 auto，使div水平居中

```
.center{
    width: 100px;
    height: 30px;
    background-color: teal;
    margin: 0 auto;
}
```

## 文本样式

### 字体

- 字体系列

```
font-family 设置文字的字体，如：font-family:'微软雅黑';为了避免中文字不兼容，一般写成：font-family:'Microsoft Yahei';
```

通用字体系列

```
cursive 手写体，类似草书
fantasy 艺术体，有装饰或非标准字符表现形式
monospace 等宽字体
sans-serif 无衬线字体
serif  印刷体
```

- 字体设置

```
font-style 设置字体是否倾斜，如：font-style:'normal'; 设置不倾斜，font-style:'italic';设置文字倾斜
font-size 设置文字的大小，如：font-size:12px;
font-weight 设置文字是否加粗，如：font-weight:bold; 设置加粗 font-weight:normal 设置不加粗
opacity 设置文字的透明度，如: opacity:0.3;
color 设置文字的颜色，如： color:red;
font-kerning 设置字距调整，有auto,normal,none
font-streth 设置字体拉伸宽度，有normal, extra-condensed, extra-expanded等

font 同时设置文字的几个属性，写的顺序有兼容问题，建议按照如下顺序写： font：是否加粗 字号/行高 字体；如： font:normal 12px/36px '微软雅黑';
```

### 格式化

```
text-align 设置文字水平对齐方式，如text-align:center 设置文字水平居中
text-indent 设置文字首行缩进
text-decoration 设置文字的下划线，如：text-decoration:none; 将文字下划线去掉
text-shadow 为文本添加阴影，接收一个颜色值和三个距离值(x偏移量，y偏移量，模糊半径)
text-transform 强制文本大小写，仅影响浏览器上字符的呈现方式，而在DOM中文本仍保持原始值，支持：capitalize,uppercase,lowercase,none
```

### 间距和对齐

基本间距

```
letter-spacing 若是正值，则创建字符之间的额外空间，若为负值，则删除空间，把那个可能产生重叠，允许像素和相对单位(em,rem,vw)。默认值normal。
word-sapcing 影响单词和内联元素之间的空间。可以使绝对值或相对值，不支持负数。默认值normal
line-height 指定每行文本的垂直空间，默认值normal，比字体大小大约20%，也可直接指定，如：line-height:24px; 表示文字高度加上文字上下的间距是24px
list-style 设置无序列表中的小圆点，一般把它设为"无"，如：list-style:none
```

处理空白

```
空白字符通常被忽略，这是浏览器的默认行为，除非空白字符包含在pre元素中，然而可以使用white-space特性，更改空白字符的处理方式
white-space特性支持如下值：
normal		默认值，空白字符序列被折叠为单个空格，而换行符被视为一个空格。文本仅在需要时换行，以填充块
nowrap		空白被折叠，但文本不会换行
pre			与pre元素的功能类似，保留空白字符，文本仅在换行符除换行
pre-wrap	与pre值类似，文本只在需要时换行，以填充块
pre-line	与normal值类似，空白字符被折叠，但文本可以子啊碰到换行符时换行
```

垂直对齐

```
vertical-align特性可以实现垂直对齐，但是主要适用于表格单元格，其他内容，则仅适用于inline和inline-block元素
verttical-align的特性值:
baseline	元素的基线与父元素的基线对齐
bottom		元素的底部与当前行的底部对齐
middle		元素的中间与父元素的基线加上父元素中字母x的高度的一半对齐
sub			基线与父元素的下标基线对齐
super		基线与父元素的上标基线对齐
text-bottom	元素的底部与父元素的底部对齐
text-top	元素的顶部与父元素的顶部对齐
top			元素的顶部与当前行的顶部对齐
```

### break

单词换行

```
overflow-wrap	确定插入换行符的位置，以便为了使用容器的宽度二换行文本
此特性的值：
noraml		换行只能在单词之间
wbr			元素中或者硬或软连字符上发生

word-break	类似overflow-wrap
此特性的值：
normal		换行只能在单词之间
break-all	元素中或者硬或软连字符上发生
keep-all	应用于中文、日文和韩文

hyphens		连字符
此特性的值：
none		当换行一个单词时，不使用连字符			
manual		只有当硬或软连字符存在于单词换行的位置时，才使用连字符
auto		默认值，类似manual
```

分页符

```
page-break-after	标识要在元素末尾应用的规则
page-break-before	指定应在元素前面引用的任何分页规则
此两特性支持的值：
auto		默认值，未定义任何规则
avoid		防止在特定元素中间出现分页符
always		在元素之前或之后始终应该创建一个分页符
left		类似work值，但不同的是它可能会产生一个或两个分页符并强制下一页为左侧页
right		类似left，强制下一页为右侧页

page-break-inside	标识是否允许分页符出现在元素内部
此特性的值：
auto		默认值，未定义任何规则
avoid		防止在特定元素中间出现分页符
```

### 光标

```
# 设置光标的形状
a{ cursor:pointer; }

# 指定图像文件的URL自定义光标
a{
    cursor: url(custom.png), url(fallback.cur), crosshair;
}
```

## 边框和背景

### 边框

```
border 设置元素四周的边框，如：border:1px solid black; 设置元素四周边框是1像素宽的黑色实线

等同于
boder-width: 1px;
border-style: solid;
border-color: black;
```

- 基本样式

```
<section>
	<div class="box solid"></div>
	<div class="box double"></div>
	<div class="box dashed"></div>
	<div class="box dotted"></div>
	<div class="box inset"></div>
	<div class="box outset"></div>
	<div class="box ridge"></div>
	<div class="box groove"></div>
</section>
```

- 单个边

```
border-top
border-right
border-left
border-bottom
```

- 半径

```
border-radius: 5px

单个值
border-top-left-radius
border-top-right-radius
border-bottom-left-radius
border-bottom-right-radius
```

### 阴影

```
box-shadow 实现阴影
支持如下值(必须按照如下顺序)
inset
x偏移
y偏移
模糊半径
扩散半径
颜色
```

### 轮廓

轮廓不占用任何空间，在分配给现有元素的空间之上绘制。常与伪类一起使用，删除或添加轮廓并不会对页面布局产生影响

```
定义一个轮廓时，可以指定如下特性
outline-color
outline-style
outline-width

可以简写为
outline: 1px solid black
```

### 背景

background属性是css中应用比较多，且比较重要的一个属性，它是负责给盒子设置背景图片和背景颜色的，它可以分解成如下几个设置项

`background:url(bgimage.gif) left center no-repeat #00FF00`。

| 属性                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| background-image    | 设置背景图片地址，需要用url()函数提供图片链接                |
| background-position | 使用图片作为背景图，当图片大于背景时，优先显示图片的哪一块。left right center top bottom center |
| background-repeat   | 当图片不足以覆盖DOM元素时，是否重复平铺                      |
| background-color    | 设置背景颜色，RGB,十六进制、颜色名均可                       |
| background-size     | 使用图片作为背景图，背景图片的大小                           |

## 注释与颜色

- 注释

```
/* 设置头部的样式 */
.header{
    width:960px;
    height:80px;
    background:gold;
}
```

- 颜色

```
颜色值主要有三种表示方法：
1、颜色名表示，比如：red 红色，gold 金色
2、rgb表示，比如：rgb(255,0,0)表示红色
3、16进制数值表示，如：#ff0000表示红色，可以简写成#f00
```

## 盒子模型

```
把元素叫做盒子，设置对应的样式分别为：盒子的宽度(width)、盒子的高度(height)、盒子的边框(border)、盒子内的内容和边框之间的间距(padding)、盒子与盒子之间的间距(margin)。

盒子宽度 = width + padding左右 + border左右
盒子高度 = height + padding上下 + border上下
```
### 垂直外边距合并 
```
当两个不浮动的元素，它们的垂直外边距相遇时，它们将形成一个外边距。合并后的外边距的高度等于两个发生合并的外边距的高度中的较大者，
实际开发中一般只设置margin-top来避开这种合并的问题，或者将元素浮动，也可以避开这种问题。
```
### margin-top 塌陷 
```
在两个不浮动的盒子嵌套时候，内部的盒子设置的margin-top会加到外边的盒子上，导致内部的盒子margin-top设置失败，解决方法如下：
1、外部盒子设置一个边框
2、外部盒子设置 overflow:hidden
3、外部盒子使用伪元素类 .clearfix:before{content:'';display:table;}
```
### 相关技巧
```
1、设置不浮动的元素相对于父级水平居中： margin:x auto;
2、margin负值让元素位移及边框合并
```

## 样式重置

```javascript
/*清除浮动，及解决margin-top塌陷*/
.clearfix:after,clearfix:before{
	content:'';
	display:table;
}
.clearfix:after{ clear:both;}
.clearfix{zoom:1;}
/* 清除标签默认的外边距和内边距 */
body,h1,h2,h3,h4,p,ul,input{
    margin:0px;
    padding:0px;
}
/*  将标题的文字设置为默认大小 */
h1,h2,h3,h4{
    font-size:100%;
}
/*  清除小圆点 */
ul{
    list-style:none;
}
/*  清除下划线 */
a{
    text-decoration:none;
}
/*  清除IE下，对图片做链接时，图片产生的边框 */
img{
    border:0px;
}
/*  清除默认的斜体 */
em,i{
    font-style:normal;
}
/* 清除加粗 */
b{ 
    font-weight:normal;
}
/*  清除input获取焦点时高亮的蓝色框 */
input{
    outline:none;
}
```

