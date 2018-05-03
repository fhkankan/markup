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

```
1、标签选择器
标签选择器，此种选择器影响范围大，一般用来做一些通用设置，或用在层级选择器中。
eg: div{}
2、类选择器
通过类名来选择元素，一个类可应用于多个元素，一个元素上也可以使用多个类，应用灵活，可复用，是css中应用最多的一种选择器。
eg: .className{}
3、层级选择器
要应用在标签嵌套的结构中，层级选择器，是结合上面的两种选择器来写的选择器,它可与标签选择器结合使用，减少命名，同时也可以通过层级，限制样式的作用范围。
eg: .className a{}
4、id选择器
通过id名来选择元素，元素的id名称不能重复，所以一个样式设置项只能对应于页面上一个元素，不能复用，id名一般给程序使用，所以不推荐使用id作为选择器。
5、组选择器
多个选择器，如果有同样的样式设置，可以使用组选择器。
eg: .box1,.box2,.box3{width:30px;height:30px}
6、伪类及伪元素选择器
常用的伪类选择器有hover，表示鼠标悬浮在元素上时的状态
伪元素选择器有before和after,它们可以通过样式在元素中插入内容。
eg:
.box1:hover{color:red}
.box2:before{content:'行首文字';}
.box3:after{content:'行尾文字';}
```

## 属性

### 布局

```
width 设置元素(标签)的宽度，如：width:100px;

height 设置元素(标签)的高度，如：height:200px;

background 设置元素背景色或者背景图片，如:background:gold; 
可分解为：
background-image 设置背景图片地址
background-position 设置背景图片的位置		left right center top bottom center 1px
background-repeat 设置背景图片如何重复平铺
background-color 设置背景颜色
合并为：
background:url(bgimage.gif) left center no-repeat #00FF00

float 设置元素浮动，浮动可以让块元素排列在一行，浮动分为左浮动：float:left; 右浮动：float:right;

border 设置元素四周的边框，如：border:1px solid black; 设置元素四周边框是1像素宽的黑色实线

border-radius	设置四角圆弧半径

padding 设置元素包含的内容和元素边框的距离，也叫内边距，如padding:20px;

margin 设置元素和外界的距离，也叫外边距，如margin:20px;

border，padding，margin均可分别设置四个边top,right,bottom，left

over-flow
z-index			设置优先级

opacity			设置透明度
filter:alpha(opacity(数字))	兼容IE
```

### 文本

```
font-size 设置文字的大小，如：font-size:12px;
font-family 设置文字的字体，如：font-family:'微软雅黑';为了避免中文字不兼容，一般写成：font-family:'Microsoft Yahei';
font-weight 设置文字是否加粗，如：font-weight:bold; 设置加粗 font-weight:normal 设置不加粗
font-style 设置字体是否倾斜，如：font-style:'normal'; 设置不倾斜，font-style:'italic';设置文字倾斜
font 同时设置文字的几个属性，写的顺序有兼容问题，建议按照如下顺序写： font：是否加粗 字号/行高 字体；如： font:normal 12px/36px '微软雅黑';

text-decoration 设置文字的下划线，如：text-decoration:none; 将文字下划线去掉
text-align 设置文字水平对齐方式，如text-align:center 设置文字水平居中
text-indent 设置文字首行缩进

color 设置文字的颜色，如： color:red;
line-height 设置文字的行高，如：line-height:24px; 表示文字高度加上文字上下的间距是24px，也就是每一行占有的高度是24px
list-style 设置无序列表中的小圆点，一般把它设为"无"，如：list-style:none
```

### 溢出

```
当子元素的尺寸超过父元素的尺寸时，需要设置父元素显示溢出的子元素的方式，设置的方法是通过overflow属性来设置。

overflow的设置项： 
1、visible 默认值。内容不会被修剪，会呈现在元素框之外。
2、hidden 内容会被修剪，并且其余内容是不可见的，此属性还有清除浮动、清除margin-top塌陷的功能。
3、scroll 内容会被修剪，但是浏览器会显示滚动条以便查看其余的内容。
4、auto 如果内容被修剪，则浏览器会显示滚动条以便查看其余的内容。
```

## 注释

```
/* 设置头部的样式 */
.header{
    width:960px;
    height:80px;
    background:gold;
}
```

## 颜色

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

## 块元素类型及特性

### 块元素特性

块元素，也可以称为行元素

布局中常用的标签如：div、p、ul、li、h1~h6等等都是块元素

它在布局中的行为：

```
支持全部的样式
如果没有设置宽度，默认的宽度为父级宽度100%
盒子占据一行、即使设置了宽度
```

### 包含默认样式的块元素

上面讲的块标签中，有些标签是包含默认的样式的，这个含默认样式的有

- p标签：含有默认外边距
- ul：含有默认外边距和内边距，以及条目符号
- h1~h6标签：含有默认的外边距和字体大小
- body标签：含有默认的外边距

实际开发中，我们会把这些默认的样式在样式定义开头清除掉，清除掉这些默认样式，方便我们写自己的定义的样式，这种做法叫样式重置。

## 内联元素类型及特性

### 内联元素特性

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

### 解决内联元素间隙的方法

1、去掉内联元素之间的换行
2、将内联元素的父级设置font-size为0，内联元素自身再设置font-size

### 其他内联元素

- em 标签 行内元素，表示语气中的强调词
- i 标签 行内元素，表示专业词汇
- b 标签 行内元素，表示文档中的关键字或者产品名
- strong 标签 行内元素，表示非常重要的内容

### 包含默认样式的内联元素

- a标签：含有的下划线以及文字颜色
- em、i标签：文字默认为斜体
- b、strong标签：文字默认加粗


## 内联块元素类型及特性

内联块元素，也叫行内块元素，是新增的元素类型，现有元素没有归于此类别的，img和input元素的行为类似这种元素，但是也归类于内联元素，我们可以用display属性将块元素或者内联元素转化成这种元素。它们在布局中表现的行为：

```
支持全部样式
如果没有设置宽高，宽高由内容决定
盒子并在一行
代码换行，盒子会产生间距
子元素是内联块元素，父元素可以用text-align属性设置子元素水平对齐方式。
```

## 三种元素转换

可以通过display属性来相互转化：

display属性是用来设置元素的类型及隐藏的，常用的属性有：

- none 元素隐藏且不占位置
- block 元素以块元素显示
- inline 元素以内联元素显示
- inline-block 元素以内联块元素显示

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

## 浮动

### 浮动特性

```
浮动元素有左浮动(float:left)和右浮动(float:right)两种
浮动的元素会向左或向右浮动，碰到父元素边界、其他元素才停下来
相邻浮动的块元素可以并在一行，超出父级宽度就换行
浮动让行内元素或块元素转化为有浮动特性的行内块元素(此时不会有行内块元素间隙问题)
父元素如果没有设置尺寸(一般是高度不设置)，父元素内整体浮动的子元素无法撑开父元素，父元素需要清除浮动
```

### 清除浮动

- 父级上增加属性overflow：hidden

- 在最后一个子元素的后面加一个空的div，给它样式属性 clear:both（不推荐）

- 使用成熟的清浮动样式类，clearfix

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

## 定位

### 文档流 

文档流，是指盒子按照html标签编写的顺序依次从上到下，从左到右排列，块元素占一行，行内元素在一行之内从左到右排列，先写的先排列，后写的排在后面，每个盒子都占据自己的位置。

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

**新增相关样式属性**

```
/* 设置元素圆角,将元素四个角设置4px半径的圆角 */
border-radius:4px;

/* 设置元素透明度,将元素透明度设置为0.3，此属性需要加一个兼容IE的写法 */
opacity:0.3;
/* 兼容IE */
filter:alpha(opacity=30);
```

## 背景

ackground属性是css中应用比较多，且比较重要的一个属性，它是负责给盒子设置背景图片和背景颜色的，background是一个复合属性，它可以分解成如下几个设置项：

- background-image 设置背景图片地址
- background-position 设置背景图片的位置
- background-repeat 设置背景图片如何重复平铺
- background-color 设置背景颜色

可以将上面的属性设置用background属性合并成一句：
“background:url(bgimage.gif) left center no-repeat #00FF00”

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

