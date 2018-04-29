[TOC]
# HTML














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

