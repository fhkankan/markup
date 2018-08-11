# CSS

## 安装

```
<link rel="stylesheet" href="https://cdn.bootcss.com/bootstrap/3.3.7/css/bootstrap.min.css">  
  <script src="https://cdn.bootcss.com/jquery/2.1.1/jquery.min.js"></script>
  <script src="https://cdn.bootcss.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
```

## 概览

文档开头

```
<!DOCTYPE html>
<html>
....
</html>
```

移动优先

```
// 在网页的 head 之中添加 viewport meta 标签
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```

响应式图像

```
<img src="..." class="img-responsive" alt="响应式图像">
```

容器

```
<div class="container">
  ...
</div>
```

## 网格系统

```
Bootstrap 包含了一个响应式的、移动设备优先的、不固定的网格系统，可以随着设备或视口大小的增加而适当地扩展到 12 列。它包含了用于简单的布局选项的预定义类，也包含了用于生成更多语义布局的功能强大的混合类

内容
决定什么是最重要的。
布局
优先设计更小的宽度。
基础的 CSS 是移动设备优先，媒体查询是针对于平板电脑、台式电脑。
渐进增强
随着屏幕大小的增加而添加元素。

响应式网格系统随着屏幕或视口（viewport）尺寸的增加，系统会自动分为最多12列
```

- 原理

```
网格系统通过一系列包含内容的行和列来创建页面布局。下面列出了 Bootstrap 网格系统是如何工作的：
行必须放置在 .container class 内，以便获得适当的对齐（alignment）和内边距（padding）。
使用行来创建列的水平组。
内容应该放置在列内，且唯有列可以是行的直接子元素。
预定义的网格类，比如 .row 和 .col-xs-4，可用于快速创建网格布局。LESS 混合类可用于更多语义布局。
列通过内边距（padding）来创建列内容之间的间隙。该内边距是通过 .rows 上的外边距（margin）取负，表示第一列和最后一列的行偏移。
网格系统是通过指定您想要横跨的十二个可用的列来创建的。例如，要创建三个相等的列，则使用三个 .col-xs-4
```

- 媒体查询

媒体查询是非常别致的"有条件的 CSS 规则"。它只适用于一些基于某些规定条件的 CSS。如果满足那些条件，则应用相应的样式。

Bootstrap 中的媒体查询允许您基于视口大小移动、显示并隐藏内容。下面的媒体查询在 LESS 文件中使用，用来创建 Bootstrap 网格系统中的关键的分界点阈值。

```
/* 超小设备（手机，小于 768px） */
/* Bootstrap 中默认情况下没有媒体查询 */

/* 小型设备（平板电脑，768px 起） */
@media (min-width: @screen-sm-min) { ... }

/* 中型设备（台式电脑，992px 起） */
@media (min-width: @screen-md-min) { ... }

/* 大型设备（大台式电脑，1200px 起） */
@media (min-width: @screen-lg-min) { ... }
```

- 网格选项

下表总结了 Bootstrap 网格系统如何跨多个设备工作：

|              | 超小设备手机（<768px）         | 小型设备平板电脑（≥768px）     | 中型设备台式电脑（≥992px）     | 大型设备台式电脑（≥1200px）    |
| ------------ | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| 网格行为     | 一直是水平的                   | 以折叠开始，断点以上是水平的   | 以折叠开始，断点以上是水平的   | 以折叠开始，断点以上是水平的   |
| 最大容器宽度 | None (auto)                    | 750px                          | 970px                          | 1170px                         |
| Class 前缀   | **.col-xs-**                   | **.col-sm-**                   | **.col-md-**                   | **.col-lg-**                   |
| 列数量和     | 12                             | 12                             | 12                             | 12                             |
| 最大列宽     | Auto                           | 60px                           | 78px                           | 95px                           |
| 间隙宽度     | 30px （一个列的每边分别 15px） | 30px （一个列的每边分别 15px） | 30px （一个列的每边分别 15px） | 30px （一个列的每边分别 15px） |
| 可嵌套       | Yes                            | Yes                            | Yes                            | Yes                            |
| 偏移量       | Yes                            | Yes                            | Yes                            | Yes                            |
| 列排序       | Yes                            | Yes                            | Yes                            | Yes                            |

基本的网格结构

```
<div class="container">
   <div class="row">
      <div class="col-*-*"></div>
      <div class="col-*-*"></div>      
   </div>
   <div class="row">...</div>
</div>
<div class="container">....
```

## 排版

Bootstrap 使用 Helvetica Neue、 Helvetica、 Arial 和 sans-serif 作为其默认的字体栈。

使用 Bootstrap 的排版特性，您可以创建标题、段落、列表及其他内联元素。

- 标题

```
<h1>我是标题1 h1</h1>
<h2>我是标题2 h2</h2>
<h3>我是标题3 h3</h3>
<h4>我是标题4 h4</h4>
<h5>我是标题5 h5</h5>
<h6>我是标题6 h6</h6>
```

- 内联子标题

如果需要向任何标题添加一个内联子标题，只需要简单地在元素两旁添加` <small>`，或者添加` .small class`，这样子您就能得到一个字号更小的颜色更浅的文本，如下面实例所示：

```
<h1>我是标题1 h1. <small>我是副标题1 h1</small></h1>
<h2>我是标题2 h2. <small>我是副标题2 h2</small></h2>
<h3>我是标题3 h3. <small>我是副标题3 h3</small></h3>
<h4>我是标题4 h4. <small>我是副标题4 h4</small></h4>
<h5>我是标题5 h5. <small>我是副标题5 h5</small></h5>
<h6>我是标题6 h6. <small>我是副标题6 h6</small></h6>
```

