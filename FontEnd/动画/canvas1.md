# Canvas1

[参考](https://www.runoob.com/w3cnote/html5-canvas-intro.html)

## 简介

`<canvas>` 是 HTML5 新增的，一个可以使用脚本(通常为JavaScript)在其中绘制图像的 HTML 元素。它可以用来制作照片集或者制作简单(也不是那么简单)的动画，甚至可以进行实时视频处理和渲染。

 它最初由苹果内部使用自己MacOS X WebKit推出，供应用程序使用像仪表盘的构件和 Safari 浏览器使用。 后来，有人通过Gecko内核的浏览器 (尤其是Mozilla和Firefox)，Opera和Chrome和超文本网络应用技术工作组建议为下一代的网络技术使用该元素。

 Canvas是由HTML代码配合高度和宽度属性而定义出的可绘制区域。JavaScript代码可以访问该区域，类似于其他通用的二维API，通过一套完整的绘图函数来动态生成图形。

 Mozilla 程序从 Gecko 1.8 (Firefox 1.5)开始支持` <canvas>`, Internet Explorer 从IE9开始`<canvas>` 。Chrome和Opera 9+ 也支持` <canvas>`。


## 基本使用

```
<canvas id="tutorial" width="300" height="300"></canvas>
```

### 元素

`<canvas>` 看起来和 `<img>` 标签一样，只是 `<canvas>` 只有两个可选的属性 `width、heigth` 属性，而没有 `src、alt` 属性。

如果不给 `<canvas>` 设置 `widht、height` 属性时，则默认 `width`为300、`height` 为 150，单位都是 `px`。也可以使用 `css` 属性来设置宽高，但是如宽高属性和初始比例不一致，他会出现扭曲。所以，建议永远不要使用 `css` 属性来设置 `<canvas>` 的宽高。

**替换内容**

由于某些较老的浏览器（尤其是 IE9 之前的 IE 浏览器）或者浏览器不支持 HTML 元素 `<canvas>`，在这些浏览器上你应该总是能展示替代内容。

支持 `<canvas>` 的浏览器会只渲染 `canvas` 标签，而忽略其中的替代内容。不支持 `<canvas>` 的浏览器则 会直接渲染替代内容。

用文本替换：

```
<canvas>
    你的浏览器不支持 canvas，请升级你的浏览器。
</canvas>
```

用 `<img>` 替换：

```
<canvas>
    <img src="./美女.jpg" alt=""> 
</canvas>
```

结束标签 `</canvas>` 不可省略。

与 `<img>` 元素不同，`<canvas>` 元素需要结束标签(`</canvas>`)。如果结束标签不存在，则文档的其余部分会被认为是替代内容，将不会显示出来。

### 渲染上下文

`<canvas>` 会创建一个固定大小的画布，会公开一个或多个**渲染上下文**(画笔)，使用**渲染上下文**来绘制和处理要展示的内容。上下文是是所有的绘制操作api的入口或者集合。

 Canvas自身无法绘制任何内容。Canvas的绘图是使用JavaScript操作的。

Context对象就是JavaScript操作Canvas的接口。

使用`[CanvasElement].getContext(‘2d’)`来获取2D绘图上下文。

```
var canvas = document.getElementById('tutorial');
//获得 2d 上下文对象
var ctx = canvas.getContext('2d');
```

### 检测支持性

```
var canvas = document.getElementById('tutorial');

if (canvas.getContext){
  var ctx = canvas.getContext('2d');
  // drawing code here
} else {
  // canvas-unsupported code here
}
```

### 代码模版


```html
<html>
<head>
    <title>Canvas tutorial</title>
    <style type="text/css">
        canvas {
            border: 1px solid black;
        }
    </style>
</head>
<canvas id="tutorial" width="300" height="300"></canvas>
</body>
<script type="text/javascript">
    function draw(){
        var canvas = document.getElementById('tutorial');
        if(!canvas.getContext) return;
      	var ctx = canvas.getContext("2d");
      	//开始代码
        
    }
    draw();
</script>
</html>
```
## 绘制形状

### 栅格和坐标空间

如下图所示，`canvas` 元素默认被网格所覆盖。通常来说网格中的一个单元相当于 `canvas` 元素中的一像素。栅格的起点为左上角，坐标为 (0,0) 。所有元素的位置都相对于原点来定位。所以图中蓝色方形左上角的坐标为距离左边（X 轴）x 像素，距离上边（Y 轴）y 像素，坐标为 (x,y)。

后面我们会涉及到坐标原点的平移、网格的旋转以及缩放等。

![img](https://www.runoob.com/wp-content/uploads/2018/12/Canvas_default_grid.png)

### 绘制矩形

`<canvas>` 只支持一种原生的图形绘制：**矩形**。所有其他图形都至少需要生成一种路径 (`path`)。不过，我们拥有众多路径生成的方法让复杂图形的绘制成为了可能。

canvast 提供了三种方法绘制矩形：

| 函数 | 描述 |
| ---- | ---- |
| `rect(x,y,width,height)` | 规划矩形的路径，没有填充和描边 |
| `fillRect(x, y, width, height)`     |  绘制一个填充的矩形    |
| `strokeRect(x, y, width, height)`    |  绘制一个矩形的边框    |
| `clearRect(x, y, widh, height)`    |  清除指定的矩形区域，然后这块区域会变的完全透明    |

参数

| 参数 | 描述 |
| ---- | ---- |
| `x, y`     |  指的是矩形的左上角的坐标。(相对于canvas的坐标原点)    |
| `width, height`     |  指的是绘制的矩形的宽和高    |


```javascript
function draw(){
    var canvas = document.getElementById('tutorial');
    if(!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.fillRect(10, 10, 100, 50);     // 绘制矩形，填充的默认颜色为黑色
    ctx.strokeRect(10, 70, 100, 50);   // 绘制矩形边框
    
}
draw();
```



![img](https://www.runoob.com/wp-content/uploads/2018/12/2400420933-5b74dd8f80306_articlex.png)

```
ctx.clearRect(15, 15, 50, 25);
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/2347163070-5b74dd8f813a6_articlex.png)

## 绘制路径 

图形的基本元素是路径。

路径是通过不同颜色和宽度的线段或曲线相连形成的不同形状的点的集合。

一个路径，甚至一个子路径，都是闭合的。

基本步骤

```
第一步：获得上下文=>canvasElem.getContext('2d');

第二步：开始路径规划=>ctx.beginPath()

第三步：移动起始点=>ctx.moveTo(x, y)

第四步：绘制线(矩形、圆形、图片...) =>ctx.lineTo(x, y)

第五步：闭合路径=>ctx.closePath();

第六步：绘制描边=>ctx.stroke();
```

下面是需要用到的方法：

| 函数 | 描述 |
| ---- | ---- |
|  `beginPath()`    | 新建一条路径，路径一旦创建成功，图形绘制命令被指向到路径上生成路径，核心的作用是将 不同绘制的形状进行隔离，可以进行分开样式设置和管理 |
| `closePath()` | 闭合路径，会自动把最后的线头和开始的线头连在一起。 |
| `moveTo(x, y)`     |  把画笔移动到指定的坐标`(x, y)`。相当于设置路径的起始点坐标。    |
|  `closePath()`    |  闭合路径之后，图形绘制命令又重新指向到上下文中    |
| `stroke()`     | 通过线条来绘制图形轮廓，路径只是草稿，真正绘制线必须执行stroke |
|   `fill()`    |  通过填充路径的内容区域生成实心的图形    |

### 绘制线段

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath(); //新建一条path
    ctx.moveTo(50, 50); //把画笔移动到指定的坐标
    ctx.lineTo(200, 50);  //绘制一条从当前位置到指定坐标(200, 50)的直线.
    //闭合路径。会拉一条从当前点到path起始点的直线。如果当前点与起始点重合，则什么都不做
    ctx.closePath();
    ctx.stroke(); //绘制路径。
}
draw();
```

### 绘制三角形边框

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(50, 50);
    ctx.lineTo(200, 50);
    ctx.lineTo(200, 200);
    ctx.closePath(); //虽然我们只绘制了两条线段，但是closePath会closePath，仍然是一个3角形
    ctx.stroke(); //描边。stroke不会自动closePath()
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/2106846415-5b74dd8f67000_articlex.png)

### 填充三角形

```javascript
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(50, 50);
    ctx.lineTo(200, 50);
    ctx.lineTo(200, 200);
   
    ctx.fill(); //填充闭合区域。如果path没有闭合，则fill()会自动闭合路径。
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/3457015746-5b74dd8f72860_articlex.png)

### 绘制圆弧

有两个方法可以绘制圆弧：

- 方法

```
arc(x, y, r, startAngle, endAngle, anticlockwise)
```

以`(x, y)`为圆心，以`r`为半径，从 `startAngle`弧度开始到`endAngle`弧度结束。`anticlosewise`是布尔值，`true`表示逆时针，`false`表示顺时针(默认是顺时针)。

注意：

1. 这里的度数都是弧度。

2. `0` 弧度是指的 `x` 轴正方形。

   ```
   radians=(Math.PI/180)*degrees   //角度转换成弧度
   ```
   
```
arcTo(x1, y1, x2, y2, radius)
```

根据给定的控制点和半径画一段圆弧，最后再以直线连接两个控制点。

`arcTo` 方法的说明：

这个方法可以这样理解。绘制的弧形是由两条切线所决定。

第 1 条切线：起始点和控制点1决定的直线。

第 2 条切线：控制点1 和控制点2决定的直线。

**其实绘制的圆弧就是与这两条直线相切的圆弧。**

- 圆弧案例

案例 1

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.arc(50, 50, 40, 0, Math.PI / 2, false);
    ctx.stroke();
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/3832141455-5b74dd8f658df_articlex.png)

案例 2

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.arc(50, 50, 40, 0, Math.PI / 2, false);
    ctx.stroke();
 
    ctx.beginPath();
    ctx.arc(150, 50, 40, 0, -Math.PI / 2, true);
    ctx.closePath();
    ctx.stroke();
 
    ctx.beginPath();
    ctx.arc(50, 150, 40, -Math.PI / 2, Math.PI / 2, false);
    ctx.fill();
 
    ctx.beginPath();
    ctx.arc(150, 150, 40, 0, Math.PI, false);
    ctx.fill();
 
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/2218794221-5b74dd8f43f98_articlex.png)

案例 3

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(50, 50);
      //参数1、2：控制点1坐标   参数3、4：控制点2坐标  参数4：圆弧半径
    ctx.arcTo(200, 50, 200, 200, 100);
    ctx.lineTo(200, 200)
    ctx.stroke();
    
    ctx.beginPath();
    ctx.rect(50, 50, 10, 10);
    ctx.rect(200, 50, 10, 10)
    ctx.rect(200, 200, 10, 10)
    ctx.fill()
}
draw();
```



![img](https://www.runoob.com/wp-content/uploads/2018/12/3556678928-5b74dd8f1bd2a_articlex.png)



### 绘制贝塞尔曲线

**什么是贝塞尔曲线**

贝塞尔曲线(Bézier curve)，又称贝兹曲线或贝济埃曲线，是应用于二维图形应用程序的数学曲线。

一般的矢量图形软件通过它来精确画出曲线，贝兹曲线由线段与节点组成，节点是可拖动的支点，线段像可伸缩的皮筋，我们在绘图工具上看到的钢笔工具就是来做这种矢量曲线的。

贝塞尔曲线是计算机图形学中相当重要的参数曲线，在一些比较成熟的位图软件中也有贝塞尔曲线工具如 PhotoShop 等。在 Flash4 中还没有完整的曲线工具，而在 Flash5 里面已经提供出贝塞尔曲线工具。

贝塞尔曲线于 1962，由法国工程师皮埃尔·贝塞尔（Pierre Bézier）所广泛发表，他运用贝塞尔曲线来为汽车的主体进行设计。贝塞尔曲线最初由Paul de Casteljau 于 1959 年运用 de Casteljau 演算法开发，以稳定数值的方法求出贝兹曲线。

一次贝塞尔曲线其实是一条直线

![img](https://www.runoob.com/wp-content/uploads/2018/12/240px-b_1_big.gif)

二次贝塞尔曲线

![img](https://www.runoob.com/wp-content/uploads/2018/12/b_2_big.gif)

![img](https://www.runoob.com/wp-content/uploads/2018/12/1544764428-5713-240px-BC3A9zier-2-big.svg-.png)

三次贝塞尔曲线

![img](https://www.runoob.com/wp-content/uploads/2018/12/b_3_big.gif)

![img](https://www.runoob.com/wp-content/uploads/2018/12/1544764428-2467-240px-BC3A9zier-3-big.svg-.png)

**绘制贝塞尔曲线**

绘制二次贝塞尔曲线:

```
quadraticCurveTo(cp1x, cp1y, x, y)
```

说明：

-  参数 1 和 2：控制点坐标
-  参数 3 和 4：结束点坐标

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(10, 200); //起始点
    var cp1x = 40, cp1y = 100;  //控制点
    var x = 200, y = 200; // 结束点
    //绘制二次贝塞尔曲线
    ctx.quadraticCurveTo(cp1x, cp1y, x, y);
    ctx.stroke();
 
    ctx.beginPath();
    ctx.rect(10, 200, 10, 10);
    ctx.rect(cp1x, cp1y, 10, 10);
    ctx.rect(x, y, 10, 10);
    ctx.fill();
 
}
draw();
```



![img](https://www.runoob.com/wp-content/uploads/2018/12/274915666-5b74dd8ecb2e2_articlex.png)

绘制三次贝塞尔曲线:

```
bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y)
```

说明：

-  参数 1 和 2：控制点 1 的坐标
-  参数 3 和 4：控制点 2 的坐标
-  参数 5 和 6：结束点的坐标

```
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(40, 200); //起始点
    var cp1x = 20, cp1y = 100;  //控制点1
    var cp2x = 100, cp2y = 120;  //控制点2
    var x = 200, y = 200; // 结束点
    //绘制二次贝塞尔曲线
    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x, y);
    ctx.stroke();
 
    ctx.beginPath();
    ctx.rect(40, 200, 10, 10);
    ctx.rect(cp1x, cp1y, 10, 10);
    ctx.rect(cp2x, cp2y, 10, 10);
    ctx.rect(x, y, 10, 10);
    ctx.fill();
 
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/3947786617-5b74dd8ec8678_articlex.png)

## 绘制文本

canvas 提供了两种方法来渲染文本:

- `fillText(text, x, y [, maxWidth])` 

在指定的 (x,y) 位置填充指定的文本，绘制的最大宽度是可选的。

- `strokeText(text, x, y [, maxWidth])` 

在指定的 (x,y) 位置绘制文本边框，绘制的最大宽度是可选的。

```python
var ctx;
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    ctx = canvas.getContext("2d");
    ctx.font = "100px sans-serif"
    ctx.fillText("天若有情", 10, 100);
    ctx.strokeText("天若有情", 10, 200)
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/404304980-5b74dd8e7499e_articlex.png)

**给文本添加样式**

`font = value `

当前我们用来绘制文本的样式。这个字符串使用和 CSS font 属性相同的语法。 默认的字体是 `10px sans-serif`。

`textAlign = value`

文本对齐选项。 可选的值包括：start, end, left, right or center。 默认值是 start。

`textBaseline = value`

基线对齐选项，可选的值包括：top, hanging, middle, alphabetic, ideographic, bottom。默认值是 alphabetic。

`direction = value` 

文本方向。可能的值包括：ltr, rtl, inherit。默认值是 inherit。

## 绘制图片


我们也可以在 `canvas` 上直接绘制图片。

### canvas创建图片

- 基本绘制

```
ctx.drawImage(img,x,y); 
```

参数
`img`：要绘制的 img 
`x,y`：绘制图片左上角坐标

> 注意：
> 考虑到图片是从网络加载，如果 `drawImage` 的时候图片还没有完全加载完成，则什么都不做，个别浏览器会抛异常。所以我们应该保证在 `img` 绘制完成之后再 `drawImage`。

```javascript
var img = new Image();   // 创建img元素
img.onload = function(){
  ctx.drawImage(img, 0, 0)
}
img.src = 'myImage.png'; // 设置图片源地址

```

### js创建img对象

方法一

```javascript
var img = document.getElementById("imgId");
```

方法二

```javascript
var img = new Image();   // 创建一个<img>元素 
img.src = 'myImage.png'; // 设置图片源地址
img.alt = "文本信息";
img.onload = function(){
  // 图片加载完成后，执行此方法
}
```

第一张图片就是页面中的 `<img>` 标签：

![img](https://www.runoob.com/wp-content/uploads/2018/12/2255709523-5b74dd8eb033e_articlex.png)

### 缩放

`drawImage()` 也可以再添加两个参数：

```
drawImage(image, x, y, width, height)
```

这个方法多了 2 个参数：`width` 和 `height`，这两个参数用来控制 当像 canvas 画入时应该缩放的大小。

```
ctx.drawImage(img, 0, 0, 400, 200)
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/327614951-5b74dd8e71ab1_articlex.png)

### 裁剪

```
drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
```

第一个参数和其它的是相同的，都是一个图像或者另一个 canvas 的引用。

其他 8 个参数：

前 4 个是定义图像源的切片位置和大小，后 4 个则是定义切片的目标显示位置和大小。

![img](https://www.runoob.com/wp-content/uploads/2018/12/2106688680-54566fa3d81dc_articlex.jpeg)

## 