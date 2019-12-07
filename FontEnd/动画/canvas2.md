# canvas2

## 变换

### 位移画布

```
translate(x, y)
```

用来移动 `canvas` 的**原点**到指定的位置

`translate` 方法接受两个参数。`x` 是左右偏移量，`y` 是上下偏移量，如右图所示。

在做变形之前先保存状态是一个良好的习惯。大多数情况下，调用 `restore` 方法比手动恢复原先的状态要简单得多。又如果你是在一个循环中做位移但没有保存和恢复 `canvas` 的状态，很可能到最后会发现怎么有些东西不见了，那是因为它很可能已经超出 `canvas` 范围以外了。

 注意：`translate` 移动的是 `canvas` 的坐标原点(坐标变换)。

![img](https://www.runoob.com/wp-content/uploads/2018/12/829832336-5b74dd8e3ad9a_articlex.png)

```
var ctx;
function draw(){
    var canvas = document.getElementById('tutorial1');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.save(); //保存坐原点平移之前的状态
    ctx.translate(100, 100);
    ctx.strokeRect(0, 0, 100, 100)
    ctx.restore(); //恢复到最初状态
    ctx.translate(220, 220);
    ctx.fillRect(0, 0, 100, 100)
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/1230266743-5b74dd8e3b0ce_articlex.png)

### 旋转

```
rotate(angle)
```

旋转坐标轴。

这个方法只接受一个参数：旋转的角度(angle)，它是顺时针方向的，以弧度为单位的值。若需将角度转换为弧度，使用`degrees*Math.PI/180`转换

 旋转的中心是坐标原点。

![img](https://www.runoob.com/wp-content/uploads/2018/12/3322150878-5b74dd8e2b6a4_articlex.png)

```javascript
var ctx;
function draw(){
  var canvas = document.getElementById('tutorial1');
  if (!canvas.getContext) return;
  var ctx = canvas.getContext("2d");

  ctx.fillStyle = "red";
  ctx.save();

  ctx.translate(100, 100);
  ctx.rotate(Math.PI / 180 * 45);
  ctx.fillStyle = "blue";
  ctx.fillRect(0, 0, 100, 100);
  ctx.restore();

  ctx.save();
  ctx.translate(0, 0);
  ctx.fillRect(0, 0, 50, 50)
  ctx.restore();
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/1819968878-5b74dd8e1e770_articlex.png)

### 缩放

```
scale(x, y)
```

我们用它来增减图形在 `canvas` 中的像素数目，对形状，位图进行缩小或者放大。

`scale`方法接受两个参数。`x,y` 分别是横轴和纵轴的缩放因子，它们都必须是正值。值比 1.0 小表示缩 小，比 1.0 大则表示放大，值为 1.0 时什么效果都没有。

 默认情况下，`canvas` 的 1 单位就是 1 个像素。举例说，如果我们设置缩放因子是 0.5，1 个单位就变成对应 0.5 个像素，这样绘制出来的形状就会是原先的一半。同理，设置为 2.0 时，1 个单位就对应变成了 2 像素，绘制的结果就是图形放大了 2 倍。

### 变换矩阵

```
transform(a, b, c, d, e, f)
```
变形矩阵

![img](https://www.runoob.com/wp-content/uploads/2018/12/2958376259-5b74dd8e15192_articlex.png)

| name | Desc |
| ---- | ---- |
| `a(m11)`   |  Horizontal scaling    |
| `b(m12)`   |  Horizontal skewing    |
| `c(m21)`   |  Vertical skewing    |
| `d(m22)`   |  Vertical scaling    |
| `e(dx)`   |   Horizontal moving   |
| `f(dy)`   |   Vertical moving   |

```javascript
var ctx;
function draw(){
    var canvas = document.getElementById('tutorial1');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.transform(1, 1, 0, 1, 0, 0);
    ctx.fillRect(0, 0, 100, 100);
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/489430190-5b74dd8e17ad2_articlex.png)

------

## 合成

在前面的所有例子中、，我们总是将一个图形画在另一个之上，对于其他更多的情况，仅仅这样是远远不够的。比如，对合成的图形来说，绘制顺序会有限制。不过，我们可以利用 globalCompositeOperation 属性来改变这种状况。

```
globalCompositeOperation = type
```

```
var ctx;
function draw(){
        var canvas = document.getElementById('tutorial1');
        if (!canvas.getContext) return;
        var ctx = canvas.getContext("2d");
        
        ctx.fillStyle = "blue";
        ctx.fillRect(0, 0, 200, 200);

        ctx.globalCompositeOperation = "source-over"; //全局合成操作
        ctx.fillStyle = "red";
        ctx.fillRect(100, 100, 200, 200);
    }
    draw();

</script>

```

**注**：下面的展示中，蓝色是原有的，红色是新的。

type 是下面 13 种字符串值之一：

### `source-over`

这是默认设置，新图像会覆盖在原有图像。

![img](https://www.runoob.com/wp-content/uploads/2018/12/1858023544-5b74dd8e0813d.png)



### `source-in`

仅仅会出现新图像与原来图像重叠的部分，其他区域都变成透明的。(包括其他的老图像区域也会透明)

![img](https://www.runoob.com/wp-content/uploads/2018/12/2183873141-5b74dd8e02a4a_articlex.png)

### `source-out`

仅仅显示新图像与老图像没有重叠的部分，其余部分全部透明。(老图像也不显示)

![img](https://www.runoob.com/wp-content/uploads/2018/12/2402253130-5b74dd8dd7621_articlex.png)

### `source-atop`

新图像仅仅显示与老图像重叠区域。老图像仍然可以显示。

![img](https://www.runoob.com/wp-content/uploads/2018/12/1206278247-5b74dd8dd9036_articlex.png)

### `destination-over`

新图像会在老图像的下面。

![img](https://www.runoob.com/wp-content/uploads/2018/12/2492190378-5b74dd8dca608_articlex.png)

### `destination-in`

仅仅新老图像重叠部分的老图像被显示，其他区域全部透明。

![img](https://www.runoob.com/wp-content/uploads/2018/12/284693590-5b74dd8dc7f3e_articlex.png)

### `destination-out`

仅仅老图像与新图像没有重叠的部分。 注意显示的是老图像的部分区域。

![img](https://www.runoob.com/wp-content/uploads/2018/12/1921976761-5b74dd8daba2d_articlex.png)

### `destination-atop`

老图像仅仅仅仅显示重叠部分，新图像会显示在老图像的下面。

![img](https://www.runoob.com/wp-content/uploads/2018/12/4055109887-5b74dd8db283c_articlex.png)

### `lighter`

新老图像都显示，但是重叠区域的颜色做加处理。

![img](https://www.runoob.com/wp-content/uploads/2018/12/1200224117-5b74dd8d9453e_articlex.png)

### `darken`

保留重叠部分最黑的像素。(每个颜色位进行比较，得到最小的)

```
blue: #0000ff
red: #ff0000
```

所以重叠部分的颜色：`#000000`。

![img](https://www.runoob.com/wp-content/uploads/2018/12/3835256030-5b74dd8d92ba5_articlex.png)

### `lighten`

保证重叠部分最量的像素。(每个颜色位进行比较，得到最大的)

```
blue: #0000ff
red: #ff0000
```

所以重叠部分的颜色：`#ff00ff`。

![img](https://www.runoob.com/wp-content/uploads/2018/12/1617768463-5b74dd8d99843_articlex.png)

### `xor`

重叠部分会变成透明。

![img](https://www.runoob.com/wp-content/uploads/2018/12/2521026104-5b74dd8d6abd6_articlex.png)

### `copy`

只有新图像会被保留，其余的全部被清除(边透明)。

![img](https://www.runoob.com/wp-content/uploads/2018/12/2454891415-5b74dd8d67aec_articlex.png)

## canvas样式

### 设置填充和描边颜色

 在前面的绘制矩形章节中，只用到了默认的线条和颜色。

 如果想要给图形上色，有两个重要的属性可以做到。

`fillStyle = color` 设置图形的填充颜色

`strokeStyle = color` 设置图形轮廓的颜色

>  备注：
>
>  color 可以是表示 css 颜色值的字符串、渐变对象或者图案对象。
>
>  默认情况下，线条和填充颜色都是黑色。
>
>  一旦您设置了 strokeStyle 或者 fillStyle 的值，那么这个新值就会成为新绘制的图形的默认值。如果你要给每个图形上不同的颜色，你需要重新设置 fillStyle 或 strokeStyle 的值。

fillStyle

```javascript
function draw(){
  var canvas = document.getElementById('tutorial');
  if (!canvas.getContext) return;
  var ctx = canvas.getContext("2d");
  for (var i = 0; i < 6; i++){
    for (var j = 0; j < 6; j++){
      ctx.fillStyle = 'rgb(' + Math.floor(255 - 42.5 * i) + ',' +
        Math.floor(255 - 42.5 * j) + ',0)';
      ctx.fillRect(j * 50, i * 50, 50, 50);
    }
  }
}
draw();
```



![img](https://www.runoob.com/wp-content/uploads/2018/12/2505008676-5b74dd8ebad41_articlex.png)

strokeStyle

```python
<script type="text/javascript">
    function draw(){
        var canvas = document.getElementById('tutorial');
        if (!canvas.getContext) return;
        var ctx = canvas.getContext("2d");
        for (var i = 0; i < 6; i++){
            for (var j = 0; j < 6; j++){
                ctx.strokeStyle = `rgb(${randomInt(0, 255)},${randomInt(0, 255)},${randomInt(0, 255)})`;
                ctx.strokeRect(j * 50, i * 50, 40, 40);
            }
        }
    }
    draw();
    
    function randomInt(from, to){
        return parseInt(Math.random() * (to - from + 1) + from);
    }

</script>
```



![img](https://www.runoob.com/wp-content/uploads/2018/12/3288535670-5b74dd8ea12d9_articlex.png)

### 设置阴影

类比于CSS3的阴影，性能较差，少用

`shadowColor`： 设置或返回用于阴影的颜色

`shadowBlur`： 设置或返回用于阴影的模糊级别,大于1的正整数，数值越高，模糊程度越大

`shadowOffsetX`： 设置或返回阴影距形状的水平距离

`shadowOffsetY`： 设置或返回阴影距形状的垂直距离

```javascript
ctx.fillStyle = "rgba(255,0,0, .9)"

ctx.shadowColor = "teal";

ctx.shadowBlur = 10;

ctx.shadowOffsetX = 10;

ctx.shadowOffsetY = 10;

ctx.fillRect(100, 100, 100, 100);
```

### 设置透明度

`ctx.globalAlpha = transparencyValue`: 这个属性影响到 canvas 里所有图形的透明度，有效的值范围是 0.0 （完全透明）到 1.0（完全不透明），默认是 1.0。

 `globalAlpha` 属性在需要绘制大量拥有相同透明度的图形时候相当高效。不过，我认为使用`rgba()`设置透明度更加好一些。

### 线条样式

- `lineWidth=value`

线宽。只能是正值。默认是 1.0。

起始点和终点的连线为中心，**上下各占线宽的一半**。

```
ctx.beginPath();
ctx.moveTo(10, 10);
ctx.lineTo(100, 10);
ctx.lineWidth = 10;
ctx.stroke();
 
ctx.beginPath();
ctx.moveTo(110, 10);
ctx.lineTo(160, 10)
ctx.lineWidth = 20;
ctx.stroke()
```



![img](https://www.runoob.com/wp-content/uploads/2018/12/3410060825-5b74dd8ea12d9_articlex.png)

- `lineCap = type`

线条末端样式。

共有 3 个值：

1. `butt`：线段末端以方形结束

2. `round`：线段末端以圆形结束

3. `square`：线段末端以方形结束，但是增加了一个宽度和线段相同，高度是线段厚度一半的矩形区域。

```javascript
var lineCaps = ["butt", "round", "square"];
    

for (var i = 0; i < 3; i++){
       ctx.beginPath();
    ctx.moveTo(20 + 30 * i, 30);
       ctx.lineTo(20 + 30 * i, 100);
    ctx.lineWidth = 20;
       ctx.lineCap = lineCaps[i];
    ctx.stroke();
   }

   ctx.beginPath();
ctx.moveTo(0, 30);
   ctx.lineTo(300, 30);

   ctx.moveTo(0, 100);
ctx.lineTo(300, 100)
    
ctx.strokeStyle = "red";
   ctx.lineWidth = 1;
ctx.stroke();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/3380216230-5b74dd8e97e85_articlex.png)

- `lineJoin = type`

同一个 path 内，设定线条与线条间接合处的样式。

共有 3 个值：

1. `round`: 通过填充一个额外的，圆心在相连部分末端的扇形，绘制拐角的形状。 圆角的半径是线段的宽度。

2. `bevel `:在相连部分的末端填充一个额外的以三角形为底的区域， 每个部分都有各自独立的矩形拐角。

3. `miter`:(默认) 通过延伸相连部分的外边缘，使其相交于一点，形成一个额外的菱形区域。

```javascript
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
 
    var lineJoin = ['round', 'bevel', 'miter'];
    ctx.lineWidth = 20;
 
    for (var i = 0; i < lineJoin.length; i++){
        ctx.lineJoin = lineJoin[i];
        ctx.beginPath();
        ctx.moveTo(50, 50 + i * 50);
        ctx.lineTo(100, 100 + i * 50);
        ctx.lineTo(150, 50 + i * 50);
        ctx.lineTo(200, 100 + i * 50);
        ctx.lineTo(250, 50 + i * 50);
        ctx.stroke();
    }
 
}
draw();
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/1584506777-5b74dd8e82768_articlex.png)

### 虚线

用 setLineDash 方法和 lineDashOffset 属性来制定虚线样式。 setLineDash 方法接受一个数组，来指定线段与间隙的交替；lineDashOffset属性设置起始偏移量。

```javascript
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
 
    ctx.setLineDash([20, 5]);  // [实线长度, 间隙长度]
    ctx.lineDashOffset = -0;
    ctx.strokeRect(50, 50, 210, 210);
}
draw();
```

备注： `getLineDash() `返回一个包含当前虚线样式，长度为非负偶数的数组。

1. ![img](https://www.runoob.com/wp-content/uploads/2018/12/2805401035-5b74dd8e6833c_articlex.png)

### 渐变样式

- 线性渐变

一般不用，都是用图片代替，canvas绘制图片效率更高。

线性渐变可以用于矩形、圆形、文字等颜色样式

线性渐变是一个对象

```
ctx.createLinearGradient(x0,y0,x1,y1); 
//参数：x0,y0起始坐标，x1,y1结束坐标
```

例如：

```
//创建线性渐变的对象，

var grd=ctx.createLinearGradient(0,0,170,0);

grd.addColorStop(0,"black");  //添加一个渐变颜色，第一个参数介于0.0与1.0之间的值，表示渐变中开始与结束之间的位置。

grd.addColorStop(1,"white");  //添加一个渐变颜色

ctx.fillStyle =grd;      //关键点，把渐变设置到 填充的样式
```

- 圆形渐变(径向渐变)

创建放射状/圆形渐变对象。可以填充文本、形状等

```
context.createRadialGradient(x0,y0,r0,x1,y1,r1);
```

参数详解：

```
ox0:渐变的开始圆的x坐标
oy0:渐变的开始圆的y坐标
or0:开始圆的半径
ox1:渐变的结束圆的x坐标
oy1:渐变的结束圆的y坐标
or1:结束圆的半径
```

示例

```javascript
var rlg = ctx.createRadialGradient(300,300,10,300,300,200);

rlg.addColorStop(0, 'teal');   //添加一个渐变颜色

rlg.addColorStop(.4, 'navy');

rlg.addColorStop(1, 'purple');

ctx.fillStyle = rlg;//设置 填充样式为延续渐变的样式

ctx.fillRect(100, 100, 500, 500);
```

### 绘制背景图

```
ctx.createPattern(img,repeat)
```

在指定的方向内重复指定的元素

参数：

参数1: 设置平铺背景的图片

```
image： 规定要使用的图片、画布或视频元素。
```

参数2: 背景平铺的方式。

```
repeat： 默认。该模式在水平和垂直方向重复。
repeat-x： 该模式只在水平方向重复。
repeat-y： 该模式只在垂直方向重复。
no-repeat： 该模式只显示一次（不重复）。
```

示例

```javascript
var ctx=c.getContext("2d");

var img=document.getElementById("lamp");

var pat=ctx.createPattern(img,"repeat");

ctx.rect(0,0,150,100);

ctx.fillStyle=pat;//把背景图设置给填充的样式

ctx.fill();
```

## 绘制环境的保存和恢复

保存和还原是绘制复杂图形时必不可少的操作。

`Canvas` 的状态就是当前画面应用的所有样式和变形的一个快照。

`save` 和 `restore` 方法是用来保存和恢复 `canvas` 状态的，都没有参数。

- `save()` ：

Canvas状态存储在栈中，每当save()方法被调用后，当前的状态就被推送到栈中保存。

一个绘画状态包括：

1. 当前应用的变形（即移动，旋转和缩放）

2. `strokeStyle`, `fillStyle`, `globalAlpha`, `lineWidth`, `lineCap`, `lineJoin`, `miterLimit`, `shadowOffsetX`, `shadowOffsetY`, `shadowBlur`, `shadowColor`, `globalCompositeOperation 的值`

3. 当前的裁切路径（`clipping path`）

可以调用任意多次 `save`方法(类似数组的 `push()`)。

- `restore()`：

每一次调用 restore 方法，上一个保存的状态就从栈中弹出，所有设定都恢复(类似数组的 `pop()`)。

```javascript
var ctx;
function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");

    ctx.fillRect(0, 0, 150, 150);   // 使用默认设置绘制一个矩形
    ctx.save();                  // 保存默认状态

    ctx.fillStyle = 'red'       // 在原有配置基础上对颜色做改变
    ctx.fillRect(15, 15, 120, 120); // 使用新的设置绘制一个矩形

    ctx.save();                  // 保存当前状态
    ctx.fillStyle = '#FFF'       // 再次改变颜色配置
    ctx.fillRect(30, 30, 90, 90);   // 使用新的配置绘制一个矩形

    ctx.restore();               // 重新加载之前的颜色状态
    ctx.fillRect(45, 45, 60, 60);   // 使用上一次的配置绘制一个矩形

    ctx.restore();               // 加载默认颜色配置
    ctx.fillRect(60, 60, 30, 30);   // 使用加载的配置绘制一个矩形
}
draw();
```

## 画布限定区域

```
clip()
```

把已经创建的路径转换成裁剪路径。

裁剪路径的作用是遮罩。只显示裁剪路径内的区域，裁剪路径外的区域会被隐藏。

**注意：**`clip() `只能遮罩在这个方法调用之后绘制的图像，如果是 clip() 方法调用之前绘制的图像，则无法实现遮罩。

![img](https://www.runoob.com/wp-content/uploads/2018/12/2023283460-5b74dd8d67aec_articlex.png)

```javascript
var ctx;
function draw(){
    var canvas = document.getElementById('tutorial1');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");

    ctx.beginPath();
    ctx.arc(20,20, 100, 0, Math.PI * 2);
    ctx.clip();
    
    ctx.fillStyle = "pink";
    ctx.fillRect(20, 20, 100,100);
}
draw();
```

## 画布保存base64编码内容

把canvas绘制的内容输出成base64内容。
```
canvas.toDataURL(type, encoderOptions);
```

参数说明：
```
otype，设置输出的类型，比如image/png image/jpeg等

oencoderOptions：0-1之间的数字，用于标识输出图片的质量，1表示无损压缩，类型为：image/jpeg或者image/webp才起作用。
```

示例

```
var canvas = document.getElementById("canvas");

var dataURL = canvas.toDataURL();

console.log(dataURL);

// "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNby

// blAAAADElEQVQImWNgoBMAAABpAAFEI8ARAAAAAElFTkSuQmCC"

var img = document.querySelector("#img-demo");//拿到图片的dom对象

img.src = canvas.toDataURL("image/png");    //将画布的内容给图片标签显示
```
## 画布渲染画布
```
context.drawImage(img,x,y);
```
`img`参数也可以是画布，也就是把一个画布整体的渲染到另外一个画布上。

示例

```javascript
var canvas1 = document.querySelector('#cavsElem1');

var canvas2 = document.querySelector('#cavsElem2');

var ctx1 = canvas1.getContext('2d');

var ctx2 = canvas2.getContext('2d');

ctx1.fillRect(20, 20, 40, 40);    //在第一个画布上绘制矩形

ctx2.drawImage(canvas1, 10, 10);   //将第一个画布整体绘制到第二个画布上
```
## 动画

### 基本步骤

- 清空 `canvas`

再绘制每一帧动画之前，需要清空所有。清空所有最简单的做法就是 `clearRect()` 方法。
- 保存 `canvas` 状态

如果在绘制的过程中会更改 `canvas` 的状态(颜色、移动了坐标原点等),又在绘制每一帧时都是原始状态的话，则最好保存下 `canvas` 的状态

- 绘制动画图形

这一步才是真正的绘制动画帧

- 恢复 `canvas` 状态

如果你前面保存了 `canvas` 状态，则应该在绘制完成一帧之后恢复 `canvas` 状态。

### 控制动画

我们可用通过 `canvas` 的方法或者自定义的方法把图像绘制到 `canvas` 上。正常情况，我们能看到绘制的结果是在脚本执行结束之后。例如，我们不可能在一个 `for` 循环内部完成动画。

也就是，为了执行动画，我们需要一些可以定时执行重绘的方法。

一般用到下面三个方法：

```
setInterval()
setTimeout()
requestAnimationFrame()
```

太阳系

```javascript
let sun;
let earth;
let moon;
let ctx;
function init(){
    sun = new Image();
    earth = new Image();
    moon = new Image();
    sun.src = "sun.png";
    earth.src = "earth.png";
    moon.src = "moon.png";

    let canvas = document.querySelector("#solar");
    ctx = canvas.getContext("2d");
    
    sun.onload = function (){
        draw()
    }

}
init();
function draw(){
    ctx.clearRect(0, 0, 300, 300); //清空所有的内容
    /*绘制 太阳*/
    ctx.drawImage(sun, 0, 0, 300, 300);

    ctx.save();
    ctx.translate(150, 150);
    
    //绘制earth轨道
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,0,0.5)";
    ctx.arc(0, 0, 100, 0, 2 * Math.PI)
    ctx.stroke()
    
    let time = new Date();
    //绘制地球
    ctx.rotate(2 * Math.PI / 60 * time.getSeconds() + 2 * Math.PI / 60000 * time.getMilliseconds())
    ctx.translate(100, 0);
    ctx.drawImage(earth, -12, -12)
    
    //绘制月球轨道
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255,.3)";
    ctx.arc(0, 0, 40, 0, 2 * Math.PI);
    ctx.stroke();
    
    //绘制月球
    ctx.rotate(2 * Math.PI / 6 * time.getSeconds() + 2 * Math.PI / 6000 * time.getMilliseconds());
    ctx.translate(40, 0);
    ctx.drawImage(moon, -3.5, -3.5);
    ctx.restore();
    
    requestAnimationFrame(draw);
}
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/796853783-5b74dd8f41e21_articlex.gif)

模拟时钟

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <style>
        body {
            padding: 0;
            margin: 0;
            background-color: rgba(0, 0, 0, 0.1)
        }

        canvas {
            display: block;
            margin: 200px auto;
        }
    </style>
</head>
<body>
<canvas id="solar" width="300" height="300"></canvas>
<script>
    init();

    function init(){
        let canvas = document.querySelector("#solar");
        let ctx = canvas.getContext("2d");
        draw(ctx);
    }
    
    function draw(ctx){
        requestAnimationFrame(function step(){
            drawDial(ctx); //绘制表盘
            drawAllHands(ctx); //绘制时分秒针
            requestAnimationFrame(step);
        });
    }
    /*绘制时分秒针*/
    function drawAllHands(ctx){
        let time = new Date();
    
        let s = time.getSeconds();
        let m = time.getMinutes();
        let h = time.getHours();
        
        let pi = Math.PI;
        let secondAngle = pi / 180 * 6 * s;  //计算出来s针的弧度
        let minuteAngle = pi / 180 * 6 * m + secondAngle / 60;  //计算出来分针的弧度
        let hourAngle = pi / 180 * 30 * h + minuteAngle / 12;  //计算出来时针的弧度
    
        drawHand(hourAngle, 60, 6, "red", ctx);  //绘制时针
        drawHand(minuteAngle, 106, 4, "green", ctx);  //绘制分针
        drawHand(secondAngle, 129, 2, "blue", ctx);  //绘制秒针
    }
    /*绘制时针、或分针、或秒针
     * 参数1：要绘制的针的角度
     * 参数2：要绘制的针的长度
     * 参数3：要绘制的针的宽度
     * 参数4：要绘制的针的颜色
     * 参数4：ctx
     * */
    function drawHand(angle, len, width, color, ctx){
        ctx.save();
        ctx.translate(150, 150); //把坐标轴的远点平移到原来的中心
        ctx.rotate(-Math.PI / 2 + angle);  //旋转坐标轴。 x轴就是针的角度
        ctx.beginPath();
        ctx.moveTo(-4, 0);
        ctx.lineTo(len, 0);  // 沿着x轴绘制针
        ctx.lineWidth = width;
        ctx.strokeStyle = color;
        ctx.lineCap = "round";
        ctx.stroke();
        ctx.closePath();
        ctx.restore();
    }
    
    /*绘制表盘*/
    function drawDial(ctx){
        let pi = Math.PI;
        
        ctx.clearRect(0, 0, 300, 300); //清除所有内容
        ctx.save();
    
        ctx.translate(150, 150); //一定坐标原点到原来的中心
        ctx.beginPath();
        ctx.arc(0, 0, 148, 0, 2 * pi); //绘制圆周
        ctx.stroke();
        ctx.closePath();
    
        for (let i = 0; i < 60; i++){//绘制刻度。
            ctx.save();
            ctx.rotate(-pi / 2 + i * pi / 30);  //旋转坐标轴。坐标轴x的正方形从 向上开始算起
            ctx.beginPath();
            ctx.moveTo(110, 0);
            ctx.lineTo(140, 0);
            ctx.lineWidth = i % 5 ? 2 : 4;
            ctx.strokeStyle = i % 5 ? "blue" : "red";
            ctx.stroke();
            ctx.closePath();
            ctx.restore();
        }
        ctx.restore();
    }
</script>
</body>
</html>

```

![img](https://www.runoob.com/wp-content/uploads/2018/12/2372262871-5b74dd8da51e5_articlex.gif)

```

```