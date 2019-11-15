# canvas2

## 绘制图片


我们也可以在 `canvas` 上直接绘制图片。

### 由零开始创建图片

- 创建`<img>`

```
var img = new Image();   // 创建一个<img>元素 
img.src = 'myImage.png'; // 设置图片源地址
```

脚本执行后图片开始装载。

- 绘制 `img`

```
// 参数 1：要绘制的 img  
// 参数 2、3：绘制的 img 在 canvas 中的坐标
ctx.drawImage(img,0,0); 
```

> 注意：
考虑到图片是从网络加载，如果 `drawImage` 的时候图片还没有完全加载完成，则什么都不做，个别浏览器会抛异常。所以我们应该保证在 `img` 绘制完成之后再 `drawImage`。

```javascript
var img = new Image();   // 创建img元素
img.onload = function(){
  ctx.drawImage(img, 0, 0)
}
img.src = 'myImage.png'; // 设置图片源地址

```

### 绘制 `img` 标签元素中的图片

`img` 可以 `new` 也可以来源于我们页面的 `<img>`标签。

```html
<img src="./美女.jpg" alt="" width="300"><br>
<canvas id="tutorial" width="600" height="400"></canvas>
<script type="text/javascript">
    function draw(){
        var canvas = document.getElementById('tutorial');
        if (!canvas.getContext) return;
        var ctx = canvas.getContext("2d");
        var img = document.querySelector("img");
        ctx.drawImage(img, 0, 0);
    }
    document.querySelector("img").onclick = function (){
        draw();
    }

</script>
```

第一张图片就是页面中的 `<img>` 标签：

![img](https://www.runoob.com/wp-content/uploads/2018/12/2255709523-5b74dd8eb033e_articlex.png)

### 缩放图片

`drawImage()` 也可以再添加两个参数：

```
drawImage(image, x, y, width, height)
```

这个方法多了 2 个参数：`width` 和 `height`，这两个参数用来控制 当像 canvas 画入时应该缩放的大小。

```
ctx.drawImage(img, 0, 0, 400, 200)
```

![img](https://www.runoob.com/wp-content/uploads/2018/12/327614951-5b74dd8e71ab1_articlex.png)

### 切片

```
drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
```

第一个参数和其它的是相同的，都是一个图像或者另一个 canvas 的引用。

其他 8 个参数：

前 4 个是定义图像源的切片位置和大小，后 4 个则是定义切片的目标显示位置和大小。

![img](https://www.runoob.com/wp-content/uploads/2018/12/2106688680-54566fa3d81dc_articlex.jpeg)


## 状态的保存和恢复

`Saving and restoring state` 是绘制复杂图形时必不可少的操作。

`save()` 和 `restore()`

`save` 和 `restore` 方法是用来保存和恢复 `canvas` 状态的，都没有参数。

`Canvas` 的状态就是当前画面应用的所有样式和变形的一个快照。

- 关于 save() ：

Canvas状态存储在栈中，每当save()方法被调用后，当前的状态就被推送到栈中保存。

一个绘画状态包括：

1. 当前应用的变形（即移动，旋转和缩放）

2. `strokeStyle`, `fillStyle`, `globalAlpha`, `lineWidth`, `lineCap`, `lineJoin`, `miterLimit`, `shadowOffsetX`, `shadowOffsetY`, `shadowBlur`, `shadowColor`, `globalCompositeOperation 的值`

3. 当前的裁切路径（`clipping path`）

可以调用任意多次 `save`方法(类似数组的 `push()`)。

- 关于restore()：

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

## 变形

### translate

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

### rotate

```
rotate(angle)
```

旋转坐标轴。

这个方法只接受一个参数：旋转的角度(angle)，它是顺时针方向的，以弧度为单位的值。

 旋转的中心是坐标原点。

![img](https://www.runoob.com/wp-content/uploads/2018/12/3322150878-5b74dd8e2b6a4_articlex.png)

```
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

### scale

```
scale(x, y)
```

我们用它来增减图形在 `canvas` 中的像素数目，对形状，位图进行缩小或者放大。

`scale`方法接受两个参数。`x,y` 分别是横轴和纵轴的缩放因子，它们都必须是正值。值比 1.0 小表示缩 小，比 1.0 大则表示放大，值为 1.0 时什么效果都没有。

 默认情况下，`canvas` 的 1 单位就是 1 个像素。举例说，如果我们设置缩放因子是 0.5，1 个单位就变成对应 0.5 个像素，这样绘制出来的形状就会是原先的一半。同理，设置为 2.0 时，1 个单位就对应变成了 2 像素，绘制的结果就是图形放大了 2 倍。

### transform 

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

### lighten

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

## 裁剪路径

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

我们可用通过 `canvas` 的方法或者自定义的方法把图像会知道到 `canvas` 上。正常情况，我们能看到绘制的结果是在脚本执行结束之后。例如，我们不可能在一个 `for` 循环内部完成动画。

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