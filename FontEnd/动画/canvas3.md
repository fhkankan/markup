# 案例

## 撒花

[参考](http://www.qiutianaimeili.com/html/page/2018/01/cp6x1dycjh.html)

一个简单的撒花效果，利用canvas绘制的，效果图如下：

![canvas撒花瓣效果](http://active.qiutianaimeili.com/flower_e11.gif)

=> [canvas撒花瓣效果](http://www.qiutianaimeili.com/html/page/2018/01/source/flower/example.html)(二维码)

这里只使用了一张花瓣素材，如下：

![img](http://active.qiutianaimeili.com/1piece_of_pedal.png)

第一步我们需要制作出不同大小，不同透明度的花瓣，这个简单，关键代码如下:

```javascript
var canvasArray = [],
c = document.getElementById('show_rose_canvas'),
a = c.getContext('2d'),
width = 20,
height = 20,
allWidth =parseFloat(getComputedStyle(c).width),
allHeight =parseFloat(getComputedStyle(c).height);
c.width = allWidth;
c.height = allHeight;
var img = new Image();

img.src = 'http://active.qiutianaimeili.com/1piece_of_pedal.png';
img.onload = function() {   
  for(var i = 0; i < 20; i++) { 
    var cloneNode = c.cloneNode(0),   
    cloneContext = cloneNode.getContext('2d');
    var randomScale = Math.random() * .8 + .6;//创建不同大小的花瓣        
    cloneNode.width = width * randomScale;
    cloneNode.height = height * randomScale;        
    cloneNode.style.width = width * randomScale + 'px';        
    cloneNode.style.height = height * randomScale + 'px';        
    cloneContext.globalAlpha = Math.random() * .4 + .6;        
    cloneContext.drawImage(this, 0, 0, width * randomScale, height * randomScale);        
    canvasArray[i] = cloneNode;   
	}
}
```

第二步我们需要将花瓣从侧面喷射出来，这里我们要稍微了解一下速度，阻力和加速度的概念：

水平方向：

阻力，速度衰减：v2=v1*k(k<1)*

垂直方向：

阻力，速度衰减：v2=v1*k(k<1)

重力，匀加速：v2=v1+g(单位时间,速度每次增大g)

我们将任意角度的速度进行分解，可以分解成水平速度和垂直速度，然后在水平方向上会遇到阻力，阻力会让速度进行衰减；在垂直方向上除了阻力外，还有重力，因为重力是匀加速，就是每次速度增加的大小都是一样的，都是g，因此v2=v1+g。对应的关键代码如下所示:

```javascript
Flower.prototype.update = function() { 
  // apply resistance   
  this.vel.x *= this.resistance;   
  this.vel.y *= this.resistance;   
  // gravity down     
  this.vel.y += this.gravity;    
  // update position based on speed     
  this.pos.x += this.vel.x;    
  this.pos.y += this.vel.y;
};
```

这里的衰减值（阻力）和重力值（g）可以自己根据实际情况微调，比如你想让花瓣落下的更快，可以将重力值调大：

```javascript
flower.gravity = 0.08;//重力g的大小
flower.resistance = 0.93;//速度衰减值（阻力）
```

了解了核心的速度变化之后，那么了解喷射就比较简单了，我们只要初始化一个喷射速度，然后控制喷射角度（左边喷射的区间在0度到90度之间，右边的在90度到180度之间），然后让花瓣按照我们的速度变化公式自己变化就好了。

```javascript
var angle = -(Math.random() * Math.PI / 6 + Math.PI / 6);
var speed = Math.cos(Math.random() * Math.PI / 2) * (10 * Math.random() + 15);
flower.vel.x = Math.cos(angle) * speed;
flower.vel.y = Math.sin(angle) * speed;
```

完成上面两步基本上差不多了，不过花瓣在空中肯定是会旋转的，不可能硬邦邦的掉落下来，因此我们现在要让花瓣在空中旋转。

我们首先要找到花瓣的旋转中心点，然后确定花瓣左右旋转的最大角度：

旋转中心点

这个左右摆动和正余弦函数比较类似，我们可以每次加上一个角度，然后计算它的正弦，得到的就是一个摆动的角度：

```javascript
var _rotate = Math.sin(angle) * Math.PI / 4; //angle逐渐增加，产生-Math.PI/4－Math.PI/4之间的角度
```

当然除了上面的旋转摇摆之外，还会转圈，转圈就比较简单了，大概就是每次生成一个角度，然后累加旋转即可。最后为了让屏幕的花瓣运动看起来更加匀称，增加了左边花瓣慢慢右边移动，右边花瓣慢慢左边移动的效果。关键代码如下：

```javascript
this.rotate += Math.random() * Math.PI / 18;//每次生成随即角度
if(this.rotate > (Math.PI * this.rotateCircle + Math.PI / 2)) {
  //如果转了几圈，差不多可以摇摆了    
  var _rotate = Math.sin(this.rotate) * Math.PI / 4 * this.rotateResistance + Math.PI / 4;//前后摇摆Math.PI/4    
  if(this.rotateResistance > 0) {    
    this.rotateResistance -= .0001;   
  }   
  if(this.pos.x > 0.001) {
    //左边的花瓣向右边跑，右边的向左边跑
    if(this.type == 0) {     
      this.pos.x += Math.sin(Math.random() * Math.PI / 2) * .8;        
    } 
    else 
    {     
      this.pos.x -= Math.sin(Math.random() * Math.PI / 2) * .8;        
    }       
  }   
  c.rotate(_rotate);
} 
else
{
  //花瓣转圈    
  c.rotate(this.rotate);
}
```

## 小球拖拽

[参考](https://www.cnblogs.com/ye-hcj/p/10361027.html)

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <style type="text/css">
        canvas {
            border: 1px solid black;
        }
    </style>
</head>

<body>
    <canvas id="canvas" width="500px" height="500px"></canvas>
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        var cRect = canvas.getBoundingClientRect();  
        var raf;
        var running = false;

        var ball = {
            x: 100,
            y: 100,
            vx: 5,
            vy: 2,
            radius: 25,
            color: 'red',
            draw: function () {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2, true);
                ctx.closePath();
                ctx.fillStyle = this.color;
                ctx.fill();
            }
        };

        function clear() {
            
            // ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 尾影效果使用下面
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        function draw() {
            clear();
            ball.draw();

            // 移动
            ball.x += ball.vx;
            ball.y += ball.vy;

            // 曲线运动
            ball.vy *= .99;
            ball.vy += .25;

            // 边界
            if (ball.y + ball.vy > canvas.height || ball.y + ball.vy < 0) {
                ball.vy = -ball.vy;
            }
            if (ball.x + ball.vx > canvas.width || ball.x + ball.vx < 0) {
                ball.vx = -ball.vx;
            }

            raf = window.requestAnimationFrame(draw);

        }

        canvas.addEventListener('mousemove', function (e) {
            if (!running) {
                clear();
                ball.x = Math.round(e.clientX - cRect.left);
                ball.y = Math.round(e.clientY - cRect.top);
                ball.draw();   // 直接调用draw只会绘制一帧
            }
        });

        canvas.addEventListener('click', function (e) {
            if (!running) {
                raf = window.requestAnimationFrame(draw);
                running = true;
            }
        });

        canvas.addEventListener('mouseout', function (e) {
            window.cancelAnimationFrame(raf);
            running = false;
        });

        ball.draw();


    </script>
</body>

</html>
```

## 进度波浪

[参考](https://www.jb51.net/article/99612.htm)

- 核心部分

**绘制 sin() 曲线**

```javascript
`var` `canvas = document.getElementById(``'c'``);``var` `ctx = canvas.getContext(``'2d'``);` `//画布属性``var` `mW = canvas.width = 700;``var` `mH = canvas.height = 300;``var` `lineWidth = 1;` `//Sin 曲线属性``var` `sX = 0;``var` `sY = mH / 2;``var` `axisLength = mW; ``//轴长``var` `waveWidth = 0.011 ; ``//波浪宽度,数越小越宽 ``var` `waveHeight = 70; ``//波浪高度,数越大越高` `ctx.lineWidth = lineWidth;` `//画sin 曲线函数``var` `drawSin = ``function``(xOffset){`` ``ctx.save();` ` ``var` `points=[]; ``//用于存放绘制Sin曲线的点` ` ``ctx.beginPath();`` ``//在整个轴长上取点`` ``for``(``var` `x = sX; x < sX + axisLength; x += 20 / axisLength){`` ``//此处坐标(x,y)的取点，依靠公式 “振幅高*sin(x*振幅宽 + 振幅偏移量)”`` ``var` `y = -Math.sin((sX + x) * waveWidth);` ` ``points.push([x, sY + y * waveHeight]);`` ``ctx.lineTo(x, sY + y * waveHeight); `` ``}` ` ``//封闭路径`` ``ctx.lineTo(axisLength, mH);`` ``ctx.lineTo(sX, mH);`` ``ctx.lineTo(points[0][0],points[0][1]);`` ``ctx.stroke()` ` ``ctx.restore();``};``drawSin()`
```

此处通过`waveWidth`和`waveHeight`调节曲线的陡峭度和周期。

**加入动态效果**

```
var speed = 0.04; //波浪速度，数越大速度越快
var xOffset = 0; //波浪x偏移量
```

速度变量和x偏移变量

```
var y = -Math.sin((sX + x) * waveWidth + xOffset);
```

修改y点的函数。

```javascript
`var render = function(){`` ``ctx.clearRect(``0``, ``0``, mW, mH);` ` ``drawSin(xOffset);`` ``xOffset += speed; ``//形成动态效果`` ``requestAnimationFrame(render);``}` `render()`
```

加入渲染。

**百分比控制**

因为要加入百分比不同的涨幅效果，所以要对y的坐标时行百分比控制修改。

```
var dY = mH * (1 - nowRange / 100 );
```

球型显示

这里需要用到`clip()`进行球型裁切显示。

```javascript
`ctx.beginPath();``ctx.arc(r, r, cR, 0, 2 * Math.PI);``ctx.clip();`
```

其他

可以通过修改如下变量来修改曲线的形状以及速度：

```javascript
`var` `waveWidth = 0.015 ; ``//波浪宽度,数越小越宽 ``var` `waveHeight = 6; ``//波浪高度,数越大越高``var` `speed = 0.09; ``//波浪速度，数越大速度越快`
```

- 完整代码

```html
<!doctype html>
<html lang="en">
<head>
 <meta charset="UTF-8" />
 <title>Document</title>
 <style type="text/css">
  #c{
   margin: 0 auto;
   display: block;
  }
  #r{
   display: block;
   margin: 0 auto;
  }
  #r::before{
   color: black;
   content: attr(min);
   padding-right: 10px;
  }
  #r::after{
   color: black;
   content: attr(max);
   padding-left: 10px;
  }  
 </style>
</head>
<body>
 <canvas id="c"></canvas>
 <input type="range" id="r" min="0" max="100" step="1">
 
 <script type="text/javascript">
  var canvas = document.getElementById('c');
  var ctx = canvas.getContext('2d');
  var range = document.getElementById('r');
 
  //range控件信息
  var rangeValue = range.value;
  var nowRange = 0; //用于做一个临时的range
 
  //画布属性
  var mW = canvas.width = 250;
  var mH = canvas.height = 250;
  var lineWidth = 2;
 
  //圆属性
  var r = mH / 2; //圆心
  var cR = r - 16 * lineWidth; //圆半径
 
  //Sin 曲线属性
  var sX = 0;
  var sY = mH / 2;
  var axisLength = mW; //轴长
  var waveWidth = 0.015 ; //波浪宽度,数越小越宽 
  var waveHeight = 6; //波浪高度,数越大越高
  var speed = 0.09; //波浪速度，数越大速度越快
  var xOffset = 0; //波浪x偏移量
 
  ctx.lineWidth = lineWidth;
 
  //画圈函数
  var IsdrawCircled = false;
  var drawCircle = function(){
 
   ctx.beginPath();
   ctx.strokeStyle = '#1080d0';
   ctx.arc(r, r, cR+5, 0, 2 * Math.PI);
   ctx.stroke();
   ctx.beginPath();
   ctx.arc(r, r, cR, 0, 2 * Math.PI);
   ctx.clip();
 
  }
 
  //画sin 曲线函数
  var drawSin = function(xOffset){
   ctx.save();
 
   var points=[]; //用于存放绘制Sin曲线的点
 
   ctx.beginPath();
   //在整个轴长上取点
   for(var x = sX; x < sX + axisLength; x += 20 / axisLength){
    //此处坐标(x,y)的取点，依靠公式 “振幅高*sin(x*振幅宽 + 振幅偏移量)”
    var y = -Math.sin((sX + x) * waveWidth + xOffset);
 
    var dY = mH * (1 - nowRange / 100 );
 
    points.push([x, dY + y * waveHeight]);
    ctx.lineTo(x, dY + y * waveHeight);  
   }
 
   //封闭路径
   ctx.lineTo(axisLength, mH);
   ctx.lineTo(sX, mH);
   ctx.lineTo(points[0][0],points[0][1]);
   ctx.fillStyle = '#1c86d1';
   ctx.fill();
 
   ctx.restore();
  };
 
  //写百分比文本函数
  var drawText = function(){
   ctx.save();
 
   var size = 0.4*cR;
   ctx.font = size + 'px Microsoft Yahei';
   ctx.textAlign = 'center';
   ctx.fillStyle = "rgba(06, 85, 128, 0.8)";
   ctx.fillText(~~nowRange + '%', r, r + size / 2);
 
   ctx.restore();
  };
 
  var render = function(){
   ctx.clearRect(0, 0, mW, mH);
 
   rangeValue = range.value;
 
   if(IsdrawCircled == false){
    drawCircle(); 
   }
 
   if(nowRange <= rangeValue){
    var tmp = 1;
    nowRange += tmp;
   }
 
   if(nowRange > rangeValue){
    var tmp = 1;
    nowRange -= tmp;
   }
 
   drawSin(xOffset);
   drawText(); 
 
   xOffset += speed;
   requestAnimationFrame(render);
  }
 
  render();  
 </script>
</body>
</html>
```

