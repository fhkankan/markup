# 案例

## 撒花

[参考](http://www.qiutianaimeili.com/html/page/2018/01/cp6x1dycjh.html)

一个简单的撒花效果，利用canvas绘制的，效果图如下：

![canvas撒花瓣效果](http://active.qiutianaimeili.com/flower_e11.gif)

=> [canvas撒花瓣效果](http://www.qiutianaimeili.com/html/page/2018/01/source/flower/example.html)(二维码)

#### 思路分析

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