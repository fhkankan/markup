# CSS3动画

## transform

[Transform](http://www.w3.org/TR/css3-2d-transforms/)字面上就是变形，改变的意思。在[CSS3](http://www.w3.org/TR/css3-roadmap/)中[transform](http://www.w3.org/TR/css3-2d-transforms/)主要包括以下几种：**旋转rotate**、**扭曲skew**、**缩放scale**和**移动translate**以及**矩阵变形matrix**。下面我们一起来看看CSS3中transform的旋转rotate、扭曲skew、缩放scale和移动translate具体如何实现，老样子，我们就从[transform](http://www.w3.org/TR/css3-2d-transforms/)的语法开始吧。是构成transtion和animation的基础。

- 语法

```javascript
transform ： none | <transform-function> [ <transform-function> ]* 
// 也就是：
transform: rotate | scale | skew | translate |matrix;
```

- 参数

`none`:表示不进么变换；

`<transform-function>`表示一个或多个变换函数，以空格分开；换句话说就是我们同时对一个元素进行transform的多种属性操作，例如rotate、scale、translate三种，但这里需要提醒大家的，以往我们叠加效果都是用逗号（“，”）隔开，但transform中使用多个属性时却需要有空格隔开。大家记住了是空格隔开。

transform属性实现了一些可用SVG实现的同样的功能。它可用于内联(inline)元素和块级(block)元素。它允许我们旋转、缩放和移动元素 ，他有几个属性值参数：rotate;translate;scale;skew;matrix。下面我们分别来介绍这几个属性值参数的具体使用方法：

### 旋转rotate

`rotate(<angle>)` ：通过指定的角度参数对原元素指定一个[2D rotation](http://www.w3.org/TR/SVG/coords.html#RotationDefined)（2D 旋转），需先有transform-origin属性的定义。transform-origin定义的是旋转的基点，其中angle是指旋转角度，如果设置的值为正数表示顺时针旋转，如果设置的值为负数，则表示逆时针旋转。如：`transform:rotate(30deg)`:

![img](http://cdn1.w3cplus.com/cdn/farfuture/z_h-BB20wAjCAy2velbYrqK8CVIUstAOR_uOb_5RMkM/mtime:1341237776/sites/default/files/rotate.png)

### 移动translate

移动translate我们分为三种情况：

```
- translate(x,y)水平方向和垂直方向同时移动（也就是X轴和Y轴同时移动）

- translateX(x)仅水平方向移动（X轴移动）

- translateY(Y)仅垂直方向移动（Y轴移动） 
```

具体使用方法如下：

- `translate(<translation-value>[, <translation-value>]) `

通过矢量[tx, ty]指定一个[2D translation](http://www.w3.org/TR/SVG/coords.html#TranslationDefined)，tx 是第一个过渡值参数，ty 是第二个过渡值参数选项。如果 未被提供，则ty以 0 作为其值。也就是translate(x,y),它表示对象进行平移，按照设定的x,y参数值,当值为负数时，反方向移动物体，其基点默认为元素 中心点，也可以根据transform-origin进行改变基点。如`transform:translate(100px,20px)`:

![img](http://cdn.w3cplus.com/cdn/farfuture/cpg1_ccfmJB3YCskfTY4wafGLRW_PqZ_hmTATY0bLhI/mtime:1341237813/sites/default/files/translate-x-y.png)

- `translateX(<translation-value>) `

通过给定一个X方向上的数目指定一个[translation](http://www.w3.org/TR/SVG/coords.html#TranslationDefined)。只向x轴进行移动元素，同样其基点是元素中心点，也可以根据transform-origin改变基点位置。如：`transform:translateX(100px)`:

![img](http://cdn.w3cplus.com/cdn/farfuture/xmWvEQJv1-SDS31NWYdMZCm5Y-mFhmVeov3byX8xQGM/mtime:1341237813/sites/default/files/translate-x.png)

- `translateY(<translation-value>) `

通过给定Y方向的数目指定一个[translation](http://www.w3.org/TR/SVG/coords.html#TranslationDefined)。只向Y轴进行移动，基点在元素心点，可以通过transform-origin改变基点位置。如：`transform:translateY(20px)`:

![img](http://cdn2.w3cplus.com/cdn/farfuture/91IxTFBu9C9BOcZ-7l_v-x1IoOyr0uTiYpa4XxlxFWc/mtime:1341237813/sites/default/files/translate-y.png)

### 缩放scale

缩放scale和移动translate是极其相似，他也具有三种情况：scale(x,y)使元素水平方向和垂直方向同时缩放（也就是X轴和Y轴同时缩放）；scaleX(x)元素仅水平方向缩放（X轴缩放）；scaleY(y)元素仅垂直方向缩放（Y轴缩放），但它们具有相同的缩放中心点和基数，其中心点就是元素的中心位置，缩放基数为1，如果其值大于1元素就放大，反之其值小于1，元素缩小。下面我们具体来看看这三种情况具体使用方法：

- `scale(<number>[, <number>])`：

提供执行[sx,sy]缩放矢量的两个参数指定一个[2D scale](http://www.w3.org/TR/SVG/coords.html#ScalingDefined)（2D缩放）。如果第二个参数未提供，则取与第一个参数一样的值。scale(X,Y)是用于对元素进行缩放，可以通过transform-origin对元素的基点进行设置，同样基点在元素中心位置；基中X表示水平方向缩放的倍数，Y表示垂直方向的缩放倍数，而Y是一个可选参数，如果没有设置Y值，则表示X，Y两个方向的缩放倍数是一样的。并以X为准。如：`transform:scale(2,1.5)`:

![img](http://cdn.w3cplus.com/cdn/farfuture/OVgmVMi2MDjTG8KSooxaZIJEzSbTqHPuD3lgcM1Ahvk/mtime:1341237776/sites/default/files/scale-x-y.png)

- `scaleX(<number>) `

 使用 [sx,1] 缩放矢量执行缩放操作，sx为所需参数。scaleX表示元素只在X轴(水平方向)缩放元素，他的默认值是(1,1)，其基点一样是在元素的中心位置，我们同样是通过transform-origin来改变元素的基点。如：`transform:scaleX(2)`:

![img](http://cdn2.w3cplus.com/cdn/farfuture/GwlGkCboT7jfuw8Y_EtqY07GMcDUMNEt4Y2kj61FcD8/mtime:1341237776/sites/default/files/scale-x.png)

- `scaleY(<number>)`

 使用 [1,sy] 缩放矢量执行缩放操作，sy为所需参数。scaleY表示元素只在Y轴（垂直方向）缩放元素，其基点同样是在元素中心位置，可以通过transform-origin来改变元素的基点。如`transform:scaleY(2)`:

![img](http://cdn1.w3cplus.com/cdn/farfuture/-h73QRWTLl4beHBNxbo8IxCBPalKqtRRoAyaGAksQW8/mtime:1341237776/sites/default/files/scale-y.png)

### 扭曲skew

扭曲skew和translate、scale一样同样具有三种情况：skew(x,y)使元素在水平和垂直方向同时扭曲（X轴和Y轴同时按一定的角度值进行扭曲变形）；skewX(x)仅使元素在水平方向扭曲变形（X轴扭曲变形）；skewY(y)仅使元素在垂直方向扭曲变形（Y轴扭曲变形），具体使用如下：

- `skew(<angle> [, <angle>]) `

X轴Y轴上的[skew transformation](http://www.w3.org/TR/SVG/coords.html#SkewXDefined)（斜切变换）。第一个参数对应X轴，第二个参数对应Y轴。如果第二个参数未提供，则值为0，也就是Y轴方向上无斜切。skew是用来对元素进行扭曲变行，第一个参数是水平方向扭曲角度，第二个参数是垂直方向扭曲角度。其中第二个参数是可选参数，如果没有设置第二个参数，那么Y轴为0deg。同样是以元素中心为基点，我们也可以通过transform-origin来改变元素的基点位置。如：`transform:skew(30deg,10deg)`:

![img](http://cdn1.w3cplus.com/cdn/farfuture/L_YIhJ0lb-t2WzPPcxliu82nnLN0OL7BTm6Rao3X7_0/mtime:1341237780/sites/default/files/skew-x-y.png)

- `skewX(<angle>)`

 按给定的角度沿X轴指定一个[skew transformation](http://www.w3.org/TR/SVG/coords.html#SkewXDefined)（斜切变换）。skewX是使元素以其中心为基点，并在水平方向（X轴）进行扭曲变行，同样可以通过transform-origin来改变元素的基点。如：`transform:skewX(30deg)`

![img](http://cdn1.w3cplus.com/cdn/farfuture/JnUewTlzfSZsJhlDYUuPiVXpM6wuagEgpFkSorwtk6k/mtime:1341237780/sites/default/files/skew-x.png)

- `skewY(<angle>) `

按给定的角度沿Y轴指定一个[skew transformation](http://www.w3.org/TR/SVG/coords.html#SkewYDefined)（斜切变换）。skewY是用来设置元素以其中心为基点并按给定的角度在垂直方向（Y轴）扭曲变形。同样我们可以通过transform-origin来改变元素的基点。如：`transform:skewY(10deg)`

![img](http://cdn.w3cplus.com/cdn/farfuture/NEOscxx3CZUOpk6R3GBgzSEbkZg8OcC5WfrKdjnko4k/mtime:1341237781/sites/default/files/skew-y.png)

### 矩阵matrix

`matrix(<number>, <number>, <number>, <number>, <number>, <number>) `

 以一个含六值的(a,b,c,d,e,f)[变换矩阵](http://www.w3.org/TR/SVG/coords.html#TransformMatrixDefined)的形式指定一个2D变换，相当于直接应用一个[a b c d e f]变换矩阵。就是基于水平方向（X轴）和垂直方向（Y轴）重新定位元素,此属性值使用涉及到数学中的矩阵，我在这里只是简单的说一下CSS3中的transform有这么一个属性值，如果有感兴趣的朋友可以去了解更深层次的martix使用方法，这里就不多说了。

- 改变元素基点`transform-origin`

前面我们多次提到transform-origin这个东东，他的主要作用就是让我们在进行transform动作之前可以改变元素的基点位置，因为我们元素默认基点就是其中心位置，换句话说我们没有使用transform-origin改变元素基点位置的情况下，transform进行的rotate,translate,scale,skew,matrix等操作都是以元素自己中心位置进行变化的。但有时候我们需要在不同的位置对元素进行这些操作，那么我们就可以使用transform-origin来对元素进行基点位置改变，使元素基点不在是中心位置，以达到你需要的基点位置。下面我们主要来看看其使用规则：

transform-origin(X,Y):用来设置元素的运动的基点（参照点）。默认点是元素的中心点。其中X和Y的值可以是百分值,em,px，其中X也可以是字符参数值left,center,right；Y和X一样除了百分值外还可以设置字符值top,center,bottom，这个看上去有点像我们background-position设置一样；下面我列出他们相对应的写法：
```
1、top left | left top 等价于 0 0 | 0% 0%

2、top | top center | center top 等价于 50% 0

3、right top | top right 等价于 100% 0

4、left | left center | center left 等价于 0 50% | 0% 50%

5、center | center center 等价于 50% 50%（默认值）

6、right | right center | center right 等价于 100% 50%

7、bottom left | left bottom 等价于 0 100% | 0% 100%

8、bottom | bottom center | center bottom 等价于 50% 100%

9、bottom right | right bottom 等价于 100% 100%
```
其中 left,center right是水平方向取值，对应的百分值为left=0%;center=50%;right=100%而top center bottom是垂直方向的取值，其中top=0%;center=50%;bottom=100%;如果只取一个值，表示垂直方向值不变，我们分别来看看以下几个实例

`transform-origin:(left,top)`

![img](http://cdn1.w3cplus.com/cdn/farfuture/IARPBybmPGq3biBF6HVZzncmgQSg0NCEV3WGaP5Q_vM/mtime:1341237763/sites/default/files/origin-x-y.png)

`transform-origin:right`

![img](http://cdn1.w3cplus.com/cdn/farfuture/eujJD35rExsh96YKXjWxwbzcApSUeXYwjQk6lDr9OO0/mtime:1341237763/sites/default/files/origin-x.png)

`transform-origin(25%,75%)`

![img](http://cdn.w3cplus.com/cdn/farfuture/R8UAy6eITsixD6HElHelwUVGge_ZuEBRL-39L0KglqU/mtime:1341237763/sites/default/files/origin-per.png)

更多的改变中心基点办法，大家可以在本地多测试一下，多体会一下，这里还要提醒大家一点的是，transform-origin并不是transform中的属性值，他具有自己的语法，前面我也说过了，说简单一点就是类似于我们的background-position的用法，但又有其不一样，因为我们background-position不需要区别浏览器内核不同的写法，但transform-origin跟其他的css3属性一样，我们需要在不同的浏览内核中加上相应的前缀，下面列出各种浏览器内核下的语法规则：

```
  //Mozilla内核浏览器：firefox3.5+
  -moz-transform-origin: x y;
  //Webkit内核浏览器：Safari and Chrome
  -webkit-transform-origin: x y;
  //Opera
  -o-transform-origin: x y ;
  //IE9
  -ms-transform-origin: x y;
  //W3C标准
  transform-origin: x y ;
```

**transform在不同浏览器内核下的书写规则**

```
//Mozilla内核浏览器：firefox3.5+
  -moz-transform: rotate | scale | skew | translate ;
 //Webkit内核浏览器：Safari and Chrome
  -webkit-transform: rotate | scale | skew | translate ;
 //Opera
  -o-transform: rotate | scale | skew | translate ;
 //IE9
  -ms-transform: rotate | scale | skew | translate ;
 //W3C标准
  transform: rotate | scale | skew | translate ;
```

 上面列出是不同浏览内核transform的书写规则，如果需要兼容各浏览器的话，以上写法都需要调用。

**支持transform浏览器**

![img](http://cdn2.w3cplus.com/cdn/farfuture/-wJc3FaCYFoN91POYMTmOmSYfUEHbIz0JEkyzJk1SEY/mtime:1341237805/sites/default/files/transform-browers.png)

　　同样的transform在IE9下版本是无法兼容的，之所以有好多朋友说，IE用不了，搞这个做什么？个人认为，CSS3推出来了，他是一门相对前沿的技术，做为Web前端的开发者或者爱好者都有必要了解和掌握的一门新技术，如果要等到所有浏览器兼容，那我们只能对css3说NO，我用不你。因为IE老大是跟不上了，，，，纯属个人观点，不代表任何。还是那句话，感兴趣的朋友跟我一样，不去理会IE，我们继续看下去。

## transition

[W3C](http://www.w3.org/)标准中对[css3](http://www.w3.org/TR/css3-roadmap/)的[transition](http://www.w3.org/TR/css3-transitions/)这是样描述的：“css的[transition](http://www.w3.org/TR/css3-transitions/)允许css的属性值在一定的时间区间内平滑地过渡。这种效果可以在鼠标单击、获得焦点、被点击或对元素任何改变中触发，并圆滑地以动画效果改变CSS的属性值。”

下面我们同样从其最语法和属性值开始一步一步来学习[transition](http://www.w3.org/TR/css3-transitions/)的具体使用

**语法:**

```javascript
transition ： [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'> [, [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'>]]* 
```

transition主要包含四个属性值：执行变换的属性：transition-property,变换延续的时间：transition-duration,在延续时间段，变换的速率变化transition-timing-function,变换延迟时间transition-delay。下面分别来看这四个属性值

### transition-property

**语法：**

```
transition-property ： none | all | [ <IDENT> ] [ ',' <IDENT> ]*
```

transition-property是用来指定当元素其中一个属性改变时执行transition效果，其主要有以下几个值：none(没有属性改变)；all（所有属性改变）这个也是其默认值；indent（元素属性名）。当其值为none时，transition马上停止执行，当指定为all时，则元素产生任何属性值变化时都将执行transition效果，ident是可以指定元素的某一个属性值。其对应的类型如下：
```
color: 通过红、绿、蓝和透明度组件变换（每个数值处理）如：background-color,border-color,color,outline-color等css属性；

length: 真实的数字 如：word-spacing,width,vertical-align,top,right,bottom,left,padding,outline-width,margin,min-width,min-height,max-width,max-height,line-height,height,border-width,border-spacing,background-position等属性；

percentage:真实的数字 如：word-spacing,width,vertical-align,top,right,bottom,left,min-width,min-height,max-width,max-height,line-height,height,background-position等属性；

integer离散步骤（整个数字），在真实的数字空间，以及使用floor()转换为整数时发生 如：outline-offset,z-index等属性；

number真实的（浮点型）数值，如：zoom,opacity,font-weight,等属性；

transform list:详情请参阅：《[CSS3 Transform](http://www.w3cplus.com/content/css3-transform)》

rectangle:通过x, y, width 和 height（转为数值）变换，如：crop

visibility: 离散步骤，在0到1数字范围之内，0表示“隐藏”，1表示完全“显示”,如：visibility

shadow: 作用于color, x, y 和 blur（模糊）属性,如：text-shadow

gradient: 通过每次停止时的位置和颜色进行变化。它们必须有相同的类型（放射状的或是线性的）和相同的停止数值以便执行动画,如：background-image

paint server (SVG): 只支持下面的情况：从gradient到gradient以及color到color，然后工作与上面类似

space-separated list of above:如果列表有相同的项目数值，则列表每一项按照上面的规则进行变化，否则无变化

a shorthand property: 如果缩写的所有部分都可以实现动画，则会像所有单个属性变化一样变化
```
具体什么css属性可以实现transition效果，在W3C官网中列出了所有可以实现transition效果的CSS属性值以及值的类型，大家可以点[这里](http://www.w3.org/TR/css3-transitions/#properties-from-css-)了解详情。这里需要提醒一点是，并不是什么属性改变都为触发transition动作效果，比如页面的自适应宽度，当浏览器改变宽度时，并不会触发transition的效果。但上述表格所示的属性类型改变都会触发一个transition动作效果。

### transition-duration

**语法：**

```
transition-duration ： <time> [, <time>]* 
```

`transition-duration`是用来指定元素 转换过程的持续时间，取值：`<time>`为数值，单位为s（秒）或者ms(毫秒),可以作用于所有元素，包括:before和:after伪元素。其默认值是0，也就是变换时是即时的。

### transition-timing-function

**语法：**

```
transition-timing-function ： ease | linear | ease-in | ease-out | ease-in-out | cubic-bezier(<number>, <number>, <number>, <number>) [, ease | linear | ease-in | ease-out | ease-in-out | cubic-bezier(<number>, <number>, <number>, <number>)]*  
```

**取值：**

transition-timing-function的值允许你根据时间的推进去改变属性值的变换速率，transition-timing-function有6个可能值：
```
1、ease：（逐渐变慢）默认值，ease函数等同于贝塞尔曲线(0.25, 0.1, 0.25, 1.0).

2、linear：（匀速），linear 函数等同于贝塞尔曲线(0.0, 0.0, 1.0, 1.0).

3、ease-in：(加速)，ease-in 函数等同于贝塞尔曲线(0.42, 0, 1.0, 1.0).

4、ease-out：（减速），ease-out 函数等同于贝塞尔曲线(0, 0, 0.58, 1.0).

5、ease-in-out：（加速然后减速），ease-in-out 函数等同于贝塞尔曲线(0.42, 0, 0.58, 1.0)

6、cubic-bezier：（该值允许你去自定义一个时间曲线）， 特定的[cubic-bezier曲线](http://en.wikipedia.org/wiki/Bézier_curve)。 (x1, y1, x2, y2)四个值特定于曲线上点P1和点P2。所有值需在[0, 1]区域内，否则无效。
```
其是cubic-bezier为通过贝赛尔曲线来计算“转换”过程中的属性值，如下曲线所示，通过改变P1(x1, y1)和P2(x2, y2)的坐标可以改变整个过程的Output Percentage。初始默认值为default.

![img](http://cdn2.w3cplus.com/cdn/farfuture/V_eZPWDqH27qCwOpJCKxEOac-OGLhW2U4Zc4dDTpIrE/mtime:1341237580/sites/default/files/cubic-bezier.png)

其他几个属性的示意图：

![img](http://cdn1.w3cplus.com/cdn/farfuture/iEpdD5HOE9exHIfIWCd2bZ0JXYBaB73pEaokX4-8X9U/mtime:1341237812/sites/default/files/transition-timing-function.png)

### transition-delay

**语法：**

```
  transition-delay ： <time> [, <time>]* 
```

 transition-delay是用来指定一个动画开始执行的时间，也就是说当改变元素属性值后多长时间开始执行transition效果，其取值：<time>为数值，单位为s（秒）或者ms(毫秒)，其使用和transition-duration极其相似，也可以作用于所有元素，包括:before和:after伪元素。 默认大小是"0"，也就是变换立即执行，没有延迟。

有时我们不只改变一个css效果的属性,而是想改变两个或者多个css属性的transition效果，那么我们只要把几个transition的声明串在一起，用逗号（“，”）隔开，然后各自可以有各自不同的延续时间和其时间的速率变换方式。但需要值得注意的一点：transition-delay与transition-duration的值都是时间，所以要区分它们在连写中的位置，一般浏览器会根据先后顺序决定，第一个可以解析为时间的怭值为transition-duration第二个为transition-delay。如：

```
  a {
    -moz-transition: background 0.5s ease-in,color 0.3s ease-out;
    -webkit-transition: background 0.5s ease-in,color 0.3s ease-out;
    -o-transition: background 0.5s ease-in,color 0.3s ease-out;
    transition: background 0.5s ease-in,color 0.3s ease-out;
  }
```

 

如果你想给元素执行所有transition效果的属性，那么我们还可以利用all属性值来操作，此时他们共享同样的延续时间以及速率变换方式，如：

```
  a {
    -moz-transition: all 0.5s ease-in;
    -webkit-transition: all 0.5s ease-in;
    -o-transition: all 0.5s ease-in;
    transition: all 0.5s ease-in;
  }
```

 

综合上述我们可以给**transition一个速记法：transition:   **如下图所示：

![img](http://cdn.w3cplus.com/cdn/farfuture/qGrQbdA_Yq7QLKbkOGaaA4Ol27HVBLJiaRLTogqSMZo/mtime:1341237811/sites/default/files/transition-suji.png)

相对应的一个示例代码：

```
p {
  -webkit-transition: all .5s ease-in-out 1s;
  -o-transition: all .5s ease-in-out 1s;
  -moz-transition: all .5s ease-in-out 1s;
  transition: all .5s ease-in-out 1s;
}
```

 

**浏览器的兼容性:**

![img](http://cdn1.w3cplus.com/cdn/farfuture/39u1NHSUBwgSp9XVpTTV1Bj8m7cU5DprkOP_FJGf0l8/mtime:1341237809/sites/default/files/transition-browers.png)

因为transition最早是有由webkit内核浏览器提出来的，mozilla和opera都是最近版本才支持这个属性，而我们的大众型浏览器IE全家都是不支持，另外由于各大现代浏览器Firefox,Safari,Chrome,Opera都还不支持W3C的标准写法，所以在应用transition时我们有必要加上各自的前缀，最好在放上我们W3C的标准写法，这样标准的会覆盖前面的写法，只要浏览器支持我们的transition属性，那么这种效果就会自动加上去：

```
   //Mozilla内核
   -moz-transition ： [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'> [, [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'>]]* 
   //Webkit内核
   -webkit-transition ： [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'> [, [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'>]]* 
   //Opera
   -o-transition ： [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'> [, [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'>]]* 
   //W3C 标准
   transition ： [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'> [, [<'transition-property'> || <'transition-duration'> || <'transition-timing-function'> || <'transition-delay'>]]* 
```

```
div{
            width:200px;
            height:100px;
            background: #00f;
            -webkit-transition: all 1s ease .1s;
        }
        div:hover{
            background: #f00;
            transform: translateZ(-25px) rotateX(90deg);
        }
```

## Animation

单从[Animation](http://www.w3.org/TR/css3-animations/)字面上的意思，我们就知道是“动画”的意思。但CSS3中的[Animation](http://www.w3.org/TR/css3-animations/)与HTML5中的Canvas绘制动画又不同，[Animation](http://www.w3.org/TR/css3-animations/)只应用在页面上已存在的DOM元素上，而且他跟Flash和JavaScript以及jQuery制作出来的动画效果又不一样，因为我们使用CSS3的[Animation](http://www.w3.org/TR/css3-animations/)制作动画我们可以省去复杂的js,jquery代码（像我这种不懂js的人来说是件很高兴的事了），只是有一点不足之处，我们运用[Animation](http://www.w3.org/TR/css3-animations/)能创建自己想要的一些动画效果，但是有点粗糙，如果想要制作比较好的动画，我见意大家还是使用flash或js等。虽然说[Animation](http://www.w3.org/TR/css3-animations/)制作出来的动画简单粗糙，但我想还是不能减少我们大家对其学习的热情。

### Keyframes

在开始介绍[Animation](http://www.w3.org/TR/css3-animations/)之前我们有必要先来了解一个特殊的东西，那就是"[Keyframes](http://www.w3.org/TR/css3-animations/)",我们把他叫做“关键帧”，玩过flash的朋友可能对这个东西并不会陌生。下面我们就一起来看看这个“Keyframes”是什么东西。前面我们在使用transition制作一个简单的transition效果时，我们包括了初始属性和最终属性，一个开始执行动作时间和一个延续动作时间以及动作的变换速率，其实这些值都是一个中间值，如果我们要控制的更细一些，比如说我要第一个时间段执行什么动作，第二个时间段执行什么动作（换到flash中说，就是第一帧我要执行什么动作，第二帧我要执行什么动作），这样我们用Transition就很难实现了，此时我们也需要这样的一个“关键帧”来控制。那么CSS3的[Animation](http://www.w3.org/TR/css3-animations/)就是由“keyframes”这个属性来实现这样的效果。下面我们一起先来看看Keyframes:

[Keyframes](http://www.w3.org/TR/css3-animations/)具有其自己的语法规则，他的命名是由"@keyframes"开头，后面紧接着是这个“动画的名称”加上一对花括号“{}”，括号中就是一些不同时间段样式规则，有点像我们css的样式写法一样。对于一个"@keyframes"中的样式规则是由多个百分比构成的，如“0%”到"100%"之间，我们可以在这个规则中创建多个百分比，我们分别给每一个百分比中给需要有动画效果的元素加上不同的属性，从而让元素达到一种在不断变化的效果，比如说移动，改变元素颜色，位置，大小，形状等，不过有一点需要注意的是，我们可以使用“fromt”“to”来代表一个动画是从哪开始，到哪结束，也就是说这个 "from"就相当于"0%"而"to"相当于"100%",值得一说的是，其中"0%"不能像别的属性取值一样把百分比符号省略，我们在这里必须加上百分符号（“%”）如果没有加上的话，我们这个keyframes是无效的，不起任何作用。因为keyframes的单位只接受百分比值。

Keyframes可以指定任何顺序排列来决定Animation动画变化的关键位置。其具体语法规则如下：

```javascript
 keyframes-rule: '@keyframes' IDENT '{' keyframes-blocks '}';
 keyframes-blocks: [ keyframe-selectors block ]* ;
 keyframe-selectors: [ 'from' | 'to' | PERCENTAGE ] [ ',' [ 'from' | 'to' | PERCENTAGE ] ]*;
```

我把上面的语法综合起来

```javascript
  @keyframes IDENT {
     from {
       Properties:Properties value;
     }
     Percentage {
       Properties:Properties value;
     }
     to {
       Properties:Properties value;
     }
   }
   或者全部写成百分比的形式：
   @keyframes IDENT {
      0% {
         Properties:Properties value;
      }
      Percentage {
         Properties:Properties value;
      }
      100% {
         Properties:Properties value;
      }
    }
 
```

其中IDENT是一个动画名称，你可以随便取，当然语义化一点更好，Percentage是百分比值，我们可以添加许多个这样的百分比，Properties为css的属性名，比如说left,background等，value就是相对应的属性的属性值。值得一提的是，我们from和to 分别对应的是0%和100%。这个我们在前面也提到过了。到目前为止支技animation动画的只有webkit内核的浏览器，所以我需要在上面的基础上加上-webkit前缀，据说Firefox5可以支持css3的 animation动画属性。

我们来看一个[W3C](http://www.w3.org/TR/css3-animations/)官网的实例

```javascript
  @-webkit-keyframes 'wobble' {
     0% {
        margin-left: 100px;
        background: green;
     }
     40% {
        margin-left: 150px;
        background: orange;
     }
     60% {
        margin-left: 75px;
        background: blue;
     }
     100% {
        margin-left: 100px;
        background: red;
     }
  }     
```

这里我们定义了一个叫“wobble”的动画，他的动画是从0%开始到100%时结束，从中还经历了一个40%和60%两个过程，上面代码具体意思是：wobble动画在0%时元素定位到left为100px的位置背景色为green，然后40%时元素过渡到left为150px的位置并且背景色为orange，60%时元素过渡到left为75px的位置,背景色为blue，最后100%结束动画的位置元素又回到起点left为100px处,背景色变成red。假设置我们只给这个动画有10s的执行时间，那么他每一段执行的状态如下图所示：

![img](http://cdn1.w3cplus.com/cdn/farfuture/KZN7qhpnq2Iyu4-PVAt3sEX5XkzSWzvONjibSOY9zfs/mtime:1341237744/sites/default/files/keyframes.png)

Keyframes定义好了以后，我们需要怎么去调用刚才定义好的动画“wobble”

CSS3的animation类似于transition属性，他们都是随着时间改变元素的属性值。他们主要区别是transition需要触发一个事件(hover事件或click事件等)才会随时间改变其css属性；而animation在不需要触发任何事件的情况下也可以显式的随着时间变化来改变元素css的属性值，从而达到一种动画的效果。这样我们就可以直接在一个元素中调用animation的动画属性,基于这一点，css3的animation就需要明确的动画属性值，这也就是回到我们上面所说的，我们需要keyframes来定义不同时间的css属性值,达到元素在不同时间段变化的效果。

下面我们来看看怎么给一个元素调用animation属性

```
.demo1 {
     width: 50px;
     height: 50px;
     margin-left: 100px;
     background: blue;
     -webkit-animation-name:'wobble';/*动画属性名，也就是我们前面keyframes定义的动画名*/
     -webkit-animation-duration: 10s;/*动画持续时间*/
     -webkit-animation-timing-function: ease-in-out; /*动画频率，和transition-timing-function是一样的*/
     -webkit-animation-delay: 2s;/*动画延迟时间*/
     -webkit-animation-iteration-count: 10;/*定义循环资料，infinite为无限次*/
     -webkit-animation-direction: alternate;/*定义动画方式*/
  }
```

CSS Animation动画效果将会影响元素相对应的css值，在整个动画过程中，元素的变化属性值完全是由animation来控制，动画后面的会覆盖前面的属性值。如上面例子：因为我们这个demo只是在不同的时间段改变了demo1的背景色和左边距，其默认值是：margin-left:100px;background: blue；但当我们在执行动画0%时，margin-left:100px,background:green；当执行到40%时，属性变成了：margin-left:150px;background:orange;当执行到60%时margin-left:75px;background:blue;当动画 执行到100%时：margin-left:100px;background: red;此时动画将完成，那么margin-left和background两个属性值将是以100%时的为主,他不会产生叠加效果，只是一次一次覆盖前一次出将的css属性。就如我们平时的css一样，最后出现的权根是最大的。当动画结束后，样式回到默认效果。

我们可以看一张来自[w3c](http://www.w3.org/TR/css3-animations/)官网有关于css3的[animation](http://www.w3.org/TR/css3-animations/)对属性变化的过程示意图

![img](http://cdn2.w3cplus.com/cdn/farfuture/LDNfVU-RoRMIS34CqUNgQwJH6c3w8GrHFHXS2Rjfw2c/mtime:1341237470/sites/default/files/animation-2.png)

从上面的Demo中我们可以看出animation和transition一样有自己相对应的属性，那么在animation主要有以下几种：animation-name;animation-duration;animation-timing-function;animation-delay;animation-iteration-count;animation-direction;animation-play-state。下面我们分别来看看这几个属性的使用

### animation-name

**语法：**

```
  animation-name: none | IDENT[,none | IDENT]*;
```

**取值说明：**

animation-name:是用来定义一个动画的名称，其主要有两个值：IDENT是由Keyframes创建的动画名，换句话说此处的IDENT要和Keyframes中的IDENT一致，如果不一致,将不能实现任何动画效果；none为默认值，当值为none时，将没有任何动画效果。另外我们这个属性跟前面所讲的transition一样，我们可以同时附几个animation给一个元素，我们只需要用逗号“，”隔开。

### animation-duration

**语法：**

```
  animation-duration: <time>[,<time>]*
```

**取值说明：**

animation-duration是用来指定元素播放动画所持续的时间长，取值:<time>为数值，单位为s （秒.）其默认值为“0”。这个属性跟transition中的[transition-duration](http://www.w3cplus.com/content/css3-transition)使用方法是一样的。

### animation-timing-function

**语法：**

```
   animation-timing-function:ease | linear | ease-in | ease-out | ease-in-out | cubic-bezier(<number>, <number>, <number>, <number>) [, ease | linear | ease-in | ease-out | ease-in-out | cubic-bezier(<number>, <number>, <number>, <number>)]* 
```

**取值说明：**

animation-timing-function:是指元素根据时间的推进来改变属性值的变换速率，说得简单点就是动画的播放方式。他和transition中的[transition-timing-function](http://www.w3cplus.com/content/css3-transition)一样，具有以下六种变换方式：ease;ease-in;ease-in-out;linear;cubic-bezier。具体的使用方法大家可以点[这里](http://www.w3cplus.com/content/css3-transition)，查看其中transition-timing-function的使用方法。

### animation-delay

**语法：**

```
  animation-delay: <time>[,<time>]*
```

**取值说明：**

animation-delay:是用来指定元素动画开始时间。取值为<time>为数值，单位为s(秒)，其默认值也是0。这个属性和[transition-delay](http://www.w3cplus.com/content/css3-transition)y使用方法是一样的。

### animation-iteration-count

**语法：**

```
  animation-iteration-count:infinite | <number> [, infinite | <number>]* 
```

**取值说明：**

animation-iteration-count是用来指定元素播放动画的循环次数，其可以取值<number>为数字，其默认值为“1”；infinite为无限次数循环。

### animation-direction

**语法：**

```
  animation-direction: normal | alternate [, normal | alternate]* 
```

**取值说明：**

animation-direction是用来指定元素动画播放的方向，其只有两个值，默认值为normal，如果设置为normal时，动画的每次循环都是向前播放；另一个值是alternate，他的作用是，动画播放在第偶数次向前播放，第奇数次向反方向播放。

### Animation-play-state

**语法：**

```
animation-play-state:running | paused [, running | paused]* 
```

**取值说明：**

animation-play-state主要是用来控制元素动画的播放状态。其主要有两个值，running和paused其中running为默认值。他们的作用就类似于我们的音乐播放器一样，可以通过paused将正在播放的动画停下了，也可以通过running将暂停的动画重新播放，我们这里的重新播放不一定是从元素动画的开始播放，而是从你暂停的那个位置开始播放。另外如果暂时了动画的播放，元素的样式将回到最原始设置状态。这个属性目前很少内核支持，所以只是稍微提一下。

上面我们分别介绍了animation中的各个属性的语法和取值，那么我们综合上面的内容可以给animation属性一个速记法：

```
  animation:[<animation-name> || <animation-duration> || <animation-timing-function> || <animation-delay> || <animation-iteration-count> || <animation-direction>] [, [<animation-name> || <animation-duration> || <animation-timing-function> || <animation-delay> || <animation-iteration-count> || <animation-direction>] ]* 
```

 

如下图所示

![img](http://cdn2.w3cplus.com/cdn/farfuture/JKnJarl8UI_49aRemGLqW-JQZzr0GO7rmXEaN7vSfkM/mtime:1341237472/sites/default/files/animation-pro.png)

**兼容的浏览器**

前面我也简单的提过，CSS3的animation到目前为止只支持webkit内核的浏览器，因为最早提出这个属性的就是safari公司,据说Firefox5.0+将支持Animation。如图所示

![img](http://cdn1.w3cplus.com/cdn/farfuture/uN_cQLGCwEnWYuTR1kEzMpQU78CExCg-j4310vmn9Nk/mtime:1341237472/sites/default/files/animation-browers.png)

那么到此为止，我们主要一起学习了有关animation的理论知识，下面我们一起来看两个实例制作过程，来加强对animation的实践能力

通过上面，我想大家对CSS3的Transition属性的使用有一定的概念存在了，下面为了加强大家在这方面的使用，我们一起来看下面的DEMO。我们通过实践来巩固前面的理论知识，也通过实践来加强transition的记忆。

最后，提一下，animation的不同状态下的连续动画的连写方式：

```
<div class="element">小火箭</div>

.element { animation: fadeIn 1s, float .5s 1s infinite; }  /* 我淡出, 需要1秒；我1秒后开始无限漂浮 */
```

还有就是标签嵌套与独立动画：

```
<div class="element-wrap"><div class="element">小火箭</div></div>

.element-wrap { animation: fadeIn 1s; }          /* 我淡出, 需要1秒 */
.element { animation: float .5s 1s infinite; }   /* 我1秒后开始无限漂浮 */
```

有人可能会奇怪了。`animation`本身就支持多动画并行，你还搞个标签嵌套，没有任何使用的理由啊！没错，单纯看我们这个例子，确实是这样。但是：

**① 提取公用动画**
这类多屏动画是有N多元素同时执行不同的动画。比方说，火箭是淡出，然后上下漂浮；火箭的火焰是淡出，然后大小变化；黑洞是淡出，然后左右随波。你如何实现？

如果纯粹借助`animation`语法，应该是：

```
.element1 { animation: fadeIn 1s, float .5s 1s infinite; }  /* 我淡出, 需要1秒；我1秒后开始无限漂浮 */
.element2 { animation: fadeIn 1s, size .5s 1s infinite; }   /* 我淡出, 需要1秒；我1秒后开始大小变化 */
.element3 { animation: fadeIn 1s, move .5s 1s infinite; }   /* 我淡出, 需要1秒；我1秒后开始左右移动 */
```

 

可以看到，淡出是公用的动画效果，我们可以借助嵌套标签，实现公用语法的合并，方面后期维护：

```
.element-wrap { animation: fadeIn 1s; }          /* 大家都1秒淡出 */
.element1 { animation: float .5s 1s infinite; }  /* 我1秒后无限漂浮 */
.element2 { animation: size .5s 1s infinite; }   /* 我1秒后忽大忽小 */
.element3 { animation: move .5s 1s infinite; }   /* 我1秒后左右移动 */
```

 

**②避免变换冲突**
有个元素动画是边360度旋转、边放大(从0放大到100%)，像这种具有典型特征的动画我们显然要独立提取与公用的：

```
@keyframes spin { /* transform: rotate... */ }
@keyframes zoomIn { /* transform: scale... */ }
```

 

好了，现在问题来了，变放大边旋转：

```
.element { animation: spin 1s, zoomIn 1s; }  /* 旋转：啊，完蛋啦，我被放大覆盖啦！ */
```

 

由于都是使用transform, 发生了残忍的覆盖。当然，有好事的人会说，你使用`zoom`不就好了！确实，如果只是移动端，使用`zoom`确实棒棒哒！但是，我们这个企业活动，PC是主战场，因此，FireFox浏览器（FF不识zoom）是不能无视的。

怎么办？重新建一个名为`spinZoomIn`的动画关键帧描述还是？

对啊，你直接外面套一层标签不就万事大吉了 ![经验分享：多屏复杂动画CSS技巧三则](http://mat1.gtimg.com/www/mb/images/face/29.gif)：

```
.element-wrap { animation: spin 1s; }   /* 我转转转 */
.element { animation: zoomIn 1s; }      /* 我大大大 */
```

**对于transform-origin属性：**

```
#job_page3 .j3_01 {
    -webkit-transform-origin: 50% 75%;
    -webkit-animation: scale3 0.8s both;
}
```

可以这么写，就可以在只是改变基点的情况下，运用同一个动画，非常方便。

 