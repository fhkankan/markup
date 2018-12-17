[TOC]

# JQuery

```
jquery是一个函数库，一个js文件，页面用script标签引入这个js文件就可以使用。

jQuery的版本分为1.x系列和2.x、3.x系列，1.x系列兼容低版本的浏览器，2.x、3.x系列放弃支持低版本浏览器

1、http://jquery.com/ 官方网站
2、https://code.jquery.com/ 版本下载
```

三种写法

```javascript
// 写法一：
jQuery(document).ready(function($){
    // JQuery代码
})
// 写法二
$(document).ready(function){
    // JQuery代码
}
// 写法三
$(function(){
   //QJQuery代码
})
```



## 选择元素

### 基本选择器

```javascript
jquery选择器可以快速地选择元素，选择规则和css样式相同.
// id
$('#myId') //选择id为myId的网页元素
// class
$('.myClass') // 选择class为myClass的元素
// element
$('li') //选择所有的li元素
// 复合
$('selector1, selector2') // 将selector1和selector2匹配的元素合并返回
// 通配符
$('*') // 页面上所有元素
```

### 层级选择器

```javascript
// ancestor descendan
$('#ul1 li span') //选择id为为ul1元素下的所有li下的span元素
// parent > child
$("form > input") //父级为form，子级为input的元素
// prev + next
$("div + img") //div后的<img>元素
// prev ~ siblings
$("div ~ ul") // div同辈的ul元素
```

### 过滤选择器

简单过滤器

```javascript
$("tr:first") //匹配表格第一行
$("tr:last") //匹配表格最后一行
$("tr:even")  //匹配索引值为偶数的行
$("tr:odd")  //匹配索引值为奇数的行
$("div:eq(1)")  //匹配第二个div
$("div:gt(1)")  //匹配第二个及以上的div
$("div:lt(1)")  //匹配第二个及以下的div
$(":header")  // 匹配全部的标题元素
$("input:not(:checked)")  // 匹配没有被选中的input元素
$(":animated")  //匹配所有正在执行的动画
```

内容过滤器

```javascript
$("li:contians("DOM")")  //匹配含有“DOM”文本内容的li元素
$("td:empty")  //匹配不包含子元素或者文本的单元格
$("td:has(p)")  //匹配表格的单元格中含有p标记的单元格
$("td:parent")  //匹配含有子元素或者文本的单元格
```

子元素过滤器

```javascript
$("ul li:first-child")  // 匹配ul元素中的第一个子元素li
$("ul li:last-child")  // 匹配ul元素中的最后一个子元素li
$("ul li:only-child")  // 匹配只含有一个li元素的ul元素中的li
$("ul li:nth-child(index/even/odd/equation)")  //匹配ul中索引为第index个元素(从1开始)
```

### 属性选择器

```javascript
$("div[name]") //匹配含有name属性的div元素
$('input[name='first']') // 选择name属性等于first的input元素
$('input[name!='first']') // 选择name属性不是first的input元素
$('input[name*='first']') // 选择name属性中含有first值的input元素
$('input[name^='first']') // 选择name属性以first开头的input元素
$('input[name$='first']') // 选择name属性以first结尾的input元素
$('input[name*='first']') // 选择name属性中含有first值的input元素
$('input[id][name='first']') // 选择具有id属性且name属性为first的input元素
```

### 选择集转移

```
$('#box').prev(); //选择id是box的元素前面紧挨的同辈元素
$('#box').prevAll(); //选择id是box的元素之前所有的同辈元素
$('#box').next(); //选择id是box的元素后面紧挨的同辈元素
$('#box').nextAll(); //选择id是box的元素后面所有的同辈元素
$('#box').parent(); //选择id是box的元素的父元素
$('#box').children(); //选择id是box的元素的所有子元素
$('#box').siblings(); //选择id是box的元素的同级元素
$('#box').find('.myClass'); //选择id是box的元素内的class等于myClass的元素
```

### 判断是否选中的元素

```
jquery有容错机制，即使没有找到元素，也不会出错，可以用length属性来判断是否找到了元素.
length等于0，就是没选择到元素;
length大于0，就是选择到了元素。
```

## jQuery对象

从jQuery选择器返回的对象都是jQuery对象，而不是原生DOM对象(HTMLElement)。

JQuery对象封装了原生对象，并支持后面的方法。若需要使用原生对象，则调用jQuery对象上的get()方法

所有的jQuery方法都返回支持元素集合的jQuery对象。即使first()方法也返回一个仅具有一个元素的集合的jQuery对象。

```javascript
// eq()方法返回jQuery对象
var $obj = $('#ul li').eq(0)

// get()方法返回来自jQuery对象的原生DOM对象
var body = $("body").get(0);
```

## 链式调用

```
jquery对象的方法会在执行完后返回这个jquery对象，所有jquery对象的方法可以连起来写.
$('#div1') // id为div1的元素
.children('ul') //该元素下面的ul子元素
.slideDown('fast') //高度从零变到实际高度来显示ul元素
.parent()  //跳到ul的父元素，也就是id为div1的元素
.siblings()  //跳到div1元素平级的所有兄弟元素
.children('ul') //这些兄弟元素中的ul子元素
.slideUp('fast');  //高度实际高度变换到零来隐藏ul元素
```

## 控制页面

### 元素内容和值

```javascript
// 获取或设置文本内容
$("div").text([str])
// 获取或设置html内容
$("div").html([str])
// 获取或设置元素值
$("#username").val([val])

eg
// 取出html内容
var $htm = $('#div1').html();
// 设置html内容
$('#div1').html('<span>添加文字</span>');
```

### DOM节点

- 查找节点

选择元素

- 创建节点

```javascript
$(document).ready(function(){
    // 方法一
    var $p = $("<p></p>")
    $p.html("<span>你好</span>")
    $("body").append($p)
    // 方法二
    var $textP = $("<p><sapn>你好</span></p>")
    $("body").append($textP)
    // 方法三
    $("body").append("<p><sapn>你好</span></p>")
    alert($("p").text())
})
```

- 插入节点

````javascript
// 内部插入
父元素.append(子元素)  //当前元素的内部后面放入另外一个元素
子元素.appendTo(父元素)  //当前元素放置到另一元素的内部的后面
父元素.prepend(子元素) //当前元素的内部的前面放入另外一个元素
子元素.prependTo(父元素) //当前元素放置到另一元素的内部的前面

// 外部插入
前元素.after(后元素)  //当前元素的后面放入另一个元素
后元素.insertafter(前元素) //当前元素放置到另一元素的后面
后元素.before(前元素) //当前元素的前面放入另一个元素
前元素.insertbefore(后元素) //当前元素放置到另一元素的前面
````

- 包装节点

```javascript
子元素.wrap(父元素) //插入一个元素作为选择器指定的元素的父对象
子元素.wrapAll(父元素)  //将所有选定元素包到单个父元素中，若匹配元素间存在不匹配元素，则放到新元素后面
元素.wrapInner(内容)  //将内容包裹住元素
eg:
<p class="myClass">Hello</p>
$("p").wrapInner("<strong></strong>")
```

- 删除节点

```
元素.unwrap() // 删除选择器返回的元素的直接父元素，消除wrap()方法
元素.remove() // 删除选择器返回的元素
元素.detach() // 删除选择器返回的元素，返回被删除的元素集合，后期可重添加
元素.empty() // 删除选择器返回的元素中所有的子元素,并不删除该元素
```

- 复制节点

```
clone() // 克隆匹配的DOM元素且选中这些克隆的副本
clone(boolen) // 当boolen为true时，表示克隆匹配的元素及其所有的事件处理，且选中这些克隆的副本；当boolen为false时，表示不复制元素的事件处理
```

- 替换节点

```
新元素.replaceAll(旧元素)  //使用匹配的元素替换所有selector匹配的元素
旧元素.replaceWith(新元素)  //将所有匹配的元素替换为指定的HTML或DOM元素
```

### 元素属性

```javascript
attr(name)  //获取匹配的第一个元素的属性值，无值时返回undefined
attr(key, value)  //为所有匹配元素设置一个属性值
attr(key, fn)  //为所有匹配元素设置一个函数返回的属性值
attr(properties)  //为所有匹配元素以集合({key1:value1,key2:value2})形式同时设置多个属性
removeAttr(name)  //为所有匹配元素删除一个属性

eg:
$("img").attr("src")
$("img").attr("title", "你好")
$("img").attr({src:"test.gif",title: "示例"})
$("img").removeAttr("title")


# prop() 取出或设置元素除了css之外的某个属性的值
// 取出图片的地址
var $src = $('#img1').prop('src');
// 设置图片的地址和alt属性
$('#img1').prop({src: "test.jpg", alt: "Test Image" });
```

### CSS样式

```javascript
# 操作CSS类
addClass(class)  //为所有匹配元素添加指定的CSS类名
removeClass(class) //从所有匹配元素汇总删除全部或者指定的CSS类
toggleClass(class)  //如果存在(不存在)就删除(添加)一个CSS类
toggleClass(class, switch)  //若switch为true，则添加CSS类，否则则删除
eg:
$("#div1").addClass("divClass2") //为id为div1的对象追加样式divClass2
$("#div1").removeClass("divClass")  //移除id为div1的对象的class名为divClass的样式
$("#div1").removeClass("divClass divClass2") //移除多个样式，中间空格
$("#div1").toggleClass("anotherClass") //重复切换anotherClass样式

# 操作CSS属性
css(name)  //返回第一个匹配元素的样式属性
css(name, value)  //为所有匹配元素的指定样式设置值
css(properties)  //以{属性:值，属性:值}的形式为所有匹配的元素设置样式属性
eg:
// 获取div的样式
$("div").css("width");
//设置div的样式
$("div").css("width","30px");
$("div").css({fontSize:"30px",color:"red"});
```
## 动画效果

- 基本动画效果

```
hide(speed, [callback])  // 隐藏
show(speed, [callback])  // 显示
toggle(speed, [callback]) //切换元素的可见状态
```

- 淡入淡出

```
fadeIn(speed, [callback])  //淡入,增大不透明度
fadeOut(speed, [callback])  //淡出，减小不透明度
fadeTo(speed, opacity, [callback])  // 将匹配元素的不透明度以渐进的方式调整到指定的参数
fadeToggle(speed, [callback])  //切换淡入淡出
```

- 滑动效果

```
slideDown(speed, [callback])  //向下展开
slideUp(speed, [callback])  //向上卷起
slideToggle(speed, [callback])  // 通过高度变化动态切换元素的可见性
```

- 自定义动画

```javascript
//通过animate方法可以设置元素某属性值上的动画，可以设置一个或多个属性值，动画执行完成后会执行一个函数。
animate(params, speed, swing,callback)
// animate参数：
参数一：要改变的样式属性值，写成字典的形式
参数二：动画持续的时间，默认400，单位为毫秒，一般不写单位
参数三：动画曲线，默认为‘swing’，缓冲运动，还可设置‘linear’，匀速运动
参数四：动画回调函数，动画完成后执行的匿名函数

eg:
$('#div1').animate({
    width:300,
    height:300
},1000,'swing',function(){
    alert('done!');
});

// 停止动画
stop(clearQueue, gotoEnd)
// 参数
clearQueue:表示是否清空尚未执行完的动画队列(true时表示清空)
gotoEnd:表示是否让正在执行的动画直接到达动画结束时的状态(true时表示直接到达结束状态)
```

## 循环

```javascript
// 对jquery选择的对象集合分别进行操作，需要用到jquery循环操作，此时可以用对象上的each方法：
$(function(){
    $('.list li').each(function(){
        $(this).html($this.index());
    })
})
```

## 事件处理

### 页面加载响应事件

若将获取元素的语句写到页面头部，会因为元素还没有加载而出错。

jquery提供了ready方法解决这个问题，它的速度比原生的 window.onload 更快。

```javascript
// 使用了documnet对象
$(document).ready(function(){
    ...
});
//简写                             
$(function(){
    ...
})
```

与window.onload()区别

```
1.在页面上可以无限制地使用$(document).ready()方法，各个方法不冲突，会按照在代码中的顺序执行，但是一个页面只能有一个window.onload()
2.在一个文档完全下载到浏览器时才会响应window.onload()方法，但是只需DOM元素就绪后就可调用$(document).ready()方法。
故$(document).ready()方法优于window.onload()
```

### 常用事件

|        | 方法           | 说明                                                         |
| ------ | -------------- | ------------------------------------------------------------ |
| 焦点   | blur([fn])     | (在每一个匹配的元素绑定一个处理函数)，触发元素的失去焦点事件 |
|        | focus([fn])    | (在每一个匹配的元素绑定一个处理函数)，触发元素的获得焦点事件 |
|        | change(fn)     | (在每一个匹配的元素绑定一个处理函数)，触发元素的值改变事件   |
| 点击   | click([fn])    | (在每一个匹配的元素绑定一个处理函数)，触发元素的单击事件     |
|        | dbclick([fn])  | (在每一个匹配的元素绑定一个处理函数)，触发元素的双击事件     |
| 错误   | error([fn])    | (在每一个匹配的元素绑定一个处理函数)，触发元素的错误事件     |
| 按键   | keydown([fn])  | (在每一个匹配的元素绑定一个处理函数)，触发元素的按键按下事件 |
|        | keyup([fn])    | (在每一个匹配的元素绑定一个处理函数)，触发元素的按键释放事件 |
|        | keypress([fn]) | (在每一个匹配的元素绑定一个处理函数)，触发元素的敲击按键事件 |
| 加载   | load(fn)       | (在每一个匹配的元素绑定一个处理函数)，触发元素的加载完毕事件 |
| 卸载   | unload(fn)     | (在每一个匹配的元素绑定一个处理函数)，触发元素的卸载事件     |
| 鼠标   | mousedown(fn)  | (在每一个匹配的元素绑定一个处理函数)，触发元素的鼠标单击事件 |
|        | mousemove(fn)  | (在每一个匹配的元素绑定一个处理函数)，触发元素的鼠标移动事件 |
|        | mouseout(fn)   | (在每一个匹配的元素绑定一个处理函数)，触发元素的鼠标离开事件 |
|        | mouseover(fn)  | (在每一个匹配的元素绑定一个处理函数)，触发元素的鼠标移入事件 |
|        | mouseup(fn))   | (在每一个匹配的元素绑定一个处理函数)，触发元素的鼠标单击释放事件 |
| 窗口   | resize(fn)     | (在每一个匹配的元素绑定一个处理函数)，当文档串口改变大小触发 |
| 滚动条 | scroll(fn)     | (在每一个匹配的元素绑定一个处理函数)，当滚动条变化时触发     |
| 文本框 | select(fn)     | (在每一个匹配的元素绑定一个处理函数)，当文本框选中某段文本时触发 |
| 表单   | submit(fn)     | (在每一个匹配的元素绑定一个处理函数)，当表单提交时触发       |

### 绑定事件

```javascript
// 为元素绑定事件
选择的元素.bind(type, [data], fn)
// 参数
type：事件类型
data：可选参数，作为event.data属性值传递为时间对象的额外数据对象，常不用
fn：绑定事件的处理程序

// 移除绑定
选择的元素.unbind([type], [data])
// 参数
type：事件类型
data：要从每个匹配元素的事件中反绑定的事件处理函数

// 给元素绑定click事件，可以用如下方法：
$('#btn1').click(function(){
    // 内部的this指的是原生对象
    // 使用jquery对象用 $(this)
})

// 获取元素的索引值 
获得匹配元素相对于其同胞元素的索引位置，此时用index()

```

避免重复触发

```javascript
// 为元素绑定一个一次性的事件处理函数，这个事件的处理函数智慧被执行一次
选择的元素.one(type, [data], fn)

// 为元素绑定一个事件处理函数，再次给该元素添加相同事件时不会累加绑定
选择的元素.off(type).on(type, fn)
```

### 模拟用户操作

- 模拟用户的操作触发

```javascript
triggerHandler()  // 不会导致浏览器同名的默认行为被执行
trigger()  // 会导致浏览器同名默认行为被执行

eg:
$(document).ready(function(){
    $("input:button").bind("click", function(event, msg1, msg2){
        alert(msg1, msg2);
    }).trigger("click", ["欢迎访问！"])
})
```

- 模拟悬停事件

```javascript
hover(over, out)
// 参数
over:当鼠标在移动到匹配元素上时触发的函数
out:当鼠标在移出匹配元素上时触发的函数

eg:
$(document).ready(function(){
    $("#pic").hover(function(){
        $(this).attr("border", 1);  //为图片加边框
    }, function(){
        $(this).attr("border", 0);  //去除图片边框
    })
})
```

- 模拟鼠标连续单击

```javascript
// 属于jQuery中的click事件，若要删除可用unbind('click')
toggle(odd, even)
// 参数
odd:奇次单击按钮时触发的函数
even:偶次单击按钮时触发的函数

eg:
$("#tool").togle(
	function(){$("#tip").css("display", "")},
    function(){$("#tip").css("display", "none")}
)
```



## 事件冒泡

```
在一个对象上触发某类事件，无论是否有这个对象的时间处理程序，不仅自己执行，还会向这个对象的父级对象传播，从里到外，父级对象所有同类事件都将被激活，直到到达了对象层次的最顶层，即document对象（body/html）。

阻止事件冒泡
执行函数有参数event，且执行
event.stopPropagation();

阻止默认行为
执行函数有参数event，且执行
event.preventDefault();

合并阻止写法：
return false;
```

## 事件委托

```
事件委托就是利用冒泡的原理，把事件加到父级上，通过判断事件来源的子集，执行相应的操作，事件委托首先可以极大减少事件绑定次数，提高性能；其次可以让新加入的子元素也可以拥有相同的操作。

# 一般写法：
$(function(){
    $ali = $('#list li');
    $ali.click(function(){
    	$(this).css({background:'red'});
    });
})	

# 委托写法
$(function(){
    $list = $('#list');
    $list.delegate('li', 'click', function() {        				$(this).css({background:'red'});
    });
})
```



## 表单验证（正则）

```
1、什么是正则表达式： 
能让计算机读懂的字符串匹配规则。

2、正则表达式的写法：
var re=new RegExp('规则', '可选参数');
var re=/规则/参数;

3、规则中的字符 
1）普通字符匹配：
如：/a/ 匹配字符 ‘a’，/a,b/ 匹配字符 ‘a,b’

2）转义字符匹配：
\d 匹配一个数字，即0-9
\D 匹配一个非数字，即除了0-9
\w 匹配一个单词字符（字母、数字、下划线）
\W 匹配任何非单词字符。等价于[^A-Za-z0-9_]
\s 匹配一个空白符
\S 匹配一个非空白符
\b 匹配单词边界
\B 匹配非单词边界
. 匹配一个任意字符

var sTr01 = '123456asdf';
var re01 = /\d+/;
//匹配纯数字字符串
var re02 = /^\d+$/;
alert(re01.test(sTr01)); //弹出true
alert(re02.test(sTr01)); //弹出false
4、量词：对左边的匹配字符定义个数 
? 出现零次或一次（最多出现一次）
+ 出现一次或多次（至少出现一次）
* 出现零次或多次（任意次）
{n} 出现n次
{n,m} 出现n到m次
{n,} 至少出现n次

5、任意一个或者范围 
[abc123] : 匹配‘abc123’中的任意一个字符
[a-z0-9] : 匹配a到z或者0到9中的任意一个字符

6、限制开头结尾 
^ 以紧挨的元素开头
$ 以紧挨的元素结尾

7、修饰参数：
g： global，全文搜索，默认搜索到第一个结果接停止
i： ingore case，忽略大小写，默认大小写敏感

8、常用函数 
test
用法：正则.test(字符串) 匹配成功，就返回真，否则就返回假

正则默认规则 
匹配成功就结束，不会继续匹配，区分大小写

常用正则规则

//用户名验证：(数字字母或下划线6到20位)
var reUser = /^\w{6,20}$/;

//邮箱验证：        
var reMail = /^[a-z0-9][\w\.\-]*@[a-z0-9\-]+(\.[a-z]{2,5}){1,2}$/i;

//密码验证：
var rePass = /^[\w!@#$%^&*]{6,20}$/;

//手机号码验证：
var rePhone = /^1[34578]\d{9}$/;
```

## 常用方法

去除空格

```
$.trim("字符串")
```

## JSON

```
json是 JavaScript Object Notation 的首字母缩写，单词的意思是javascript对象表示法，这里说的json指的是类似于javascript对象的一种数据格式

两种结构：
1、对象结构
对象结构是使用大括号“{}”括起来的，大括号内是由0个或多个用英文逗号分隔的“关键字:值”对（key:value）构成的。
语法：
{
    "键名1":值1,
    "键名2":值2,
    "键名n":值n
}
说明：
对象结构是以“{”开始，到“}”结束。其中“键名”和“值”之间用英文冒号构成对，两个“键名:值”之间用英文逗号分隔。
注意，这里的键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。

2、JSON数组结构
JSON数组结构是用中括号“[]”括起来，中括号内部由0个或多个以英文逗号“,”分隔的值列表组成。
语法：
[
    {
        "键名1":值1,
        "键名2":值2
    },
    {
        "键名3":值3,
        "键名4":值4
    },
    ……
]
说明：
arr指的是json数组。数组结构是以“[”开始，到“]”结束，这一点跟JSON对象不同。 在JSON数组中，每一对“{}”相当于一个JSON对象。 
注意，这里的键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。
```

## AJAX

ajax技术的目的是让javascript发送http请求，与后台通信，获取数据和信息。ajax技术的原理是实例化xmlhttp对象，使用此对象与后台通信。ajax通信的过程不会影响后续javascript的执行，从而实现异步。

**同步和异步** 
现实生活中，同步指的是同时做几件事情，异步指的是做完一件事后再做另外一件事，程序中的同步和异步是把现实生活中的概念对调，也就是程序中的异步指的是现实生活中的同步，程序中的同步指的是现实生活中的异步。

**局部刷新和无刷新** 
ajax可以实现局部刷新，也叫做无刷新，无刷新指的是整个页面不刷新，只是局部刷新，ajax可以自己发送http请求，不用通过浏览器的地址栏，所以页面整体不会刷新，ajax获取到后台数据，更新页面显示数据的部分，就做到了页面局部刷新。

**同源策略** 
ajax请求的页面或资源只能是同一个域下面的资源，不能是其他域的资源，这是在设计ajax时基于安全的考虑。特征报错提示：

```
XMLHttpRequest cannot load https://www.baidu.com/. No  
'Access-Control-Allow-Origin' header is present on the requested resource.  
Origin 'null' is therefore not allowed access.
```

### 参数

```
$.ajax([setings])
```

settings

| key               | type     | Dec                                                          |
| ----------------- | -------- | ------------------------------------------------------------ |
| options           | Object   | AJAX 请求设置。所有选项都是可选的。                          |
| async             | Boolean  | 默认值: true。默认设置下，所有请求均为异步请求。如果需要发送同步请求，请将此选项设置为 false。注意，同步请求将锁住浏览器，用户其它操作必须等待请求完成才可以执行。 |
| beforeSend(XHR)   | Function | 发送请求前可修改 XMLHttpRequest 对象的函数，如添加自定义 HTTP 头。XMLHttpRequest 对象是唯一的参数。
这是一个 Ajax 事件。如果返回 false 可以取消本次 ajax 请求。 |
| cache             | Boolean  | 默认值: true，dataType 为 script 和 jsonp 时默认为 false。设置为 false 将不缓存此页面。 |
| complete(XHR, TS) | Function | 请求完成后回调函数 (请求成功或失败之后均调用)。参数： XMLHttpRequest 对象和一个描述请求类型的字符串。
这是一个 Ajax 事件。 |
| contentType       | String   | 默认值: "application/x-www-form-urlencoded"。发送信息至服务器时内容编码类型。
默认值适合大多数情况。如果你明确地传递了一个 content-type 给 $.ajax() 那么它必定会发送给服务器（即使没有数据要发送）。 |
| context           | Object   | 这个对象用于设置 Ajax 相关回调函数的上下文。也就是说，让回调函数内 this 指向这个对象（如果不设定这个参数，那么 this 就指向调用本次 AJAX 请求时传递的 options 参数）。比如指定一个 DOM 元素作为 context 参数，这样就设置了 success 回调函数的上下文为这个 DOM 元素。 |
| data              | String   | 发送到服务器的数据。将自动转换为请求字符串格式。GET 请求中将附加在 URL 后。查看 processData 选项说明以禁止此自动转换。必须为 Key/Value 格式。如果为数组，jQuery 将自动为不同值对应同一个名称。如 {foo:["bar1", "bar2"]} 转换为 '&foo=bar1&foo=bar2'。 |
| dataFilter        | Function | 给 Ajax 返回的原始数据的进行预处理的函数。提供 data 和 type 两个参数：data 是 Ajax 返回的原始数据，type 是调用 jQuery.ajax 时提供的 dataType 参数。函数返回的值将由 jQuery 进一步处理。 |
| dataType          | String   | 预期服务器返回的数据类型。如果不指定，jQuery 将自动根据 HTTP 包 MIME 信息来智能判断，比如 XML MIME 类型就被识别为 XML。在 1.4 中，JSON 就会生成一个 JavaScript 对象，而 script 则会执行这个脚本。随后服务器端返回的数据会根据这个值解析后，传递给回调函数。可用值:"xml"、"html"、"script"、"json"、"jsonp"、"text" |
| error             | Function | 默认值: 自动判断 (xml 或 html)。请求失败时调用此函数。
有以下三个参数：XMLHttpRequest 对象、错误信息、（可选）捕获的异常对象。如果发生了错误，错误信息（第二个参数）除了得到 null 之外，还可能是 "timeout", "error", "notmodified" 和 "parsererror"。
这是一个 Ajax 事件。 |
| global            | Boolean  | 是否触发全局 AJAX 事件。默认值: true。设置为 false 将不会触发全局 AJAX 事件，如 ajaxStart 或 ajaxStop 可用于控制不同的 Ajax 事件。 |
| ifModified        | Boolean  | 仅在服务器数据改变时获取新数据。默认值: false。使用 HTTP 包 Last-Modified 头信息判断。在 jQuery 1.4 中，它也会检查服务器指定的 'etag' 来确定数据没有被修改过。 |
| jsonp             | String   | 在一个 jsonp 请求中重写回调函数的名字。这个值用来替代在 "callback=?" 这种 GET 或 POST 请求中 URL 参数里的 "callback" 部分，比如 {jsonp:'onJsonPLoad'} 会导致将 "onJsonPLoad=?" 传给服务器。 |
| jsonpCallback     | String   | 为 jsonp 请求指定一个回调函数名。这个值将用来取代 jQuery 自动生成的随机函数名。这主要用来让 jQuery 生成度独特的函数名，这样管理请求更容易，也能方便地提供回调函数和错误处理。你也可以在想让浏览器缓存 GET 请求的时候，指定这个回调函数名。 |
| password          | String   | 用于响应 HTTP 访问认证请求的密码                             |
| processData       | Boolean  | 默认值: true。默认情况下，通过data选项传递进来的数据，如果是一个对象(技术上讲只要不是字符串)，都会处理转化成一个查询字符串，以配合默认内容类型 "application/x-www-form-urlencoded"。如果要发送 DOM 树信息或其它不希望转换的信息，请设置为 false。 |
| scriptCharset     | String   | 只有当请求时 dataType 为 "jsonp" 或 "script"，并且 type 是 "GET" 才会用于强制修改 charset。通常只在本地和远程的内容编码不同时使用。 |
| success           | Function | 请求成功后的回调函数。参数：由服务器返回，并根据 dataType 参数进行处理后的数据；描述状态的字符串。这是一个 Ajax 事件。 |
| traditional       | Boolean  | 如果你想要用传统的方式来序列化数据，那么就设置为 true。请参考工具分类下面的 jQuery.param 方法。 |
| timeout           | Number   | 设置请求超时时间（毫秒）。此设置将覆盖全局设置。             |
| type              | String   | 默认值: "GET")。请求方式 ("POST" 或 "GET")， 默认为 "GET"。注意：其它 HTTP 请求方法，如 PUT 和 DELETE 也可以使用，但仅部分浏览器支持。 |
| url               | String   | 默认值: 当前页地址。发送请求的地址。                         |
| username          | String   | 用于响应 HTTP 访问认证请求的用户名                           |
| xhr               | Function | 需要返回一个 XMLHttpRequest 对象。默认在 IE 下是 ActiveXObject 而其他情况下是 XMLHttpRequest 。用于重写或者提供一个增强的 XMLHttpRequest 对象。这个参数在 jQuery 1.3 以前不可用。 |

回调函数

```
如果要处理 $.ajax() 得到的数据，则需要使用回调函数：beforeSend、error、dataFilter、success、complete。

beforeSend
在发送请求之前调用，并且传入一个 XMLHttpRequest 作为参数。

error
在请求出错时调用。传入 XMLHttpRequest 对象，描述错误类型的字符串以及一个异常对象（如果有的话）

dataFilter
在请求成功之后调用。传入返回的数据以及 "dataType" 参数的值。并且必须返回新的数据（可能是处理过的）传递给 success 回调函数。

success
当请求之后调用。传入返回后的数据，以及包含成功代码的字符串。

complete
当请求完成之后调用这个函数，无论成功或失败。传入 XMLHttpRequest 对象，以及一个包含成功或错误代码的字符串。
```

### 使用

> 常规

```
$.ajax({
    url:'/js/data.json',
    type:'POST', 
    dataType: json,
    data:{name:'wang',age:25},
    async: true,
    success:function(data){
    	alert(data);
     },
	error:function(){
    	alert("出错")
	},
});
参数说明：
url: 请求地址
type: 请求方式，默认为GET，常用的还有POST
dataType: 预期服务器返回的数据类型。如果不指定，jQuery 将自动根据 HTTP 包 MIME 信息来智能判断，比如 XML MIME 类型就被识别为 XML。可为：json/xml/html/script/jsonp/text
data： 发送给服务器的参数
async: 同步或者异步，默认为true，表示异步
timeout: 设置请求超时时间（毫秒）,此设置将覆盖全局设置。
success： 请求成功之后的回调函数
error： 请求失败后的回调函数

新的写法(推荐)：
$.ajax({
    url: 'js/data.json',
    type: 'GET',
    dataType: 'json',
    data:{'aa':1}
})
.done(function(data) {
    alert(data.name);
})
.fail(function() {
    alert('服务器超时，请重试！');
});
// data.json里面的数据： {"name":"tom","age
```

> 上传文件

发送的data必须是FormData类型

把processData设为false，让jquery不要对formData做处理，如果processData不设置为false，jquery会把formData转换为字符串。

查看文件上传的请求头里`Content-Type: multipart/form-data; boundary=OCqxMF6-JxtxoMDHmoG5W5eY9MGRsTBp` ，参数boundary为请求参数之间的界限标识。 这里的Content-Type不是你设置的，而是FormData的content-type。

如果jquery请求设置了contentType，那么就会覆盖了formData的content-type,导致服务器在分隔参数和文件内容时是找不到boundary，报no multipart boundary was found错误

默认情况下jquery会把contentType设置为application/x-www-form-urlencoded。要jquery不设置contentType,则需要把contentType设置为false。

也就是说`contentType:false`,防止contentType覆盖掉formData的content-type。

```javascript
function upload() {
    //新建FormData对象
     var formData = new FormData(); 
    //取file控件中的文件,files属性取到的是一个fileList
    var fileList = $("#f1").files;  
    //将fileList中的文件逐个放入formData中，注意，直接放入fileList后台是取不到的
    formData.append('aaa', fileList[0]);  
    //formData.append()中的"key",如果传入的是文件,就可以随意取名字了
    formData.append('aaa', fileList[1]);  
    //作为示例,同时放入表单数据
    formData.append('bbb', $("#t1").val());     
    $.ajax({
    	url: "",   
        data: formData,
        type: 'POST',
        //这里着重强调contentType和processData都要设置为false
        //防止浏览器自动转换发送出的数据格式为字符串或其他
        contentType: false,                      
        processData: false,                      
        success: function (data) {               
        	if (data === "") {
            	return false;
                } 
        },
        error: function (a, b, c) {
            alert("aaa");
         }
     });
} 
```

封装方法：

- load

```javascript
# 从服务器加载数据，并把返回的数据放入被选元素中
$(selector).load(URL[,data][,callback])
# 参数
URl:string,请求HTML页面的URL地址
data:Objec,发送至服务器的key/value数据,无参是get，有参转为post
callback:Function,请求完成时的回调函数，无论请求成功或失败

# eg
$("button").click(function(){
  $("#div1").load("demo_test.txt",function(responseTxt,statusTxt,xhr){
    if(statusTxt=="success")
      alert("外部内容加载成功！");
    if(statusTxt=="error")
      alert("Error: "+xhr.status+": "+xhr.statusText);
  });
});
```

- get

```javascript
# 通过 HTTP GET 请求从服务器上请求数据。
$.get(URL[,data][,callback][,type])
# 参数
URL:string,请求的HTML页面的URL地址
data:object,发送至服务器的key/value数据会作为QueryString附加到请求URL中
callback:Function,载入成功时回调函数(只有当Response的返回状态是success才调用该方法)自动将请求结果和状态传递给该方法
type:string,服务器端返回内容的样式，包括xml、html、script、json、text和_default

# eg
$("button").click(function(){
  $.get("demo_test.asp",function(data,status){
    alert("Data: " + data + "\nStatus: " + status);
  });
});
```

- post

```javascript
# 通过 HTTP POST 请求从服务器上请求数据
$.post(URL,data,callback)
# eg
$("button").click(function(){
  $.post("demo_test_post.asp",
  {
    name:"Donald Duck",
    city:"Duckburg"
  },
  function(data,status){
    alert("Data: " + data + "\nStatus: " + status);
  });
});
```

## jsonp

ajax只能请求同一个域下的数据或资源，有时候需要跨域请求数据，就需要用到jsonp技术，jsonp可以跨域请求数据，它的原理主要是利用了`<script>`标签可以跨域链接资源的特性。jsonp和ajax原理完全不一样，不过jquery将它们封装成同一个函数。

```javascript
$.ajax({
    url:'js/data.js',
    type:'get', // 只能是GET
    dataType:'jsonp',
    jsonpCallback:'fnBack'
})
.done(function(data){
    alert(data.name);
})
.fail(function() {
    alert('服务器超时，请重试！');
});

// data.js里面的数据： fnBack({"name":"tom","age":18});
```

eg：获取360搜索关键词联想数据

```javascript
$(function(){
    $('#txt01').keyup(function(){
        var sVal = $(this).val();
        $.ajax({
            url:'https://sug.so.360.cn/suggest?',
            type:'get',
            dataType:'jsonp',
            data: {word: sVal}
        })
        .done(function(data){
            var aData = data.s;
            $('.list').empty();
            for(var i=0;i<aData.length;i++)
            {
                var $li = $('<li>'+ aData[i] +'</li>');
                $li.appendTo($('.list'));
            }
        })        
    })
})

//......

<input type="text" name="" id="txt01">
<ul class="list"></ul>
```







