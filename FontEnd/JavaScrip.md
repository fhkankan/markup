[TOC]

# JavaScript


```
页面行为：部分动画效果、页面与用户的交互、页面功能
```

## 嵌入页面的方式

```javascript
# 行间事件
<input type="button" name="" onclick="alert('ok！');">
# 页面script标签嵌入
<script type="text/javascript"> alert('ok！');</script>
# 外部引入
<script type="text/javascript" src="js/index.js"></script>
```

## 加载执行

```javascript
注意：若把javascript写在元素的上面，就会出错，因为页面上从上往下加载执行的，javascript去页面上获取元素div1的时候，元素div1还没有加载，解决方法有两种：
第一种方法：将javascript放到页面最下边；
第二种方法：将javascript语句放到window.onload触发的函数里面,获取元素的语句会在页面加载完后才执行。
window.onload = function(){
        var oDiv = document.getElementById('div1');
    }
```



## 调试程序的方法

```
1、alert(变量名)			弹窗显示，会暂停程序运行
2、console.log(变量名)		浏览器控制台显示，不中断
3、document.title=变量名	页面标题显示，不中断
4、断点调试				   在浏览器的sources中设置断点
```

## 命名规范

```
变量、函数、属性、函数参数命名规范
1、区分大小写
2、第一个字符必须是字母、下划线（_）或者美元符号（$）
3、其他字符可以是字母、下划线、美元符或数字

匈牙利命名风格：
对象o Object 比如：oDiv
数组a Array 比如：aItems
字符串s String 比如：sUserName
整数i Integer 比如：iItemCount
布尔值b Boolean 比如：bIsComplete
浮点数f Float 比如：fPrice
函数fn Function 比如：fnHandler
正则表达式re RegExp 比如：reEmailCheck
```

## 操作元素属性

```javascript
操作元素属性 
读取属性		var 变量 = 元素.属性名 
改写属性		元素.属性名 = 新属性值 
注意：用js读取元素的属性必须是在行间已有赋值的，否则为空


属性名在js中的写法 
1、html的属性和js里面属性写法一样
2、“class” 属性写成 “className”
3、“style” 属性里面的属性，有横杠的改成驼峰式，比如：“font-size”，改成”style.fontSize”

innerHTML 
innerHTML可以读取或者写入标签包裹的内容
<script type="text/javascript">
    window.onload = function(){
        var oDiv = document.getElementById('div1');
        //读取
        var sTxt = oDiv.innerHTML;
        alert(sTxt);
        //写入
        oDiv.innerHTML = '<a href="http://www.itcast.cn">传智播客<a/>';
    }
</script>
......
<div id="div1">这是一个div元素</div>
```

## 基本概念

- 变量

```
JavaScript 是一种弱类型语言，javascript的变量类型由它的值来决定。 定义变量需要用关键字 'var'
var iNum = 123;
//同时定义多个变量可以用","隔开，公用一个‘var’关键字
var iNum = 45,sTr='qwe',sCount='68';

5种基本数据类型：
1、number 数字类型
2、string 字符串类型
3、boolean 布尔类型 true 或 false
4、undefined undefined类型，变量声明未初始化，它的值就是undefined
5、null null类型，表示空对象，如果定义的变量将来准备保存对象，可以将变量初始化为null,在页面上获取不到对象，返回的值就是null

1种复合类型：
object

所有变量都拥有可以读取和更新的特性
value		属性值，其为默认属性
writable	若属性被更新， 将其设置为true
enumerable	在枚举对象成员时，若该属性应被包括在内，设置为true
configurable若属性可以被删除，且该特性可以被修改，设置为true
```

- 对象

```
# 创建
方法一：对象字面量
var myObject = {
    color: "red",
    count: 5,
    log: function(){
        alert("hi!")
    }
}
方法二：对象构造函数
var redObject = new Object();
redObject.color = "red";
redObject.count = 5;
redObject.log = function(){
    alert("red!")
}

# 使用
myObject.color
redObject.count
```

- 构造函数

```
# 自定义构造函数
function Item(color, count){
    this.color = color;
    this.count = count;
    this.log = function(){
        alert("hello!")
    }
}

注意：不能定义多个同名的构造函数。若定义了，则后面的函数会取代前面的函数。
```

- 原型

所有对象都有一个原型。原型是实际对象被实例化的模型或蓝图

```
# 对redObject对象增加一个属性
redObject.isAvailable = true;

# 对所有使用Item构造函数的实例都增加属性,将属性添加到Item原型中
Item.prototype.isAvailable = true;
```

## 继承

- 使用原型

`javaScript`通过一种被称为原型继承的方法提供对继承的支持。这意味着一个原型可以拥有`prototype`属性，也可以拥有一个原型。称为原型链

创建一个继承自Item的新对象`SpecialItem`：

1. 创建``SpecialItem()``构造函数

```
function SpecialItem(name){
    this.name = name;
    this.deacribe = function(){
        console.log(this.name + ": color=" + this.color);
    }
}
```

2. 为构建继承关系，设置prototype属性

```
SpecialItem.prototype = new Item();
```

3. 指定其他属性

```
function SpecialItem(name, color, count){
	Item.call(this, color, count);
    this.name = name;
    this.deacribe = function(){
        console.log(this.name + ": color=" + this.color);
    }
}
```

4. 创建对象

```
var special = new SpecialItem("Widget", "Purple", 4);
special.log();
special.describe();
special.log(special);
```

- 使用Create

更改2构建继承关系

```
SpecialItem.prototype = Object.create(Item.prototype);
SpecialItem.prototype.constructor = SpecialIem;
```

- 使用类关键字

```
class Item{
    constructor(color, count){
        this.color = color;
        this.count = count;
        this.log = function(){
            console.log(this.name + ": color=" + this.color);
        };
    }
}

class SpecialItem extends Item{
    constructor(name, color, count){
        super(color, count);
        this.name = name;
        this.describe = function(){
            console.log(this.name + ": color=" + this.color);
        }
    }
}
```

为了保证两种方法解决方案的一致性，增加

```
Item.prototype.isAvailable = true;
Item.prototype.add = function(n){this.count += n;};
```

## 语句与注释

```
javascript语句开始可缩进也可不缩进，缩进是为了方便代码阅读，一条javascript语句应该以“;”结尾;

// 单行注释
/*  
    多行注释
    1、...
    2、...
*/
```

## 函数

```javascript
# 函数定义与执行
<script type="text/javascript">
    // 函数定义
    function fnAlert(){
        alert('hello!');
    }
    // 函数执行
    fnAlert();
</script>

# 变量与函数预解析 
JavaScript解析过程分为两个阶段，先是编译阶段，然后执行阶段，在编译阶段会将function定义的函数提前，并且将var定义的变量声明提前，将它赋值为undefined。

<script type="text/javascript">    
    fnAlert();       // 弹出 hello！
    alert(iNum);  // 弹出 undefined
    function fnAlert(){
        alert('hello!');
    }
    var iNum = 123;
</script>

# 函数传参
<script type="text/javascript">
    function fnAlert(a){
        alert(a);
    }
    fnAlert(12345);
</script>

# 函数'return'关键字 
1、返回函数中的值或者对象
2、结束函数的运行
```

## 事件属性和匿名函数

```
事件属性 
元素上除了有样式，id等属性外，还有事件属性，常用的事件属性有鼠标点击事件属性(onclick)，鼠标移入事件属性(mouseover),鼠标移出事件属性(mouseout),将函数名称赋值给元素事件属性，可以将事件和函数关联起来。

匿名函数 
定义的函数可以不给名称，这个叫做匿名函数，可以将匿名函数的定义直接赋值给元素的事件属性来完成事件和函数的关联，这样可以减少函数命名，并且简化代码。函数如果做公共函数，就可以写成匿名函数的形式。

<script type="text/javascript">
window.onload = function(){
    var oBtn = document.getElementById('btn1');
    /*
    oBtn.onclick = myalert;
    function myalert(){
        alert('ok!');
    }
    */
    // 直接将匿名函数赋值给绑定的事件
    oBtn.onclick = function (){
        alert('ok!');
    }
}
</script>
```

## 封闭函数

```javascript
封闭函数是javascript中匿名函数的另外一种写法，
具有如下特性：
1.脚本一旦解析，函数就开始就执行
2.函数不用命名。

封闭函数的作用 
封闭函数可以创造一个独立的空间，在封闭函数内定义的变量和函数不会影响外部同名的函数和变量，可以避免命名冲突，在页面上引入多个js文件时，用这种方式添加js文件比较安全

# 创建封闭函数：
# 方法一：
(function(){
    alert('hello!');
})();
# 方法二
;!function(){
    alert('hello!');
}()
# 方法三
;~function(){
    alert('hello!');
}()
```

## 条件语句

```
# 条件运算符 
==、===、>、>=、<、<=、!=、&&(而且)、||(或者)、!(否)

if(){}
else{}

多重if else语句
if(){}
else if(){}
else if(){}
else{}
```

## 循环语句

```
# while循环
while(条件){循环体}

# for 循环
for（var i=0;i<len;i++）{循环体}
```

## 上下文

关键字this指的是单个对象。

在一个函数中，this指向调用该函数的对象；在一个带有属性和方法的典型对象中，若函数使用了this，则this指向包含该函数的对象

```javascript
var myObject = {
    color: "Red",
    count: 5,
    log: fucntion(){
        console.log("Quantity:" + this.count + ", Color:" + this.color);
    }
};
```

call()函数允许调用一个函数并指定this值

```javascript
fucntion Vehicle(weight, cost){
    this.weight = weight;
    this.cost = cost;
}

function Truck(weight, cost, axles, length){
    Vehicle.call(this, weight, cost)
    this.axles = axles;
    this.length = length;
}

var tonka = new Truck(5, 25, 3, 15);
console.log(tonka);
```

## 异常

```
try{
    var x = 5;
    var y = 0;
    if (y == 0){
        throw("Can't divide by zero")
    }
    console.log(x/y);
}
catch(e){
    console.log("Error:" + e);
}
finally{
    console.log("Finally block executed");
}
```

## 变量作用域

```
变量作用域指的是变量的作用范围，javascript中的变量分为全局变量和局部变量。

1、全局变量：在函数之外定义的变量，为整个页面公用，函数内部外部都可以访问。
2、局部变量：在函数内部定义的变量，只能在定义该变量的函数内部访问，外部无法访问。

注意：在函数内部若使用var新建与全局变量同名的变量，则在函数内部调用时，优先调用内部变量，不对外部变量产生影响
```

## 命名空间

```
在JavaScript中，所有全局作用域的对象都需要一个唯一名称，以避免名称冲突。注意：运行在浏览器中的所有脚本都共享全局作用域。
命名空间是一种以层次结构的方式组织代码的技术。一个对象的完全限定名称包括其在层次结构中定义的路径
早JavaScript中以嵌套的对象实现命名空间的概念
```

eg

```
// global object
var mySample = {};
// Define the namespace hierarchy
mySample.things = {};
mySample.things.helpers = {};
mySample.otherThings = {};
// Add stuff to the namespace
mySample.things.count = 0;
mySample.things.helpers.logger = function(msg){
    console.log(msg);
}

// 若代码被分割为多个文件，或需要在多个地方使用该代码，就需要确保对象不会被创建两次
// 若未定义，创建对象返回False，or语句可以执行语句的后面部分，创建对象；若已被创建，则返回True，不执行后面部分
this.mySample = this.mySample || {};
```

## 数组及操作方法

```
数组里面的数据可以是不同类型的

数组是一个属性集合，本身也有一个名称，可以作为对象或变量的一个成员，可以通过一个数字索引访问数组中的条目
创建
```

- 创建数组

```
方法一：字面量
//对象的实例创建
var aList = new Array(1,2,3);
//直接创建
var aList2 = [1,2,3,'asd'];
//多维数组 
var aList = [[1,2,3],['a','b','c']];

方法二： 空+push
var colors= []
colors.push("red")
colors.push("green")
```

- 操作数组

```
1、获取数组的长度：aList.length;
2、用下标操作数组的某个数据：aList[0];
3、将数组成员通过一个分隔符合并成字符串:aList.join('-')
4、从数组最后增加成员或删除成员：aList.push(5);aList.pop();
5、将数组反转：aList.reverse();
6、返回数组中元素第一次出现的索引值，若没有返回-1：aList.indexOf(值)
7、在数组中增加或删除成员：aList.splice(2,1,7,8,9); //从02索引处，删除1个元素，然后在此位置增加'7,8,9'三个元素

批量操作数组中的数据，需要用到循环语句
遍历
for(var i=0; i<colors.length; i++){
    console.log(colors[i]);
}
```

## 字符串处理方法

```
1、合并操作：“ + ”
2、将数字字符串转化为整数：parseInt(字符串) 
3、将数字字符串转化为小数：parseFloat(字符串)
4、把一个长字符串分隔成小字符串组成的数组：字符串.split('分割符')
5、查找字符串是否含有某字符：字符串.indexOf('目的字符') 
6、截取字符串：字符串.substring(start,end)（不包括end）
7、字符串反转：字符串.split('').reverse().join('')
```

## 浏览器对象模型

### windows对象

- 控制台

console位于全局命名空间中

> 字符串替换

s	字符串

d/I	数字

f	浮点

o	对象

```
console.info("Formating: %i, %f, %s, %o", 22.651, Math.PI, "text", screen);
```

> 分析

```
console.time("Stopwatch");
var j = 1;
for(var i = 0; i < 1000; i++){
    j += j + i;
}
console.info("j = %i", j);
console.timeEnd("Stopwatch");
```

- 缓存

> cookie

```
function storeCookie(key, values, duration){
    var expDate = new Date();
    expDate.setTime(expDate.getTime() + duration * 8640000);
    document.cookie = key + "=" + value + ";expires=" + expDate.toUTCString();
}

function removeCookie(key){
    storeCookie(key, "", 0);
}

fucntion getCookie(key){
    var cookies = document.cookie.split(";");
    for(var i=0; i<cookies.length; i++){
        if(cookies[i].trim().indexOf(key + "=") ==0){
            return cookies[i].trim().substring(key.length = 1);
        }
    }
    return null;
}
```

> 储存

`HTML5`两种新的存储机制：``localStorage`和`sessionStorage`

都是window对象的属性，都返回Storage对象。浏览器关闭时`sessionSorage`就会被清除，`localStorage`则可以无限期保存

```
var cache = window.sessionStorage;
//删除所有项目
cache.clear();
//添加项目
cache.setItem("key1", "This is my saved data");
console.log("Saved data:" + cache.getItem("key1"));
//删除项目
cache.removeItem("key1");
console.log("Saved data:" + cache.getItem("key1"));
```

- 界面元素

```
locationbar		显示当前文档的URL
menubar			浏览器所提供的任何类型的菜单
personalbar		提供了用户首选项、书签、收藏夹
scrollbar		垂直或水平滚动条
statusbar		提供页面上当前所选元素的当前页面的状态
toolbar			包含了基于用户界面命令的组件，如后退按钮或刷新按钮
```

- 计时器

```
定时器在javascript中的作用
1、定时调用函数
2、制作动画

定时器
setTimeout(函数名,时间)  	   创建只执行一次的定时器 
clearTimeout(定时器名) 			关闭只执行一次的定时器
setInterval(函数名,时间)  	   创建反复执行的定时器
clearInterval(定时器名) 		关闭反复执行的定时器
```

### window子对象

- screen对象

提供了有关浏览器正在运行的设备的详细信息，`window.screen`

```
height			设备的总高度
width			设备的总宽度
availHeight		操作系统不使用垂直空间量
availWidth		可用的水平空间
colorDepth		用于指定颜色的位数
pixelDepth		与colorDepth相同
```

- location对象

提供了窗口所加载文档的Web地址，它还链接到document对象，可通过window.location、document.location或location

```
protocol		
hostname
port
pathname
search
hash
href		提供了完整的URL
host		包含了hostname和port
origin		包含了protocol、hostname和port
```

- history对象

跟踪在单个窗口中加载的页面，用于支持后退按钮，允许返回到前一个页面

```
legth		指示历史记录中有多少项
back()		导航到前一个页面
forward()	导航到下一个页面
go()		跳转到指定页面


pushState()	添加另一个历史元素
replaceState()使用所提供的数据更新现有历史记录
此两种方法支持如下参数:
state		一个JSON字符串，包含了任何想要存储的可序列化的数据
title		该参数可以忽略；可以输入一个空字符串或描述页面的简短标题
url			历史记录汇总的所存储的地址，用来向前或向后导航
```

- navigator对象

与screen对象类似，提供了有关浏览器正在运行的设备的信息。支持一个很长的属性和方法列表

> 用户代理

包含多个用来指明操作系统和浏览器详细信息的属性

```
appCodeName
appName
appVersion
platform
product
userAgent
```

> 电池

电池属性返回一个BatteryManger对象，它提供了以下属性：

```
charging		设备正在充电，则为true
charingTime		完全充满之前剩余的秒数
dischargingTime	电池完全放完电所需要的秒数
level			0~1之间的值，表示当前充电程度
```

## 窗口对象

### 创建窗口

```
// open函数参数1:页面路径，参数2：窗口名称，返回新窗口的window对象
// 若使用相同的窗口名称，则在原来的窗口中使用新的URL替换内容
window.name = "chapter1";
var people = window.open("PopUp.html", "popup");
```

- 配置参数

open函数接收第三个参数，此为窗口特性字符串

```
window.open("PopUp.html", "popup", "height=300,width=400,top=400,left=150, status");

//位置和大小
left, top, height, width, outerHeight, outerWidth
//chrome(用户界面UI)特性(Boolean值)
location, menubar, personalbar, status, titlebar, toolbar
// 窗口特性
alwaysLowered 在Z轴方向上将新窗口放置到现有窗口之下
alwaysRaised 在Z轴方向将新窗口放置到现有窗口之上
close 设置为no，禁用关闭图标
dependent 当父窗口关闭时从属窗口自动关闭
minimizable 禁用最小化图标
fullscreen 以全屏模式显示新窗口
resizable 使窗口能够缩放，默认值
scrollbars 设置为no，禁用滚动条，默认情况下，当内容不适合所分配的空间时包含滚动条

```

- 操作窗口

创建一个窗口后，可以使用window属性和方法来查看并调整其大小、位置和滚动属性

```
// 属性
innerHeight	可用于内容的空间高度
innerWidth	可用于内容的空间宽度
outerHeight	窗口总高度
outerWidth	窗口总宽度
screenX		设备左边缘与窗口左边缘之间的距离
screenY		设备上边缘与窗口上边缘之间的距离
scrollX		文档已经水平滚动的像素数
scrollY		文档已经垂直滚动的像素数

// 方法
moveBy()	将窗口移动指定的像素数，可水平或垂直移动，若某方向无移动值为0，若为负值，则向上或向右移动
moveto()	移动窗口，使其左上角位于指定的位置
resizeBy()	按指定的数量增加窗口大小，可水平或垂直增加，负值为收缩大小。上左边缘保持不变
resizeTo()	指定新窗口大小。上左边缘保持不变
scrollBy()	滚动指定的像素；可指定水平或垂直的滚动值
scrollByLines()	垂直滚动文档指定的行数
scrollByPages()	垂直滚动文档指定的页数
scrollTo()		滚动到指定的水平和垂直位置
sizeToContent()	更改窗口大小以使用现有的内容
focus()		使窗口保持聚焦
blur()		从窗口删除焦点
```

### 模态对话框窗口

之前的窗口都是非模态的(modeless),用户可以与其他窗口交互

模态(modal)窗口获取焦点并禁用应用程序的其他部分，直到关闭窗口为止

- 标准的弹出对话框

```
// 警告框
window.alter("alert box");
// 确认框, 提示框
if (windos.confirm("Is it OK to proceed?")){
    var answer = window.prompt("How many pets do you have?", 0);
    console.log("%i pets were entered.", answer);
}
else{
    console.log("Confirmation failed");
}
```

- 自定义模态对话框

```javascript
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>
<body>
    <p>Hello World</p>
    <button onclick="showDialog()">Show Dialog</button>
    <div id="glass" class="glass">
        <div class="dialog">
            <h3>Prompt</h3>
            <p>How many pets do you have</p>
            <input type="number" value="0" id="numPets" />
            <button id="dialogOK" type="Submit" onclick="OK()">OK</button>
        </div>
    </div>
    <style type="text/css">
        .glass{
            position: fixed;
            left: 0;
            top: 0;
            background-color: rgba(225,225,225,.7);
            height: 100vh;
            width: 100vw;
            z-index: 100;
        }
        .dialog{
            height: 125px;
            width: 220px;
            margin: 0 auto;
            padding: 15px;
            border: 1px solid black;
            background-color: white;
        }
    </style>
    <script type="text/javascript">
        var result = 0;
        function showDialog(){
            var dialog = document.getElementById("glass");
            dialog.style.visibility = "visible";
        }
        function closeDialog(){
            var dialog = document.getElementById("glass");
            dialog.style.visibility = "hidden"; 
        }
        function OK(){
            var input = document.getElementById("numPets");
            result = input.value;
            closeDialog();
            console.log("#Pets:" + result);
        }
        closeDialog()
    </script>
</body>
</html>
```

### 框架

可以包含在一个内联框架(iframe)元素来嵌入来自其他HTML文档的内容。

内联框架是通过使用iframe元素嵌入的，同时将该元素的src特性设置为包括在框架内的文档的位置

```
<iframe src="http://www.baidu.com"></iframe>
<style type="text/css">
    iframe{
        width: 95vw;
        height: 300px;
        border: 3px solid blue;
        margin-top: 5px;
        }
 </style>
```

- 内联框架元素支持多个可以在标记中设置的特性

```
allowfullscreen	如何想要嵌入的窗口可以切换到全屏模式，可以使用该特性
height		元素的高度
name		框架名称，可用来创建指向该元素的链接
sandbox		用来指定对窗口的限制
src			框架中所加载的文档的URL
srcdoc		与sandbox特性一起使用
width		元素的宽度
```

- 访问框架

```
window.frames.length	框架的数量
window.frames[0]		访问第一个框架
window.parent			访问外部窗口
```

- 使用sandbox

嵌入他人页面进入自己网站，为了降低风险，可指定sandbox模式

```
<iframe src="http://www.baidu.com" sandbox=""></iframe>
```

允许的功能可通过指定特性值

```
allow-forms
allow-modals
allow-orientation-lock
allow-pointer-lock
allow-popups
allow-popups-to-escape-sandbox
allow-presentation
allow-same-origin
allow-scripts
allow-top-navigation
```

## DOM元素

```javascript
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>
<body>
</body>
</html>

<script type="text/javascript">
// 用JavaScript创建HTML文件中的body元素
function populateBody(){
	//获取body元素
    var body = document.getElementsByTagName("body")[0];
    //创建段落元素
    var paragraph = document.createElement("p");
    //更新innerHTML属性
    paragraph.innerHTML = "<strong>Hello</strong>Word!";
    //将创建的段落插入到body元素中
    body.appendChild(paragraph);
}
populateBody();
</script>
```

### 查找元素

```javascript
// 通过id定位元素，返回零个或一个元素，若无，返回null
var element = document.getElementById(id);
// 返回与元素类型相匹配的元素数组,若无，返回[]
var elementArray = document.getElementsByTagName(name);
// 返回与class类型相匹配的元素数组,若无，返回[]
var elementArray = document.getElementsByClassName(names);
//通过css定位
var elementArray = document.querySelector(name)
var elementArray = document.querySelectorAll(names)
```

### 创建元素

```javascript
// 创建元素节点
var p = document.createElement("p")

// 方法一
// 创建文本节点
var text = document.createTextNode("This is a test");
// 插入节点
p.appendChild(text)

// 方法二
// 采用textContent
p.textContent = "This ia a test"

// 方法三
// 使用innerHTML，可创建子节点
p.innerHTML = "This ia a test";
```

### 移动元素

```javascript
// appendChild()添加子节点至父节点(若已有子节点，则位于同级节点尾部)
var child = parentNode.appendChild(childNode);
// insertBefore()添加子节点位于某个同级子节点前方,
var child = parentNode.isertBefore(childNode, sibling);
// removeChild()删除父节点中的子节点
var removedElement = parentNode.removeChild(childNode);
// 删除某一元素
element.parentNode.removeChild(element)
// replaceChild()用新元素替换已有的元素
var removedElement = parentNode.replaceChild(newElement, existingElement);
```

### 修改元素

```javascript
// 为了操作创建元素中使用的特性，每个元素都有attributes属性，其中包含了元素上锁定义的所有的特性。这是键/值对集合
hasAttribule()	// 确定某个特定的特性是否被指定
// 添加或修改现有特性
var link = documnet.getElementByTagName("link")[0];
link.id = "myID";
link.className = "myClass";
link.lang = "en";
var attr = link.attributes;
for (var i=0; i< attr.lenght; i++){
    console.log(attr[i].name + "=" + attr[i].value + "'");
}
```

### 相关元素

在文档中获取了一个元素之后，可以使用如下属性导航到相关联元素

```
parentNode		节点的直接父节点，若是根节点则为空
children		HTMLCollection对象，包含了该节点所有的直接子节点
firstChild		子节点集合中的第一个元素
lastChild		子节点集合中的最后一个元素
previousSibling	与当前节点有相同父节点，且在子节点集合汇总位于当前节点之前，若是父节点的第一子节点，则属性为空
nextSibling		与当前节点有相同父节点，且在子节点集合汇总位于当前节点之后，若是父节点的最后一子节点，则属性为空
```















