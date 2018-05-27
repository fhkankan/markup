[TOC]

# JavaScript进阶

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

## 动态样式设计

### 更改样式表

- 启动样式表

通过documnet.styleSheets属性访问可用的样式表，返回已加载的样式表集合

HTML

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link rel="stylesheet" href="Sample.css" title="Shared">
    <script src="SampleScript.js" defer></script>
</head>
<body>
   <p><strong>Hello</strong></p> 
   <button onclick="toggleSS()">Toggle</button>
</body>
</html>
```

Sample.css

```css
p {
    font-size: xx-large;
}
```

SampleScript.js

```javascript
"use strict";
function toggleSS(){
    for (var i = 0; i < document.styleSheets.length; i++){
        if (document.styleSheets[i].title == "Shared") {
            document.styleSheets[i].disabled = !document.styleSheets[i].disabled;
            break;
        }
    }
}
```

- 选择样式表

css

```css
// Red.css
p {
    color: red;
}
// Green.css
p {
    color: green;
}
// Blue.css
p {
    color: blue;
}
```

html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link rel="stylesheet" href="Sample.css" title="Shared">
    <link rel="stylesheet" href="Red.css">
    <link rel="stylesheet" href="Green.css">
    <link rel="stylesheet" href="Blue.css">
    <script src="SamplesScript.js" defer></script>
</head>
<body>
   <p><strong>Hello</strong></p> 
   <button onclick="toggleSS()">Toggle</button>
   <button onclick="disableAll()">Black</button>
   <button onclick="enable('Red')" style="color: red">Red</button>
   <button onclick="enable('Green')" style="color: green">Green</button>
   <button onclick="enable('Blue')" style="color: blue">Blue</button>
</body>
</html>
```

js

```javascript
function disableAll(){
    for (var i = 0; i < document.styleSheets.length; i++ ){
        if (document.styleSheets[i].title != "Shared"){
            document.styleSheets[i].disabled = true;
        }
    }
}
function enable(color){
    for (var i = 0; i < document.styleSheets.length; i++ ){
        if (document.styleSheets[i].href.includes(color)){
            document.styleSheets[i].disabled = false;
        }
        else if (document.styleSheets[i].title != "Shared"){
            document.styleSheets[i].disabled = true;
        }
    }
}
```

### 修改规则

每个样式表都有一个枚举了样式表中所有包含规则的cssRules属性。该属性是CSSStyleRule对象集合。

```javascript
var newRuleIndex
function toggleRule(){
    if (newRuleIndex == -1){
        newRuleIndex = document.styleSheets[0].insertRule("p {border: 1px solid black;}", 1)
    }
    else {
        document.styleSheets[0].deleteRule(newRuleIndex);
        newRuleIndex = -1;
    }
}
```

### 修改类

上面修改样式表中规则的方法可能会影响文档中的多个元素，然而，多数情况下，只需将现有的规则应用于某些情况的特定元素即可。:enabled、:selected、:hover等伪类允许在某些动态条件下应用样式规则，但是条件有限

可以创建一个使用类选择器的样式规则，然后使用JavaScript动态地应用合适的类。

Sample.css

```css
.special {
    opacity: .5;
}
```

js

```javascript
function toggleClass(){
    var paragraph = document.querySelector("p");
    if (paragraph){
    	// contians()返回一个布尔值，指明类是否存在
        if (paragraph.classList.contains("special")){
        	// remove()删除指定的类
            paragraph.classList.remove("special");
        }
        else {
        	// add()添加一个新类
            paragraph.classList.add("special");
        }
        // 也可以被下面的代码执行
        //paragraph.classList.toggle("special");
    }
}
```

html

```html
<button onclick="toggleClass()">Opacity</button>
```

### 修改内联样式

修改规则仅仅影响了单个元素

添加/删除一个类可以应用或删除多条样式规则，同时在每条规则上可以定义多个声明

修改内联元素适用于单个元素，同时使外部的CSS不能更改其样式

- 使用CSSStyleDeclaration

所有的HTML元素上都可以使用style特性，而在javascript中通过style属性可以访问该特性，并返回CSSStyleDeclaration对象(键/值对集合)

```
# 通过下面的方法修改此对象中的样式声明
setProperty()	添加一个声明，接收两个必须的参数(属性/值)以及一个可选的优先级参数(空白/important)
getpropertyValue()	作为参数传入属性名并返回相应的值
getPropertyPriority()	如果指定的属性有关键字important，则返回important
cssText		返回该样式中的所有声明，并格式化为CSS文件中的格式
removeProperty()	删除指定的属性
```

- 设置样式属性

针对每个CSS属性，CSSStyleDeclaration对象都有一个属性与之对应。

```
// 获取或设置属性
var p = document.querySelector("p");
p.style.fontStyle = "italic";
p.style.fontSize = "xx-small"
console.log("Current color is " + p.style.color)
```

- 使用setAttribute

通过setAttribute()方法设置任何HTML特性值，它接收两个参数：特性名称、想要设置的值

```
var p = document.querySelector("p");
p.setAttribute("style", "font-style:italic; font-size:xx-small;");
```

## 事件

事件是一些可以通过脚本响应的页面动作。事件处理是一段JavaScript代码，总是与页面中的特定部分以及一定的时间相关联。当页面特定部分关联的时间发生时，事件处理器就会被调用。

### 事件注册

```
// 方法一:内联注册,HTML中
<div id="div2" onclick="someAction()">

// 方法二：将事件处理程序分配给DOM元素的事件属性，JavaScript中
var div = document.getElementById("div2");
div.onclick = someAciton;

// 方法三：早期版本用attachEvent()
div.addEventListener("click", someAction);
```

方法一：违反了关注点分离，每个元素的每个事件只能注册单个事件处理程序

方法二：每个元素的每个事件只能注册单个事件处理程序，若注册第二个，则覆盖

方法三：允许为同一事件分配多个事件处理程序

### 事件传播

向下传播事件被称为捕获阶段，向上传播事件被称为冒泡阶段

addEventListener()方法支持传入第三个参数，可用来指定是否要监听捕获事件或冒泡事，若为True,则监听捕获事件；False/忽略,则监听冒泡事件

其他注册方法，只能用来注册冒泡事件

若只是想知道某个按钮是否被单击，则是否监听捕获或冒泡事件并不重要。然而若有多个事件处理程序分配给了不同的元素，箭筒这两个事件就非常有用

事件处理程序可以连续执行；在调用下一个处理程序之前，当前处理程序必须完成。如果在同一个元素上分配了多个事件处理程序，则按照事件注册顺序连续执行

### 删除注册事件

对于addEventListener()/attachEvent()，由于可将多个事件处理程序分配给单个元素，为了正确删除，需要制定注册时间处理程序中所使用的所有相同信息(注册事件的元素，事件类型，注册的处理程序函数，表示是否在捕获阶段或冒泡阶段注册的标志)

```
function removeHandlers(){
    var div = document.getElementById("div2");
    div.removeEventListener("click", someAction, false)
}
```

若attachEvent()，则使用detachEvent()方法删除，类似removeEventListener，无捕获/冒泡标志

若其他注册方法，`div.onclick = null;`

### 事件接口

- 常用事件属性

通过声明一个函数参数，事件处理程序就可以访问事件对象。虽参数任意取，常用e

```
fucntion someAction(e){
	// 事件类型
    console.log(e.type);
    // 触发事件的元素
    console.log(e.target);
    // 注册事件处理程序的元素
    console.log(e.currentTarget);
}
```

- 取消事件

修改事件的处理方式，可在事件对象上调用如下方法

```
stopPropagation()	//停止事件传播
stopImmediatePropagation()	// 停止传播，阻止当前元素上的任何其他处理程序执行
preventDefault()	//禁用默认动作
```

### 事件对象

在IE中，事件对象是window对象的一个属性event，且event对象只能在事件发生时被访问，所有事件处理完，该对象就消失了。而标准DOM中规定event必须作为唯一的参数传给时间处理函数，为兼容，常采用以下方法

```
function someHandle(event){
    if(window.event){
        event = window.event
    }
}
```

在IE中，事件的对象包含在event的srcElement属性中，而在标准的DOM浏览器中，对象包含在target属性中。为处理兼容性，采用如下方法

```
function handle(oEvent){
    if(window.event)
        oEvent = window.event;	//处理兼容性，获得事件对象
    var oTarget;
    if(oEvent.srcElement)		//处理兼容性，获得事件目标
        oTarget = oEvent.target;
    else
    	oTarget = oEvent.target;
    alert(oTarget.tagName)		//弹出目标的标记名称
}
window.onload = function(){
    var oImg = documnet.getElementByTagName("img")[0];
    oImg.onclick = handle;
}
```



