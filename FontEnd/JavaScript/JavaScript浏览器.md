

# JavaScript3

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

```javascript
// cookie存储
function storeCookie(key, values, duration){
    var expDate = new Date();
    expDate.setTime(expDate.getTime() + duration * 8640000);
    document.cookie = key + "=" + value + ";expires=" + expDate.toUTCString();
}

// cookie删除
function removeCookie(key){
    storeCookie(key, "", 0);
}

// cookie获取
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

`HTML5`两种新的存储机制：`localStorage`和`sessionStorage`

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

## Window对象

Window对象代表的是打开的浏览器窗口，通过window对象可以打开与关闭窗口、控制窗口的大小和位置，由窗口弹出对话框，，控制窗口上是佛硻地址栏、工具栏和状态栏等栏目。对于窗口中的内容，还可以控制是否重载页面，返回上一个文档或前进到下一个文档。

在框架方面，window对象可以处理框架与框架爱之间的关系，并通过这种关系在一个框架处理另一个框架汇中文档。window对象是所有其他对象的顶级对象，通过对window对象的子对象进行操作，可以实现更多的动态效果

### 属性和方法

- 属性

| 属性          | 描述                                       | 属性   | 描述                                       |
| ------------- | ------------------------------------------ | ------ | ------------------------------------------ |
| document      | 对话框中显示的当前文档                     | top    | 最顶层的浏览器对话框                       |
| frames        | 表示当前对话框中所有frame对象的集合        | parent | 包含当前对话框的父对话框                   |
| location      | 指定当前文档的URL                          | opener | 打开当前对话框的父对话框                   |
| name          | 对话框的名字                               | closed | 当前对话框是否关闭的逻辑值                 |
| status        | 状态栏中的当前信息                         | self   | 当前对话框                                 |
| defaultstatus | 状态栏中的当前信息                         | screen | 表示用户屏幕，提供屏幕尺寸、颜色深度等信息 |
| navigator     | 表示浏览器对象，用于获得与浏览器相关的信息 |        |                                            |

- 方法

| 方法                       | 描述                                                         |
| -------------------------- | ------------------------------------------------------------ |
| alert()                    | 弹出一个警告对话框                                           |
| confirm()                  | 在确认对话框中显示指定的字符串                               |
| prompt()                   | 弹出一个提示对话框                                           |
| open()                     | 打开浏览器对话框并且显示由URL或名字引用的文档，并设置创建对话框的属性 |
| close()                    | 关闭被引用的对话框                                           |
| focus()                    | 将被引用的对话框放在所有打开对话框的前面                     |
| blur()                     | 将被引用的对话框放在所有打开对话框的后面                     |
| scrollTo(x,y)              | 把对话框滚动到指定的坐标                                     |
| scrollBy(offsetx, offsety) | 按照指定位移量滚动对话框                                     |
| setTimeout(timer)          | 在指定的毫秒数过后，对传递的表达式求值                       |
| setInterval(interval)      | 指定周期性执行代码                                           |
| moveTo(x, y)               | 将对话框移动到指定坐标处                                     |
| moveBy(offsetx,offsety)    | 将对话框移动到指定的位移量处                                 |
| resizeTo(x,y)              | 设置对话框的大小                                             |
| resizeBy(offsetx,offsety)  | 按照指定的位移量设置对话框的大小                             |
| print()                    | 相当于浏览器工具栏中的"打印"按钮                             |
| navigate(URL)              | 使用对话框显示URL自定的页面                                  |
| status()                   | 状态条，位于对话框下部的信息条                               |
| Defaulestatus()            | 状态条，位于对话框下部的信息条                               |

- 使用

```
// window对象可以直接调用其方法和属性
window.属性名
window.方法名(参数列表)

// 特定窗口
// self代表当前窗口，parent代表父级窗口
parent.属性名
parent.方法名(参数列表)
```

### 窗口操作

- 打开窗口

```
var windowvar = window.open(url, windowname[,location])
```

参数说明

```
url:目标窗口的URL,若为空字符串，则打开一个空白窗口，允许用write()方法创建动态HTML
windowname:Window对象的名称。可最为属性值在<a><form>标记的target属性中出现
location：打开窗口的参数,如下所示
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

eg:
window.open("PopUp.html", "popup", "height=300,width=400,top=400,left=150, status");
```

- 关闭窗口

```
// 关闭当前窗口
window.close()
close()
this.close()

// 关闭其他窗口
// windowname为之前打开窗口的句柄
windowname.close()
```

- 控制窗口

```
// 移动窗口
window.moveTo(x,y)  // (x,y)窗口相对左上角的坐标
window.resizeTo(x,y)	// (x,y)窗口的水平和垂直宽度
window.resizeBy()  //按指定的数量增加窗口大小，可水平或垂直增加，负值为收缩大小。上左边缘保持不变
// 屏幕设置
window.screen.width  // 屏幕宽度
window.screen.height  // 屏幕高度
window.screen.colorDepth  // 屏幕色深
window.screen.availWidth  // 可用宽度
window.screen.avaiHeight  // 可用高度
// 窗口滚动
window.scroll(x, y)  //滚动到指定的水平和垂直位置
window.scrollTo(x, y)  //滚动到指定的水平和垂直位置
window.scrollBy(x, y)  //滚动指定的像素；可指定水平或垂直的滚动值
window.scrollByLines()	 //垂直滚动文档指定的行数
window.scrollByPages()	 //垂直滚动文档指定的页数
// 窗口历史
window.history.length  // 历史列表的长度
window.history.current  // 当前文档的URL
window.history.next  // 历史列表的下一个URL
window.history.prvious  // 历史列表的上一个URL
window.history.back()  // 退回前一页
window.history.forward()  // 重新进入下一页
window.history.go([num])  // 进入指定的页面
```

### 模态对话框窗口

之前的窗口都是非模态的(modeless),用户可以与其他窗口交互

模态(modal)窗口获取焦点并禁用应用程序的其他部分，直到关闭窗口为止

- 标准的弹出对话框

```javascript
// 警告框
window.alter(str); // str为子啊警告框汇总显示的提示字符串

// 确认框
windos.confirm(question) // question为在对话框中显示的纯文本

// 提示框
window.prompt(str1, str2) // str1为在对话框中要被显示的信息，若忽略则不显示，str2为对话框内输入框的值，若忽略则被设置为undefined
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

### 窗口事件

```
window.通用事件名=要执行的js代码
```

| 事件     | 描述                                       |
| -------- | ------------------------------------------ |
| onfocus  | 当浏览器窗口获得焦点时激活                 |
| onblur   | 浏览器串口失去焦点时激活                   |
| onload   | 当文档完全载入时触发                       |
| onunload | 但文档未载入时触发                         |
| onresize | 当用户改变窗口大小时触发                   |
| onerror  | 当出现JavaScript错误时触发一个错误处理事件 |

## 常用文档对象

### Document对象

代表浏览器窗口中的文档，该对象是window对象的子对象。由于window对象是DOM对象模型中默认对象，因此window对象中的方法和子对象不需要使用window来引用。通过Document对象可访问HTML文档中包含的任何HTML标记，并可动态地改变HTML标记中的内容。如表单、图像、表格和超链接等。

- 属性

| 属性             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| alInkColor       | 链接文字被单击时的颜色，对应于<body>标记中的alink属性        |
| all[]            | 存储HTML标记的一个数组(该属性本身也是一个对象)               |
| anchors[]        | 存储锚点的一个数组(该属性本身也是一个对象)                   |
| bgColor          | 文档的背景颜色，对应于<body>标记中的bgcolor属性              |
| cookie           | 表示cookie的值                                               |
| fgColor          | 文档的文本颜色(不包含超链接的文字)，对应于<body>中的text属性值 |
| forms[]          | 存储窗体对象的一个数组(该属性本身也是一个对象)               |
| fileCreatedDate  | 创建文档的日期                                               |
| fileModifiedDate | 文档最后修改的日期                                           |
| fileSize         | 当前文件的大小                                               |
| lastModified     | 文档最后修改的时间                                           |
| images[]         | 存储图像对象的一个数组(该属性本身也是一个对象)               |
| linkColor        | 未被访问的链接文字的颜色，对应<body>标记中的link属性         |
| links[]          | 存储link对象的一个数组(该属性本身也是一个对象)               |
| vlinkColor       | 表示已访问的链接文字的颜色，对应于<body>标记的vlink属性      |
| title            | 当前文档标题对象                                             |
| body             | 当前文档主体对象                                             |
| readyState       | 获取某个对象的当前状态                                       |
| URL              | 获取或设置URL                                                |

- 方法

| 方法           | 说明                                                     |
| -------------- | -------------------------------------------------------- |
| close          | 关闭文档的输出流                                         |
| open           | 打开一个文档输出流并接受write和writeln方法的创建页面内容 |
| write          | 向文档中写入HTML或JavaScript语句                         |
| writeln        | 向文档中写入HTML或JavaScript语句，并以换行符结束         |
| createElement  | 创建一个HTML标记                                         |
| getElementById | 获取指定id的HTML标记                                     |

eg

```javascript
// 链接文字颜色设置
[color=]documnet.alincolor[=setColor]
[color=]documnet.linColor[=setColor]
[color=]documnet.vlincolor[=setColor]
// 文档背景色和前景色设置
[color=]documnet.bgColor[=setColor]
[color=]documnet.fgColor[=setColor]
// 获取并设置URL
[url=]documnet.URL[=setUrl]
// 在文档中输出数据
document.write(text)
document.writeln(text)
// 动态添加一个HTML标记
sElement = document.createElement(sName)
// 获取文本框并修改其内容
sElement = document.getElementById(id)
```

### 表单对象

Document对象的forms属性可以返回一个数组，数组中的每个元素都是一个Form对象。可以实现输入文字、选择选项和提价数据等功能

Form对象代表了HTML文档中的表单，由于表单是由表单元素组成的，故Form对象也包含多个子对象

- 访问

> 访问表单

```
// 方法一：
document.forms[0]
// 方法二：
document.formname
// 方法三：
document.getElementById("formId")
```

> 访问表单元素

```
// 方法一：
document.form1.elements[0]
// 方法二：
document.form1.text1
// 方法三：
document.getElementById("elementId")
```

- 属性

| 属性     | 说明                                                 |
| -------- | ---------------------------------------------------- |
| name     | 返回或设置表单的名称                                 |
| action   | 返回或设置表单提交的URL                              |
| method   | 返回或设置表单提交的方式，可取值为get或post          |
| encoding | 返回或设置表单信息提交的编码方式                     |
| id       | 返回或设置表单的id                                   |
| length   | 返回表单对象中元素的个数                             |
| target   | 返回或设置提交表单时目标窗口的打开方式               |
| elements | 返回表单对象中的元素构成的数组，数组中的元素也是对象 |

- 方法

| 方法     | 说明                                             |
| -------- | ------------------------------------------------ |
| reset()  | 将所有表单元素重置为初始值，相当于单击了重置按钮 |
| submit() | 提交表单数据，相当于单击了提交按钮               |

### 图像对象

<img>标签，并将其中的src属性设置为图片的URL

- 访问

```
// 方法一：
document.images[0]
// 方法二：
document.images[imageName]
// 方法三：
document.getElementById("imageId")
```

- 属性

| 属性   | 说明                                             |
| ------ | ------------------------------------------------ |
| border | 表示图片便捷宽度，以像素为单位                   |
| height | 表示图像的高度                                   |
| hspace | 表示图像与左边和右边的水平空间大小，以像素为单位 |
| lowsrc | 低分辨率显示候补图像的URL                        |
| name   | 图片名称                                         |
| src    | 图像URL                                          |
| vspace | 便是上下边界垂直空间的大小                       |
| width  | 表示图片的宽度                                   |
| alt    | 鼠标经过图片时显示的文字                         |

## DOM对象模型

文档对象模型(Document Object Model)

```
Document
创建一个网页并将该网页加载至Web中，DOM就会根据这个网页创建一个文档对象
Object
文档对象就是文档中元素与内容的数据集合，与特定对象相关联的变量是属性，通过某个特定对象去带哦用的函数称为方法
Model
文档对象的模型为树状模型，网页中的各个元素与内容变现为一个个相互连接的节点
```

### DOM分层

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
        <h3>三号标题</h3>
		<p>加粗内容</p>
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

节点

```
根节点
父节点
子节点
兄弟节点
后代
叶子节点
```

节点类型

```
元素节点
文本节点
属性节点
```

### 节点属性

在文档中获取了一个元素之后，可以使用如下属性导航到相关联元素和属性

| 属性            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| nodename        | 节点名称                                                     |
| nodevalue       | 节点的值，常应用于文本节点                                   |
| nodeType        | 节点的类型                                                   |
| parentNode      | 返回当前节点的直接父节点，若是根节点则为空                   |
| childNodes      | HTMLCollection对象，包含了该节点所有的直接子节点列表         |
| firstChild      | 返回当前节点的第一个子节点                                   |
| lastChild       | 返回当前节点的最后一个子节点                                 |
| previousSibling | 返回当前节点的前一个兄弟节点， 若是父节点的第一子节点，则属性为空 |
| nextSibling     | 返回当前节点的后一个兄弟节点， 若是父节点的最后一子节点，则属性为空 |
| attributes      | 元素的属性列表                                               |

eg

```javascript
// 访问指定的节点
[sName=]obj.nodename
[sType=]obj.nodeType  // 有element/attribute/text/comment/document/documenTypye
[txt=]obj.nodeValue

// 遍历文档树
[pNode=]obj.parentNode
[cNode=]obj.firstChild
[cNode=]obj.lastChild
[cNode=]obj.previousSibling
[cNode=]obj.nextSibling
```

### 查找元素

```javascript
// 通过id属性定位元素，返回零个或一个元素，若无，返回null
var element = document.getElementById(id);
// 通过name属性定位元素，返回与元素类型相匹配的元素数组,若无，返回[]
var elementArray = document.getElementsByName(name);
// 返回与元素类型相匹配的元素数组,若无，返回[]
var elementArray = document.getElementsByTagName(name);
// 返回与class类型相匹配的元素数组,若无，返回[]
var elementArray = document.getElementsByClassName(names);
//通过css定位
var elementArray = document.querySelector(name)
var elementArray = document.querySelectorAll(names)
```

### 创建元素

创建单个元素

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

创建多个元素

```javascript
// 使用循环语句
function dc(){
    var aText = ["a","b","c","d"]
    for (var i=0; i<aText.length; i++)  //遍历节点
    {	
        var ce = document.createElement("p")  // 创建节点元素
        var cText = document.cteateTextNode(aText[i])  // 创建节点文本
        // 将新节点添加到页面上
        ce.appendChild(cText);
        document.body.appendChild(ce)
    }
}

//解决appendChild()方法添加每次刷新页面造成浏览器缓慢
function dc(){
    var aText = ["a","b","c","d"]
    var cdf = document.createDocumentFragment()  //创建文件碎片节点
    for (var i=0; i<aText.length; i++)  //遍历节点
    {	
        var ce = document.createElement("p")  // 创建节点元素
        var cText = document.cteateTextNode(aText[i])  // 创建节点文本
        // 将新节点添加到页面上
        ce.appendChild(cText);
        cdf.appendChild(ce)
    }
    document.body.appendChild(cdf)
}
```

### 移动元素

```javascript
// 添加
// appendChild()添加子节点至父节点(若已有子节点，则位于同级节点尾部)
var child = parentNode.appendChild(childNode);
// insertBefore()添加子节点位于某个同级子节点前方,
var child = parentNode.insertBefore(childNode, sibling);

// 复制
obj.cloneNode(true) // 深度复制，复制当前节点及其子节点
obj.cloneNode(false) // 简单复制，只复制当前节点

// 删除
// removeChild()删除父节点中的子节点
var removedElement = parentNode.removeChild(childNode);
// 删除某一元素
element.parentNode.removeChild(element)

// 替换
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

## DHTML对象模型

获取网页对象的另一种方法，可以不必了解文档对象模型的具体层次结果，而直接得到网页中所需的对象

```
innerHTMl
// 被多数浏览器支持
// 声明了元素含有的HTML文本，不包括元素本身的开始标记和结束标记，设置该属性可以为指定的HTML文本替换元素的内容
innerText
// 只被IE支持
// 只能声明元素包含的文本内容，即使是HTML文本，也被认为是普通文本而原样输出
outerHTMl
// 只被IE支持
// 替换整个目标节点的HTML文本，不仅仅更该了内容，还对元素本身进行了修改
outerText
// 只被IE支持
// 替换整个目标节点的文本内容，不仅仅更该了内容，还对元素本身进行了修改
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

### 事件概述

事件是一些可以通过脚本响应的页面动作。事件处理是一段JavaScript代码，总是与页面中的特定部分以及一定的事件相关联。当页面特定部分关联的时间发生时，事件处理器就会被调用。

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

若只是想知道某个按钮是否被单击，则是否监听捕获或冒泡事件并不重要。然而若有多个事件处理程序分配给了不同的元素，这两个事件就非常有用

事件处理程序可以连续执行；在调用下一个处理程序之前，当前处理程序必须完成。如果在同一个元素上分配了多个事件处理程序，则按照事件注册顺序连续执行

### 删除注册事件

对于addEventListener()/attachEvent()，由于可将多个事件处理程序分配给单个元素，为了正确删除，需要指定注册事件处理程序中所使用的所有相同信息(注册事件的元素，事件类型，注册的处理程序函数，表示是否在捕获阶段或冒泡阶段注册的标志)

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

### JS常用事件

| 分类     | 事件               | 说明                                                         |
| -------- | ------------------ | ------------------------------------------------------------ |
| 鼠标事件 | onclick            | 鼠标单击触发                                                 |
|          | ondbclick          | 鼠标双击触发                                                 |
|          | onmousedown        | 按下鼠标触发                                                 |
|          | onmouseup          | 鼠标按下松开后触发                                           |
|          | onmouseover        | 鼠标移动到某对象范围的上方时触发                             |
|          | onmousemove        | 鼠标移动时触发                                               |
|          | onmouseout         | 鼠标离开某对象范围时触发                                     |
| 键盘事件 | onkeypress         | 键盘上某键被按下且释放时触发                                 |
|          | onkeydown          | 键盘上某键被按下时触发                                       |
|          | onkeyup            | 键盘上某键被按下后松开时触发                                 |
| 页面相关 | onabort            | 图片在下载时被用户中断时触发                                 |
|          | onbeforeunload     | 当前页面的内容将要被改变时触发                               |
|          | onerror            | 出现错误时触发                                               |
|          | onload             | 页面内容完成时触发(页面加载)                                 |
|          | onresize           | 当浏览器的窗口大小被改变时触发                               |
|          | onunload           | 当前页面将被改变时触发                                       |
| 表单相关 | onblur             | 当前元素失去焦点时触发                                       |
|          | onchange           | 当前元素失去焦点且元素的内容发生改变时触发                   |
|          | onfocus            | 当某个元素获得焦点时触发                                     |
|          | onreset            | 当表单中RESET的属性被激活时触发                              |
|          | onsubmit           | 一个表单被提交时触发                                         |
| 滚动字幕 | onbounce           | 当Marquee内的内容移动至Marquee显示范围之外时触发             |
|          | onfinish           | 当Marquee元素完成需要显示的内容后触发                        |
|          | onstart            | 当Marquee元素开始显示内容时触发                              |
| 编辑事件 | onbeforecopy       | 当页面当前被选择内容将要复制到浏览者系统剪贴板时触发         |
|          | onbeforecut        | 当页面中的部分或全部内容被剪切到浏览者系统剪贴板时触发       |
|          | onbeforeeditfocus  | 当前元素将要进入编辑状态时触发                               |
|          | onbeforepaste      | 将内容要从浏览者的系统剪贴板中粘贴到页面上时触发             |
|          | onbeforeupdate     | 当浏览者粘贴系统剪贴板中内容时通知目标对象                   |
|          | oncontextmenu      | 当浏览者按下鼠标右键出现菜单时或者通过键盘的按键触发页面菜单时触发 |
|          | oncopy             | 当页面当前的被选择内容被复制后触发事件                       |
|          | concut             | 当页面当前的被选择内容被剪切后触发事件                       |
|          | ondrag             | 当某个对象被拖动时触发(活动事件)                             |
|          | ondragend          | 当鼠标拖动结束时触发                                         |
|          | ondragenter        | 当对象被鼠标拖动进入其容器范围内时触发                       |
|          | ondragleave        | 当对象被鼠标拖动离开其容器范围内时触发                       |
|          | ondragover         | 当被拖动的对象在另一对象容器范围内拖动时触发                 |
|          | ondragstart        | 当某对象将被拖动时触发                                       |
|          | ondrop             | 在一个拖动过程中，释放鼠标键时触发                           |
|          | onlosecapture      | 当元素失去鼠标移动所形成的选择焦点时触发                     |
|          | onpaste            | 当内容被粘贴时触发此事件                                     |
|          | onselect           | 当文本内容被选择时触发                                       |
|          | onselectstart      | 当文本内容的选择将开始发生时触发                             |
| 数据绑定 | onafterupdate      | 当数据完成由数据源到对象的传送时触发                         |
|          | oncellchange       | 当数据来源发生变化时触发                                     |
|          | ondataavailable    | 当数据接收完成时触发                                         |
|          | ondatasetchanged   | 数据在数据发生变化时触发                                     |
|          | ondatasetcomplete  | 当数据源的全部有效数据读取完毕时触发                         |
|          | onerrorupdate      | 当使用onBeforeUpdate时间触发取消了数据传送时，代替onAfterUpdate事件 |
|          | onrowwnter         | 当前数据源的数据发生变化并且有新的有效数据时触发             |
|          | onrowexit          | 当前数据源的数据将要发生变化时触发                           |
|          | onrowsdelete       | 当前数据记录将被删除时触发                                   |
|          | onrowsinserted     | 当前数据源将要插入新数据记录时触发                           |
| 外部事件 | onafterprint       | 当文档被打印后触发                                           |
|          | onbeforeprint      | 当文档即将被打印时触发                                       |
|          | onfilterchange     | 当某个对象的滤镜效果发生变化时触发                           |
|          | onhelp             | 当浏览者按下F1或浏览器的帮助菜单时触发                       |
|          | onpropertychange   | 当对象的属性之一发生变化时触发                               |
|          | onreadystatechange | 当对象的初始化属性值发生变化时触发                           |

### 键盘的键码值

```
# 字幕和数字键
a(A)~z(Z)		65~90
0~9				48~57

# 数字键盘上
0~9				96~105
*				106
+				107
Enter			108
-				109
.				110
/				111
F1~F12			112~123

# 控制键
Back Space		8
Tab				9
Clear			12
Enter			13
Shift			16
Control			17
Alt				18
Cape Lock		20
Esc				27
Spacebar		32
Page Up			33
Page Down		34
End				35
Home			36
Left Arrow		37
Up Arrow		38
Right Arror		39
Down Arror		40
Insert			45
Delete			46
Num Lock		144
;:				186
=+				187
,<				188
-_				189
.>				190
/?				191
`~				192
[{				219
\|				220
]}				221
'"				222
```





