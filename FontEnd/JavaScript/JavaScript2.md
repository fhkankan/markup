[TOC]

# JavaScript2

## 对象概述

对象是一些属性和函数的集合

### 属性

```
// 获取对象属性
对象名.属性名
// 设置对象属性
对象名.属性名 = 值
```

### 方法

```
对象名.方法名(参数)
```

### 种类

自定义对象

```
用户根据需要自己定义的新对象
```

内置对象

```
JavaScript中将一些常用的功能预先定义成对象，用户可以直接使用。
```

浏览器对象

```
浏览器根据系统当前配置和所载入的页面为JavaScript提供的一些对象，如document,window对象等。
```

### 上下文

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

### 命名空间

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
## 自定义对象

### 创建

- 对象字面量

```
var myObject = {
    color: "red",
    count: 5,
    log: function(){
        alert(this.color)
    }
}
```
- 自定义构造函数

```
// 构造函数
function Student(name,sex,age){
    this.name = name;
    this.sex = sex;
    this.age = age;
    this.showName = fucntion(){
        alert(this.name)
    };
}
// 创建对象
var student1 = new Student("LiLei", "man", 24)

注意：不能定义多个同名的构造函数。若定义了，则后面的函数会取代前面的函数。
```
- 内置对象

```
var student = new Object();
student.name = "LiLei";
student.sex = "man";
student.age = 24;
student.showName = fucntion(){
        alert(student.name)
    };
```

- 原型

所有对象都有一个原型。原型是实际对象被实例化的模型或蓝图

```
# 对redObject对象增加一个属性
redObject.isAvailable = true;

# 对所有使用Item构造函数的实例都增加属性,将属性添加到Item原型中
Item.prototype.isAvailable = true;
```

### 对象访问语句

- for … in 

用来遍历对象的每一个属性，每次都将属性名作为字符串保存在变量里

```
for (变量 in 对象){
    语句
}
```

- with

用在访问一个对象的属性或方法时避免重复引用指定对象名

```
with(对象名){
    语句
}
```

##常用内部对象

### Math

Math对象提供了大量的数学常量和数学函数，在使用Math对象时，不能使用new关键字创建对象实例，而应直接使用“对象名.成员”的格式来访问其属性和方法

- 属性

| 属性  | 说明         | 属性    | 说明                |
| ----- | ------------ | ------- | ------------------- |
| E     | 欧拉常量     | LOG2E   | 以2为底数的e的对数  |
| LN2   | 2的自然对数  | LOG10E  | 以10为底数的e的对数 |
| LN10  | 10的自然对数 | PI      | 圆周率常数          |
| SQRT2 | 2的平方根    | SQRT1_2 | 0.5的平方根         |

```
var pipValue = Math.PI  # 计算圆周率
var rootofTwo = Math.SQRT2  # 计算平方根
```



- 方法

| 方法       | 描述                                      | 方法           | 描述              |
| ---------- | ----------------------------------------- | -------------- | ----------------- |
| abs(x)     | x的绝对值                                 | log(x)         | x的自然对数       |
| acos(x)    | x弧度的反余弦值                           | max(n1,n2,...) | 参数列表最大值    |
| asin(x)    | x弧度的反正弦值                           | min(n1,n2,...) | 参数列表最小值    |
| atan(x)    | x弧度的反正切值                           | pow(x,y)       | x对y的次方        |
| atan2(x,y) | 从x轴到点(x,y)的角度，其值区间为(-PI, PI) | random()       | 返回[0,1)的随机数 |
| ceil(x)    | 大于或等于x的最小整数                     | round(x)       | 最接近x的整数     |
| cos(x)     | x的余弦值                                 | sin(x)         | x的正弦值         |
| exp(x)     | e的x乘方                                  | sqrt(x)        | x的平方根         |
| floor(x)   | 小于或等于x的最大整数                     | tan(x)         | x的正切值         |

````
// 随机取数
function numRandom(){
    var num_list = [2,3,4,8];
    var index = Math.floor(Math.random()*4)
    return num_list[index]
}

// 根据输入位数随机生成
function ran(digit){
    var result = "";
    for(i=0; i<digit; i++){
        result = result + (Math.floor(Math.random()*10));
    }
    alert(result);
}
````



### Number

常用于访问某些常量

- 创建Number对象

```
var numObj = new Number([value])

// 0
var num_obj = new Number(0)
```

- 属性

| 属性      | 说明   | 属性              | 说明         |
| --------- | ------ | ----------------- | ------------ |
| MAX_VALUE | 最大值 | NEGATIVE_INFINITY | 负无穷大的值 |
| MIN_VALUE | 最小值 | POSITIVE_INFINITY | 正无穷大的值 |

eg
```
var positive = Number.POSITIVE_INFINITY;
document.write(positive)
```
- 方法

```javascript
toString()
// 把Number对象转换成一个字符串，并返回结果,randix表示数字基数，默认为10
NumberObject.toString(radix)

toLocalString()
//把Number对象转换为本地格式的字符串
NumberObject.toLocalString()

toFixed()
// 将Number对象四舍五入为指定小数位数的数字，然后转换成字符串,num为指定位数，默认为0
NumberObject.toFixed(num)

toExponential()
// 利用科学计数法计算Number对象的值，然后将其转换为字符串,num为小数位数
NumberObject.toExponential(num)

toPrecision()
// 根据不同情况选择定点计数法或科学计数法，然后将转换后的数字转换成字符串, num为指定结果中有效数字的位数，默认尽可能多
NumberObject.toPrecision(num)
```

### Date

实现对日期和时间的控制

- 创建Date对象

```
dateObj = new Date()
dateObj = new Date(dateVal)
dateObj = new Date(year, month, date[,hours[, minutes[,seconds[,ms]]]])
```

参数

```
dateObj:必选项，要赋值为Date对象的变量名

dateVal:必选项，可是数字(1970年1月1日午夜间全球标准时间的毫秒数)，
		字符串(方式1："月 日，年 小时:分钟:秒"，月用英文表示，其余用数字表示，时间部分可省略；方式2："年/月/日 小时:分钟:秒")

year	必选项，完整的年份
month	必选项，月份(0~11)
date	必选项, 日期(1~31)
hours	可选项, 小时(0~23)
minutes	可选项，分钟(0~59)
seconds	可选项，秒(0~59)
ms		可选项，毫秒(0~999)
```

eg

```
// 当前日期和时间
var newDate = new Date()
// 指定时间
var newDate = new Date(2016,11,25)
var newDate = new Date(2016,11,25,13,12,56)
var newDate = new Date("Dec 25,2016 13:12:56")
var newDate = new Date("2016/12/25 13:12:56")
```

- 属性

| 属性        | 用法                 | 语法                      |
| ----------- | -------------------- | ------------------------- |
| constructor | 判断对象的类型       | object.constructor        |
| prototype   | 添加自定义的属性方法 | Date.prototype.name=value |

eg

```
// constructor
var newDate = new Date();
if (newDate.constructor==Date){
    docuent.write("日期类型对象")
};

// prototype
var newDate = new Date();
Date.prototype.mark=newDate.getFullYear();
```

- 方法

| 方法                 | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| Date()               | 返回系统当前的日期和时间                                     |
| getDate()            | 从Date对象返回一个月中的某一天(1~31)                         |
| getDay()             | 从Date对象返回一周中的某一天(0~6)                            |
| getMonth()           | 从Date对象返回月份(0~11)                                     |
| getFullYear()        | 从Date对象以4位数字返回年份                                  |
| getYear()            | 从Date对象以2位或4位数字返回年份                             |
| getHours()           | 返回Date对象的小时(0~23)                                     |
| getMinutes()         | 返回Date对象的分钟(0~59)                                     |
| getSeconds()         | 返回Date对象的秒数(0~59)                                     |
| getMilliseconds()    | 返回Date对象的毫秒(0~999)                                    |
| getUTCDate()         | 根据世界时从Date对象返回一个月中的某一天(1~31)               |
| ...                  | 根据世界时从Date对象返回...                                  |
| getTime()            | 返回1970年1月1日至今的毫秒数                                 |
| getTimezoneOffset()  | 返回本地时间与格林尼治时间的分钟差(GMT)                      |
| parse()              | 返回1970年1月1日午夜到指定时间(字符串)的毫秒数               |
| setDate()            | 设置Date对象中的月的某一天(1~31)                             |
| ...                  | 设置Date对象中...                                            |
| setUTCDate()         | 根据世界时设置Date对象中月份的一天(1~31)                     |
| ...                  | 根据世界时设置Date对象中...                                  |
| setTime()            | 通过从1960年1月1日午夜添加或减去指定数目的毫秒计算日期和时间 |
| toSource()           | 代表对象的源代码                                             |
| toString()           | 把Date对象转换为字符串                                       |
| toTimeString()       | 把Date对象的时间部分转换为字符串                             |
| toDateString()       | 吧Date对象的日期部分转换为字符串                             |
| toGMTString()        | 根据格林尼治时间，把Date对象转换为字符串                     |
| toUTCString()        | 根据世界时，把Date对象转换为字符串                           |
| toLocaleString()     | 根据本地时间格式，把Date对象转换为字符串                     |
| toLocaleTimeString() | 根据本地时间格式，把Date对象的时间部分转换为字符串           |
| toLocaleDateString() | 根据本地时间格式，把Date对象的日期部分转换为字符串           |
| UTC()                | 根据世界时，获得一个日期，然后返回1970年1月1日午夜到该日期的毫秒数 |
| valueOf()            | 返回Date对象的原始值                                         |

### Array

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
//指定数组长度
var arrayObj = new Array(3)
//直接创建
var aList2 = [1,2,3,'asd'];
//多维数组 
var aList = [[1,2,3],['a','b','c']];

方法二： 空+push
var colors= []
colors.push("red")
colors.push("green")
```

- 操作数组元素

> 输入输出

输入

```
方法一：
在数组元素确定的情况下，在定义Array元素时直接输入
var arr = new Array("a", "b", "c");

方法二：
运用Array的元素下标向其输入数组元素
var arr = new Array(7);
arr[3] = "a";

方法三：
for语句，用于批量输入
var n = 7;
var arr = new Array();
for (var i=0; i<n; i++){
    arr[i] = i;
}
```

输出

```
方法一：
下标获取指定元素值
var arr = new Array("a", "b", "c");
var third = arr[2]

方法二：
for语句遍历
var str = "";
var arr = new Array("a", "b", "c");
for (var i=0; i<4; i++){
    str = str + arr[i];
}

方法三：
用数组对象名输出所有的元素值
var  arr = new Array("a", "b", "c");
documnet.write(arr)
```

> 添加

虽然定义数组设置了数组元素的个数，但是并不固定

```
var  arr = new Array("a", "b", "c");
arr[2] = "1"
arr[3] = "2"
```

> 删除

delete运算符可以删除数组元素的值，但只会将元素恢复到未赋值状态，即undefined，并未真正删除元素，个数也没减少

```
var  arr = new Array("a", "b", "c");
delete arr[1]
document.write(arr)
```

- 属性

| 属性      | 说明                             | 语法                       |
| --------- | -------------------------------- | -------------------------- |
| length    | 数组的长度                       | arrayObject.length         |
| prototype | 为数组对象添加自定义的属性和方法 | Array.prototype.name=value |

eg

```
// length
var arr = new Array(1,2,3,4)
document.write(arr.length)

// prototype
Array.prototype.outLast=function(){
    document.write(this[this.length-1]);
}
var arr = new Array(1,2,3,4);
arr.outLast();
```

- 方法

| 方法             | 说明                                                         | 语法                                          |
| ---------------- | ------------------------------------------------------------ | --------------------------------------------- |
| concat()         | 连接两个或更多的数组，并返回结果                             | arrayObject.concat(arrayX,...)                |
| push()           | 向数组的末尾添加一个或多个元素，并返回新的长度               | arrayObject.push(newelement,...)              |
| unshift()        | 向数组的开头添加一个或多个元素，并返回新的长度               | arrayObject.unshift(newelement,...)           |
| pop()            | 删除并返回数组的最后一个元素                                 | arrayObject.pop()                             |
| shift()          | 删除并返回数组的第一个元素                                   | arrayObject.shift()                           |
| splice()         | 删除元素，并向数组添加新元素                                 | arrayObject.splice(start,length,element1,...) |
| reverse()        | 颠倒数组中元素的顺序                                         | arrayObject.reverse()                         |
| sort()           | 对数组元素进行排序                                           | arrayObject.sort(sortby)                      |
| slice()          | 从某个已有的数组返回选定的元素                               | arrayObject.slice(start, end)                 |
| toSource()       | 代表对象的源代码                                             |                                               |
| toString()       | 把数组转换为字符串，并返回结果                               | arrayObject.toString()                        |
| toLocaleString() | 把数组转换为本地字符串，并返回结果                           | arrayObject.toLocaleString()                  |
| join()           | 把数组的所有元素放入一个字符串，元素通过指定的分隔符进行分隔 | toLocaleString.join(separator)                |
| valueOf()        | 返回数组对象的原始值                                         |                                               |
| indexOf()        | 返回数组中第一次出现值得索引，若无则返回-1                   | arrayObject.indexOf(值)                       |

### String

用于操作和处理字符串，可以获取字符串的长度、提取子字符串、将字符串转换大小写等

- 创建string对象

string对象是动态对象，使用构造函数可以显式创建字符串对象。实际上javascript会自动在字符串和字符串对象之间进行转换，任何一个字符串常量都可看做一个String对象

```
var newstr = new String([StringText])
```

- 属性

| 属性        | 用法                           | 语法                      |
| ----------- | ------------------------------ | ------------------------- |
| length      | 长度                           | stringObject.length       |
| constructor | 对当前对象的构造函数的引用     | stringObject.constructor  |
| prototype   | 为字符串添加自定义的属性和方法 | String.protype.name=value |
eg
```
// length
var newStr = "abcdef";
var p = newStr.lenght;

// constructor
var newStr=new String("Hello Word!")
if (newStr.constructor==String){
    alert("字符串对象")
}

// prototype
String.protype.getLength=function(){
    alert(this.lenght);
}
var str = "abcde";
str.getLength;
```

- 方法

| 方法          | 说明                                               | 语法                                            |
| ------------- | -------------------------------------------------- | ----------------------------------------------- |
| charAt()      | 返回字符串中指定位置的字符                         | stringObject.charAt(index)                      |
| indexOf()     | 返回某个字符串在字符串中首次出现的位置，无则-1     | stringObject.indexOf(substring, startindex)     |
| lastIndexOf() | 返回某个字符串在字符串中最后出现的位置，无则-1     | stringObject.lastIndexOf(substring, startindex) |
| slice()       | 提取字符串的片段，并在新的字符串中返回被提取的部分 | stringObject.slice(startindex, endindex)        |
| substr()      | 从字符串的指定位置开始提取指定长度的子字符串       | stringObject.substr(startindex,length)          |
| substring()   | 提取字符串中两个指定的索引号之间的字符             | stringObject.substring(startindex, endindex)    |
| toLowerCase() | 把字符串转换为小写                                 | stringObject.toLowerCase()                      |
| toUpperCase() | 把字符串转换为大写                                 | stringObject.toUpperCase()                      |
| concat()      | 连接多个字符串                                     | stringObject.concat(string1,strign2,...)        |
| split()       | 把一个字符串分割为字符串数组                       | stringObject.split(separator, limit)            |

```
1、合并操作：“ + ”
2、将数字字符串转化为整数：parseInt(字符串) 
3、将数字字符串转化为小数：parseFloat(字符串)
4、字符串反转：字符串.split('').reverse().join('')
```
## 事件概述

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

