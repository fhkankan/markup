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

## 调试程序的方法

```
1、alert(变量名)			弹窗显示，会暂停程序运行
2、console.log(变量名)		浏览器控制台显示，不中断
3、document.title=变量名	页面标题显示，不中断
4、断点调试				   在浏览器的sources中设置断点
```

## 变量

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
```

## 对象

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

## 构造函数

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

## 原型

所有对象都有一个原型。原型是实际对象被实例化的模型或蓝图

```
# 对redObject对象增加一个属性
redObject.isAvailable = true;

# 对所有使用Item构造函数的实例都增加属性,将属性添加到Item原型中
Item.prototype.isAvailable = true;
```

## 继承

### 使用原型

javaScript通过一种被称为原型继承的方法提供对继承的支持





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

## 定时器

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
封闭函数是javascript中匿名函数的另外一种写法，创建一个一开始就执行而不用命名的函数。

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

## 变量作用域

```
变量作用域指的是变量的作用范围，javascript中的变量分为全局变量和局部变量。

1、全局变量：在函数之外定义的变量，为整个页面公用，函数内部外部都可以访问。
2、局部变量：在函数内部定义的变量，只能在定义该变量的函数内部访问，外部无法访问。

注意：在函数内部若使用var新建与全局变量同名的变量，则在函数内部调用时，优先调用内部变量，不对外部变量产生影响
```

## 获取元素与加载执行

```javascript
var 变量名 = document.getElelmentById('对象id')

注意：若把javascript写在元素的上面，就会出错，因为页面上从上往下加载执行的，javascript去页面上获取元素div1的时候，元素div1还没有加载，解决方法有两种：
第一种方法：将javascript放到页面最下边；
第二种方法：将javascript语句放到window.onload触发的函数里面,获取元素的语句会在页面加载完后才执行。
window.onload = function(){
        var oDiv = document.getElementById('div1');
    }
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

## 数组及操作方法

```
数组里面的数据可以是不同类型的
```

### 定义数组的方法

```
//对象的实例创建
var aList = new Array(1,2,3);

//直接创建
var aList2 = [1,2,3,'asd'];

//多维数组 
var aList = [[1,2,3],['a','b','c']];
```

### 操作数组中数据的方法 

```
1、获取数组的长度：aList.length;
2、用下标操作数组的某个数据：aList[0];
3、将数组成员通过一个分隔符合并成字符串:aList.join('-')
4、从数组最后增加成员或删除成员：aList.push(5);aList.pop();
5、将数组反转：aList.reverse();
6、返回数组中元素第一次出现的索引值，若没有返回-1：aList.indexOf(值)
7、在数组中增加或删除成员：aList.splice(2,1,7,8,9); //从02索引处，删除1个元素，然后在此位置增加'7,8,9'三个元素

批量操作数组中的数据，需要用到循环语句
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







