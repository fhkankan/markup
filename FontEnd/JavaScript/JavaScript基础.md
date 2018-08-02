[TOC]



# JavaScript基础

语言特定

```
解释性语言
基于对象
事件驱动，可以直接读用户或客户输入做出相应，无需经过Web服务程序
跨平台，依赖浏览器本身，与操作环境无关
安全性，不允许访问本地的硬盘，也不允许将数据存入服务器，不允许对网络文档进行修改和删除，只能通过浏览器实现信息浏览和动态交互
```

页面行为


```
部分动画效果、页面与用户的交互、页面功能
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

## 语言基础

### 数据类型

- 数值型

```javascript
0		//十进制
Oxff	//十六进制
0366	//八进制
3.14	//浮点型
6e+3	//科学计数浮点型
Infinity//无限大
NaN		//非数字

```

- 字符串

```
"你好"
'hello'
"nice to meet 'you'!"
'hello "everyone"'

# 模板字符串
// 在字符串中表示数值，可以通过字符串连接实现
let currentTemp = 19.5
const message = "the value is "+ currentTemp + "."
// ES6后，可使用字符串模板
let currentTemp = 19.5
const message = "the value is ${currentTemp}."

# 多行字符串
// ES6之后
const multiline = 'line\n'+
				'line2\n'+
				'line3'
				
# 数字字符串
const result = 3 + '10'
```

转义

```
\b		//退格
\n		//换行符
\t		//水平制表符
\f		//换页
\'		//单引号，在单引号嵌套时需用
\"		//双引号，在双引号嵌套时需用
\v		//垂直制表符
\r		//回车符
\\		//反斜杠
\OOO	//八进制整数
\xHH	//十六进制整数
\uhhh	//十六进制编码的Unicode字符
```

- 布尔型

```
true
false
```

- undefined

```
变量声明未初始化，它的值就是undefined
```

- null

```
表示空对象，如果定义的变量将来准备保存对象，可以将变量初始化为null,在页面上获取不到对象，返回的值就是null
```

- 符号

````
// ES6新特性

````

### 数据类型转换

- 字符串转换成数字

```
// 方法一
const numStr = '2.5'
const num = Number(numStr)
// 方法二
const a = parseInt('16', 16)// 转换为16进制，若缺省默认转换为10进制
const b = parseFloat('14.6 kpl')// 转换时会忽略非数字的其他字符串
```

- 转换成字符串

```
// 任何对象都有toStrign()方法，但大部分对象的返回是[object][object]
// 数字转字符串值
const s = 33.toString()
// 数组转字符串
const arr = [1, true, "hi"].toString()
```

- 转换成boolean

```
const n=0;//错误的值
const b1=!!n;//false
const b2=Boolean(n);//false
```

### 常量与变量

- 标识符

变量和常量的名字统统称作标识符，它们有些命名规范

```
变量、函数、属性、函数参数命名规范
1、区分大小写
2、第一个字符必须是字母、下划线（_）或者美元符号（$）
3、其他字符可以是字母、下划线、美元符或数字
4、不能包含空格或+、-等符号
5、不能使用关键字

匈牙利(驼峰)命名风格：
对象o Object 比如：oDiv
数组a Array 比如：aItems
字符串s String 比如：sUserName
整数i Integer 比如：iItemCount
布尔值b Boolean 比如：bIsComplete
浮点数f Float 比如：fPrice
函数fn Function 比如：fnHandler
正则表达式re RegExp 比如：reEmailCheck

蛇形命名法
current_temp_c
```

- 常量

```
在程序运行过程中保持不变的数据
```

> 声明

ES6之前

```
// 需要程序员自己约定，本质还是变量
var NUMBERFORA = 2;
```

ES6之后

```
//constant
//定义的变量不可修改(地址不能修改)，且必须初始化

const b = 2;//正确
// const b;//错误，必须初始化 
console.log('函数外const定义b：' + b);//有输出值
// b = 5;
// console.log('函数外修改const定义b：' + b);//无法输出 
```
- 变量

```
指程序中一个已经命名的存储单元，主要作用是为数据操作提供存放信息的容器。
有变量名和变量值
定义之后，可以修改
```

> 声明

var

```
全局变量，若在函数中声明同名变量，则函数内部的变量会屏蔽外部变量
可以重复定义，后面的值会覆盖之前的值
可以在申明前被引用，输出undefined,不会报错
//原因是var声明的变量都会被提升到作用域的顶部
```

eg

```
x;		//undefined 
var x=3;
x;		//3

var x = 3;
x;		//3
var x = 4;
x;		//4


//弱类型语言，变量类型由它的值来决定。 定义变量需要用关键字 'var'
var iNum = 123;
//同时定义多个变量可以用","隔开，公用一个‘var’关键字
var iNum = 45,sTr='qwe',sCount='68';

var a = 1;
// var a;//不会报错
console.log('函数外var定义a：' + a);//可以输出a=1
function change(){
a = 4;
console.log('函数内var定义a：' + a);//可以输出a=4
} 
change();
console.log('函数调用后var定义a为函数内部修改值：' + a);//可以输出a=4
```

ES6之后新增let关键字

let

```
块级作用域，函数内部使用let定义后，对函数外部无影响
不能重复定义
若引用的变量未声明会报错
```

eg

```
x;		//报错
let x=3; //未执行

let x = 3;
let x = 3;//报错

let c = 3;//全局作用域
console.log('函数外let定义c：' + c);//输出c=3
function change(){
let c = 6;// 块级作用域
console.log('函数内let定义c：' + c);//输出c=6
} 
change();
console.log('函数调用后let定义c不受函数内部定义影响：' + c);//输出c=3
```

> 赋值

```
var lesson = "English"
//或者
var lesson;
lesson = "English"
```

> 共有特性

```
所有变量都拥有可以读取和更新的特性
value		属性值，其为默认属性
writable	若属性被更新， 将其设置为true
enumerable	在枚举对象成员时，若该属性应被包括在内，设置为true
configurable若属性可以被删除，且该特性可以被修改，设置为true
```

### 运算符

> 算数运算符

| 运算符 | 描述 | 示例       |
| ------ | ---- | ---------- |
| +      | 加   | 3+6        |
| -      | 减   | 6-2        |
| *      | 乘   | 2*3        |
| /      | 除   | 12/3       |
| %      | 求模 | 7%4        |
| ++     | 自增 | i=6; j=i++ |
| --     | 自减 | i=6; j=i-- |

> 字符串运算符

| 运算符 | 描述                               | 示例                          |
| ------ | ---------------------------------- | ----------------------------- |
| +      | 连接两个字符串                     | 'a'+'b'                       |
| +=     | 连接两个字符串，并将结果赋给第一个 | var name = 'a'<br>name += 'b' |

> 比较运算符

| 运算符 | 描述                     | 示例      |
| ------ | ------------------------ | --------- |
| <      | 小于                     | 1<6       |
| >      | 大于                     | 4>3       |
| <=     | 小于等于                 | 10<=10    |
| >=     | 大于等于                 | 10>=10    |
| ==     | 等于，判断值不判断类型   | '5'==5    |
| ===    | 绝对等于，判断值和类型   | '5'==='5' |
| !=     | 不等于，判断值不判断类型 | 4 != 5    |
| !==    | 不绝对等于，判断值和类型 | '4'!=4    |

> 赋值运算符

| 运算符 | 描述       | 示例  |
| ------ | ---------- | ----- |
| =      | 赋值       | a =10 |
| +=     | 求和后赋值 | a+=10 |
| -=     | 求差后赋值 | a-=10 |
| *=     | 求乘后赋值 | a*=10 |
| /=     | 求除后赋值 | a/=10 |
| %=     | 求模后赋值 | a%=10 |

> 逻辑运算符

| 运算符 | 描述 | 示例   |
| ------ | ---- | ------ |
| &&     | 与   | a&&b   |
| \|\|   | 或   | a\|\|b |
| ！     | 非   | !a     |

> 短路求值

```
const skipIt = true;
let x = 0;
const result = skipIt || x++ //true,x=0
// 若skipIt=false,则result=true,x=1

const doIt = false;
let x = 0;
const result = doIt && x++ //false,x=0
// 若doIt=true, 则result=0，x=1

// if语句转换为短路求值语句
if(!options) options = {};
//转换为
options = options || {};
```

> 条件运算符

```
表达式？结果1：结果2

// if..else语句转换为条件表达式
if(isPrime(n)){
    label = 'prime';
} else {
    label = 'no-prime';
}
// 转换
label = isPrime(n)? 'prime':'no-prime';
```

> 其他运算符

逗号

```
// 逗号将多个表达式排在一起，整个表达式的值为最后一个表达式的值
var a,b,c,d;
a=(b=3,c=4,d=5)
```

typeof

```
// 判断操作数的数据类型，返回一个字符串
typeof 操作数
```

new

```
// 创建一个新的内置对象实例
对象实例名称 = new 对象类型(参数)
对象实例名称 = new 对象类型
```
### 解构赋值

ES6中允许将一个对象或者数组分解成多个单独的值

```
// 变量名需与对象中的属性名一致
const obj = {a:1,b:2,c:3};
const {a,b,c}=obj;
// 变量的个数需与对象中的个数一致
const arr = [1，2，3，4];
const [a,b,c,d]=arr;
```

### 注释

```
javascript语句开始可缩进也可不缩进，缩进是为了方便代码阅读，一条javascript语句应该以“;”结尾;

// 单行注释
/*  
    多行注释
    1、...
    2、...
*/
```

## 基本语句

### 条件语句

> if

```
//if语句
if(){}
//if...else
if(){}
else{}
//多重if else语句
if(){}
else if(){}
else if(){}
else{}
//嵌套
if(){
    if(){}
    else{}
}
else{
    if(){}
    else{}
}
```

> switch

```
switch(表达式){
    case 常量表达式1：
    	语句1；
    	break；
    case 常量表达式2：
    	语句2；
    	break;
    ...
    default:
    	语句n;
    	break
}
```

### 循环语句

> while

```
# while循环
while(条件)
	{循环体}
```

> do…while

```
do{
    语句
}while(表达式)
```

> for

```
for（var i=0;i<len;i++）
	{循环体}
```

> switch

```
switch(表达式){
    case 值1：
    	// 执行体1
    	[break;]
    case 值2：
    	// 执行体2
    	[break;]
    ...
    default:
    	// 执行体
    	[break;]
}
```

> for...in

```
//为循环对象中有属性key而设计
for (变量 in 对象)
	{执行体}
```

> for...of

```
// ES6新增，可遍历任何可迭代对象
for(变量 of 对象)
	{执行体}
```

### 跳转语句

> continue

```
只能用于while、for、do...while
跳过本次循环，并开始下一次循环
```

> break

```
通常用于while、for、do...while
跳出循环
```

> return

```
结束执行的函数
```

### 异常处理

`try...catch`块只在同一个函数的作用域内才有效

`try…catch…finally`

```
try{
    语句
}catch(exception){
    语句
}finally{
   	语句
}
```

Error对象属性

```
name	异常类型字符串
message	实际的异常信息
```

throw抛异常

```
throw new Error("自定义异常信息")
```
eg

```javascript
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

## 函数

### 定义

> function

```
function 函数名([参数1，参数2，...]){
    语句
    [return 返回值]
}
```

> 匿名函数

```
var 变量名 = fucntion([参数1，参数2，...]){
    语句
    [return 返回值]
}
```

> Function()

```
var 变量名 = new Function("参数1", "参数2", ... "函数体")
```

### 特殊参数

ES5中默认参数

```
function defaultValue (val, defaultVal) {
  return val === undefined ? defaultVal : val
}

function doSomething (name, age) {
  name = defaultValue(name, 'default name')
  age  = defaultValue(age , 18)

  console.log(name, age)
}
```

ES6中，新增了可变参数和默认参数

> 可变参数

```
// 使用展开操作符(...)，是最后一个参数
fucntion addPrefix(prefix, ...words){
    const prefixedWords = [];
    for(let i=0; i<words.length; i++){
		prefixedWords[i] = prefix + words[i];
	}
	return prefixedWords
}

addPrefix("con", "verse", "vex")//["converse","convex"]
```

> 默认参数

```
fucntion f(a,b="default",c=3){
    return '${a}-${b}-${c}'
}

f()//"undefined-default-3"
f(5) //"5-default-3"
f(5,6,7)//"5-6-7"
```

### 调用

> 函数提升

```
函数声明会被提升至它们作用域的顶部，允许在函数声明之前调用
f()
function f(){
    console.log('f')
}
```

> 简单调用

```
函数名(传递给函数的参数1，参数2, ...)
```

> 引用调用

```
// 变量
const f = 函数名
f()
// 对象属性
const obj = {};
obj.f = 函数名;// 将函数名赋值给对象的属性
obj.f();// 执行函数
//数组元素
const arr = [1,2,3];
arr[1] = 函数名;
arr[1]();
```

> 事件响应中

```
<input type="button" value="提交" onClick="函数名(参数)">
```

> 链接中

```
<a href="javascript: 函数名(参数)">
```

### 编译执行

```
# 变量与函数预解析 
JavaScript解析过程分为两个阶段，先是编译阶段，然后执行阶段，在编译阶段会将function定义的函数提前，并且将var定义的变量声明提前，将它赋值为undefined。
```

### 作用域

```
变量作用域指的是变量的作用范围，javascript中的变量分为全局变量和局部变量。

全局变量：在函数之外定义的变量，为整个页面公用，函数内部外部都可以访问。
局部变量：在函数内部定义的变量，只能在定义该变量的函数内部访问，外部无法访问。
块级作用域：块是由{}括起来的一系列语句，let和const声明的变量处于块作用域中，仅仅在代码块中有效

变量屏蔽：在函数内部若使用var新建与全局变量同名的变量，则在函数内部调用时，优先调用内部变量，不对外部变量产生影响
```

### 封闭函数(闭包)

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

### 内置函数

| 函数                 | 用途       | 说明                                           |
| -------------------- | ---------- | ---------------------------------------------- |
| parseInt(sting, [n]) | 数值处理   | 字符转换为整型                                 |
| parseFloat(string)   | 数值处理   | 字符转换为浮点型                               |
| isNaN(num)           | 数值判断   | 是否为NaN                                      |
| ifFinite(num)        | 数值判断   | 是否为无穷大                                   |
| eval(string)         | 字符串处理 | 计算字符串表达式的值，执行其中的JavaScript代码 |
| escape(string)       | 字符串处理 | 将特殊字符进行编码                             |
| unescape(string)     | 字符串处理 | 将编码后的字符串进行解码                       |
| encodeURL(url)       | 字符串处理 | 将URL字符串进行编码                            |
| decodeURL(url)       | 字符串处理 | 对已编码的URL字符串进行解码                    |

### 严格模式

ES5的语法允许存在隐式全局变量。若忘记使用var声明某个变量，js会认为开发人员在引用一个全局变量，若该变量不存在，则会自动创建。

为了避免这个现象，js引入严格模式，它能阻止隐式全局变量。

```
# 全部js文件使用
// 在代码最前面，单独插入一行字符串，单双引号均可
"use strict"

# 避免在全局作用域中使用
//避免在每个函数中都手动开启，将所有代码，封装近一个立即执行的函数中
(function(){
    'use strict';
    // 所有代码从这里开始...代码会按照严格模式执行
    // 不过严格模式不会干扰组合在一起的其他脚本
})
```

## 迭代器

ES6引入了迭代器和生成器

```
# 数组就是迭代器
const book = [
    "红楼梦",
    "三国演义",
    "水浒传",
    "西游记"
]

# 取值
const it = book.values();
it.next() // {value: "红楼梦", done: false}
it.next() // {value: "三国演义", done: false}
it.next() // {value: "水浒传", done: false}
it.next() // {value: "西游记", done: false}
it.next() // {value: undefined, done: true}
```

迭代协议

```
若一个类提供了一个符号方法Symbol.iterator,这个方法返回一个具有迭代行为的对象，那么这个类就是可迭代的。
```

斐波那契额数列

```
class FibonacciSeauence{
    [Symbol.iterator](){
		let a = 0, b = 1;
		return {
            next(){
                let rval = {value:b,done:false};
                b += a;
                a = rval.value;
                return rval;
            }
		}
	}
}
```

求第十个

```
const fib = new FibonacciSeauence();
let i = 0;
for(let n of fib ){
    console.log(n);
    if(++i > 9)
    	break;
}
```

## 生成器

ES6出现生成器。

生成器提供了两种能力：

```
控制函数执行的能力，使函数能够分步执行；
与执行函数对话的能力
```

生成器与一般函数的不同处

```
1.函数可以通过使用域(yield),在其运行的任意时刻将控制权交还给调用方；
2.调用生成器时候，它并不是立即执行。而是会回到迭代器中。函数会在调用迭代器的next方法时执行。
```

在JavaScript中，生成器需要在function关键字后面加一个通配符(*)来指明

```
// 彩虹色彩的生成器
function* rainbow(){
    yield 'red;
    yield 'orange';
    yield 'yellow';
    yield 'green';
    yield 'blue';
    yield 'indigo';
    yield 'violet';
}
// 调用
const it = rainbow();
it.next(); // {value: "red", done: false}
it.next(); // {value: "orange", done: false}
it.next(); // {value: "yellow", done: false}
it.next(); // {value: "green", done: false}
it.next(); // {value: "blue", done: false}
it.next(); // {value: "indigo", done: false}
it.next(); // {value: "violet", done: false}
it.next(); // {value: "undefined", done: true}
// 循环遍历
for (let color of rainbow()){
    console.log(color)
}
```

- 双向交流

生成器与其调用者进行双向交流，是通过yield表达式实现。

```
function* interrogate(){
    const name = yield "what is your name?";
    const color = yield "what is your favorite color?";
    return "${name}'s favorite color is $(color)";
} 

// 调用
const it = interrogate();
it.next() // {value: "what is your name?", done: false}
it.next() // {value: "what is your favorite color?", done: false}
it.next() // {value: "what is your favorite color?", done: false}
it.next() // {value: undefined, done: true}
// 或
const it = interrogate();
it.next(); // {value: "what is your name?", done: false}
it.next('Li'); // {value: "what is your favorite color?", done: false}
it.next('green'); // {value: "Li's favorite color is green", done: true}
```

- 返回值

yield表达式本身不能让生成器结束，即使是生成器的最后一个语句。在生成器的任何位置调用return都会使done得值变为true，而value值则是任何被返回的值

```
function* abc(){
    yield 'a';
    yield 'b';
    return 'c';
}

const it = abc();
it.next(); // {value: 'a', done:false}
it.next(); // {value: 'b', done:false}
it.next(); // {value: 'c', done:true}

// 使用for...of中，c不会出现
// 使用生成器时，不在意done为true时的值
for (let l of abc()){
	console.log(l)
}

// 生成器中的return常常无返回值
```

## 异步

### 回调

回调就是写一个函数，然后在未来的某个时刻调用它。通常，会把这些回调函数提供给别的函数，或者将它们的属性，一般来说，回调时匿名函数

错误优先回调是个好习惯

- 计时器

```
var timeoutID;
// 创建计时器
function delayedAlert() {
  timeoutID = window.setTimeout(slowAlert, 2000);
}
// 延时执行函数
function slowAlert() {
  alert('That was really slow!');
}
//清除计时器
function clearAlert() {
  window.clearTimeout(timeoutID);
}
```

循环执行函数

```
cosnt start = new Date();
let i = 0;
const intervalId = setInterval(function(){
    let now = new Date();
    if(now.getMinutes() !== start.getMinutes() || ++i>10)	{
       return clearInterval(intervalId) 
    }
})
```

setTimeout,setInterval,clearTimeout,clearInterval都是定义在全局对象中(浏览器是window，node中是global)

- scope和异步执行

每当一个函数被调用，都创建了一个闭包：所有在函数内部创建的变量(包括形参)只在有被访问的时候才存在

```
function countdown(){
    let i;// i在for循环外部，计时器执行了6次-1
    console.log("Countdown:");
    for(i=5; i>=0; i--){
        setTimeout(function(){
            console.log(i===0?"GO!":i);
        }, (5-i)*1000)
    }
}
countdown();

function countdown(){
    console.log("Countdown:");
    for(let i=5; i>=0; i--){
        setTimeout(function(){
            console.log(i===0?"GO!":i);
        }, (5-i)*1000)
    }
}
countdown();
```

### promise

promise是对象，可以被到处传递，若在异步进程中，需要把结果在别的地方被处理，可以直接把promise传递给它们

- 创建

```
function countdown(seconds){
	// 创建带有函数的promise实例，包含resolve和reject的回调
    return new Promise(function(resolve, reject){
        for (let i = seconds; i>=0; i--){
            setTimeout(fucntion(){
                if(i>0){
                    cosnole.log(i + '...');
                }
                else{
                    resolve(console.log("GO!"));
                }
            }, (seconds-i)*1000)
        }
    });
}
```

- 使用promise

```
countdown(5).then(
	// 满足的回调
	function(){
        console.log("countdown completed successfully");
	},
	// 错误的回调
	function(err){
		console.log("countdown experienced an error: " + err.message);
	}
)

//catch，把两个处理器分开
const p = countdown(5);
p.then(function(){
    console.log("countdown completed successfully");
})
p.catch(function(err){
    console.log("countdown experienced an error: " + err.message);
})
```

- 事件

```
const EventEmitter = require('events').EventEmitter;

class Countdown extends EventEmitter{
    constructor(seconds, superstitious){
        super();
        this.seconds = seconds;
        this.superstitious = !!superstitious
    }
    go(){
		const countdown = this;
		const timeoutIds = [];
		return new Promise(function(resolve, reject){
            for(let i=countdown.seconds; i>=0; i--){
                timeoutIds.push(setTimeout(function(){
				if(countdown.superstitious && i === 13){
                    // 清除所有pending的timeouts
                    timeoutIds.forEach(clearTimeout);
                    return reject(new Error('DEFINITELY NOT COUNTING THAT'));
				}
				countdown.emit('tick', i);
				if(i === 0)resolve();
				},(countdown.seconds-i)*1000));
            }
		});
	}
}
```

- promise链

```
funciton launch(){
    return new Promise(function(resolve,reject){
        console.log("Lift off!");
        setTimeout(function(){
            resolve("In orbit!");
        }, 2*1000)
    })
}
```

链式调用

```
const c = new Countdown(5)
	.on('tick', i => console.log(i + '...'));
	
c.go()
	.then(launch)
	.then(function(msg){
        console.log(msg);
	})
	.catch(function(err){
        console.error("Houston, we have a problem...")
	})
```

- 避免不被处理的promise

promise可以简化异步代码，同时确保回调函数不会被多次调用，但却不能避免那些应为promise没有被处理而产生的问题。

处理方式：给promise设定一个特定的超时。若没有在一段合理的时间内被处理，就自动被拒绝。

未被处理

```
funciton launch(){
    return new Promise(function(resolve,reject){
    	if(Math.random()<0.5) return; // rocket failure
        console.log("Lift off!");
        setTimeout(function(){
            resolve("In orbit!");
        }, 2*1000)
    })
}
```

添加超时

```
function addTimeout(fn, timeout){
    if(timeout == undefined){
        timeout = 1000;// 默认超时
    }
    return function(...args){
        return new Promise(function(resolve, reject){
            const tid = setTimeout(reject, timeout,
            new Error("promise timed out"));
        fn(...args)
        	.then(function(...args){
                clearTimeout(tid);
                resolve(...args);
        	})
        	.catch(function(...args){
                clearTimeout(tid);
                reject(...args);
        	});
        });
    }
}
```

使用

```
c.go()
	.then(addTImeout(launch, 4*1000))
	.then(function(msg){
        console.log(msg);
	})
	.catch(function(err){
        console.error("Houston, we have a problem:" + err.message)
	})
```



### 生成器

生成器可以实现代码简洁，功能异步.

1.找一个能将Node中错误优先的回调转换成promise的方法。会把它封装成一个叫做nfcall的函数(Node函数调用)

```
// 以Q promise库中的nfcall函数命名，若需要这个功能，应该使用Q这个库。
function nfcall(f, ...args){
    return new Promise(function(resolve, reject){
        f.call(null, ...args, function(err, ...args){
            if(err) return reject(err);
            resolve(args.length<2?args[0]:args);
        })
    })
}
```

2.将任何Node格式的方法转化为接受一个回调的promise。同时也需要setTimeout，可以接收一个回调。由于早于Node，不适用错误有限。这里创建ptimeout

```
function ptimeout(delay){
    return new Promise(function(resolve, reject){
        setTimeout(resolve, delay)
    })
}
```

3.生成运行器

生成器并不天生异步，但由于生成器允许函数和其调用方对话，所以可以创建一个用来管理对话的函数，同时这个函数需要知道如何处理异步调用

```
function grun(g){
    const it = g();
    (function iterate(val){
        const x = it.next(val);
        if(!x.done){
            if(x.value instanceof Promise){
                x.value.then(iterate).catch(err => it.throw(err));
            }
            else{
                setTimeout(iterate, 0, x.value)
            }
        }
    })();
}
```

4.实现

```
function* theFutureIsNow(){
    const dataA = yield nfcall(fs.readFile, 'a.txt');
    const dataB = yield nfcall(fs.readFile, 'b.txt');
    const dataC = yield nfcall(fs.readFile, 'c.txt');
    yield ptimeout(60*1000);
    yield nfcall(fs.writeFile, 'd.txt', dataA+dataB+dataC)
}
```

5.运行

```
grun(theFutureIsNow);
```

6.并行

```
function* theFutureIsNow(){
	// 会在数组中所有promise都被解决后才被调用，可能会同时执行异步代码
	// Promise.all返回的promise中会提供一个包含了所有被满足的promise值的数组，顺序跟数组中原来的promise顺序保持一致。
    const data = yield Promise.all([
        nfcall(fs.readFile, 'a.txt'),
    	nfcall(fs.readFile, 'b.txt'),
    	nfcall(fs.readFile, 'c.txt'),
    ]) 
    yield ptimeout(60*1000);
    yield nfcall(fs.writeFile, 'd.txt', dataA+dataB+dataC)
}
```

7.异常处理

```
function* theFutureIsNow(){
	let data;
	try{
        data = yield Promise.all([
        nfcall(fs.readFile, 'a.txt'),
    	nfcall(fs.readFile, 'b.txt'),
    	nfcall(fs.readFile, 'c.txt'),
    	]) 
	}
	catch(err){
		console.error("Unable to read one or more input files: " + err.message);
		throw err;
	}
    yield ptimeout(60*1000);
    try{
       yield nfcall(fs.writeFile, 'd.txt', dataA+dataB+dataC) 
    }
    catch(err){
        console.error("Unable to read one or more output file: " + err.message);
		throw err;
    }
    
}
```







