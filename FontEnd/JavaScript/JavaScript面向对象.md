[TOC]

# JavaScript面向对象

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

### this关键字

在一个函数中，this指向调用该函数的对象；在一个带有属性和方法的典型对象中，若函数使用了this，则this指向包含该函数的对象

> 绑定的一般方式

由方法如何被调用所决定的，而非函数定义所决定

```js
const o = {
    name: "LiLei",
    speak(){
        return 'My name is ${this.name}!';
    },
}
o.speak();//"My name is LiLei!"

// 改变调用
const speak = o.speak;
speak();//"My name is !",this绑定到了undefied

```

在嵌套函数中使用经常出错。

```javascript
const o = {
	name: "LiLei",
	speak:function(){
       function hello(){
           return this.name
       };
       return hello()
	},
};
o.speak(); //""

//解决方法是给this赋另一个变量
const o = {
	name: "LiLei",
	speak:function(){
       const self = this;
       function hello(){
           return self.name
       };
       return hello()
	},
};
o.speak(); //LiLei

//ES6中使用箭头函数也可以解决
const o = {
	name: "LiLei",
	speak:function(){
       const hello = () => {
           return this.name
       };
       return hello()
	},
};
o.speak(); //LiLei
```

> 指定绑定值

```
const bruce = {name: "Bruce"}
const alice = {name: "Alice"}
function update(birthYear){
    this.birthYear = birthYear
}
```

call

```
update.call(bruce, 1949)//bruce是{name: "Bruce", birthYear: 1949}
update.call(alice, 1969)//alice是{name: "Alice", birthYear: 1969}
```

apply

适用于将数组作为参数

```
update.apply(bruce,[1949])//bruce是{name: "Bruce", birthYear: 1949}
update.apply(alice,[1969])//alice是{name: "Alice", birthYear: 1969}
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

## 常用内部对象

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

> 遍历

forEach() 

方法用于调用数组的每个元素，并将元素传递给回调函数。

**注意:** forEach() 对于空数组是不会执行回调函数的。

```
array.forEach(function(currentValue, index, arr), thisValue)
```
参数

| 参数                                 | 描述                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| *function(currentValue, index, arr)* | 必需。 数组中每个元素需要调用的函数。 函数参数:参数描述*currentValue*必需。当前元素*index*可选。当前元素的索引值。*arr*可选。当前元素所属的数组对象。 |
| *thisValue*                          | 可选。传递给函数的值一般用 "this" 值。 如果这个参数为空， "undefined" 会传递给 "this" 值 |

for

```
for(index in array){
    console.log(index);
    console.log(array[index])
}
```

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

|        | 方法             | 说明                                                         | 语法                                          |
| ------ | ---------------- | ------------------------------------------------------------ | --------------------------------------------- |
| 增加   | concat()         | 连接两个或更多的数组，并返回结果                             | arrayObject.concat(arrayX,...)                |
|        | push()           | 向数组的末尾添加一个或多个元素，并返回新的长度               | arrayObject.push(newelement,...)              |
|        | unshift()        | 向数组的开头添加一个或多个元素，并返回新的长度               | arrayObject.unshift(newelement,...)           |
| 删除   | pop()            | 删除并返回数组的最后一个元素                                 | arrayObject.pop()                             |
|        | shift()          | 删除并返回数组的第一个元素                                   | arrayObject.shift()                           |
| 删增   | splice()         | 在任意位置删除元素，并向数组添加新元素                       | arrayObject.splice(start,length,element1,...) |
| 排序   | reverse()        | 颠倒数组中元素的顺序                                         | arrayObject.reverse()                         |
|        | sort()           | 对数组元素进行排序                                           | arrayObject.sort(sortby)                      |
| 子元素 | slice()          | 从某个已有的数组返回选定的元素                               | arrayObject.slice(start, end)                 |
| 转换   | toSource()       | 代表对象的源代码                                             |                                               |
|        | toString()       | 把数组转换为字符串，并返回结果                               | arrayObject.toString()                        |
|        | toLocaleString() | 把数组转换为本地字符串，并返回结果                           | arrayObject.toLocaleString()                  |
|        | join()           | 把数组的所有元素放入一个字符串，元素通过指定的分隔符进行分隔，返回数组的拷贝 | arrayObject.join(separator)                   |
| 搜索   | valueOf()        | 返回数组对象的原始值                                         |                                               |
|        | indexOf()        | 返回数组中第一次出现值得索引，若无则返回-1                   | arrayObject.indexOf(值,[startIndex])          |

ES6新增内容

|          | 方法             | 说明                                                         | 语法                                                         |
| -------- | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 分割替换 | copyWith()       | 对数组内分割和替换                                           | arrayObject.copyWith<br>(targetIndex,startCopyIndex,endCopyIndex) |
| 填充     | fill()           | 指定值填充                                                   | arrayObject.fill(element,startIndex,endIndex)                |
| 查找     | lastIndexOf()    | 从数组末尾开始查找，返回完全匹配的第一个元素的下标           | arrayObject.lastIndexOf(值,[startIndex])                     |
|          | findIndex()      | 从数组开始，可以查找数组元素是否符合函数,返回下标，无则返回-1 | const arr=[{id:5,name:"Lilei"},{id:7,name:"Alice"}]<br>arr.findIndex(o=>o.id==5) |
|          | findeLastIndex() | 从数组末尾开始，可以查找数组元素是否符合函数，返回下标,无则返回-1 | arr.findeLastIndex(o=>o.id==7)                               |
|          | find()           | 从数组开始，可以查找数组元素是否符合函数，返回数组元素,无则返回null | arr.find(o=>o.id==6)                                         |
|          | some()           | 元素是否存在，存在则true,否则false                           | const arr=[5,6,7]<br>arr.some(x=>x%2==0)                     |
|          | every()          | 所有元素都符合条件，true,否则false                           | arr.every(x=>x%2!=0)                                         |
| 转化     | map()            | 转换数组中的所有元素，返回数组的拷贝                         | const items=["a","b"]<br>const numbers=[1,2]<br>const arr=items.map((x,i)=>({name:x,price:numbers[i]})) |
|          | filter()         | 根据给定条件查找数组元素，返回数组的拷贝                     | cosnt arr=[1,2,3,4]<br>const items=arr.filter(x=>x%2==0)     |
|          | reduce()         | 把整个数组转化为另一种数组类型，返回数组的拷贝               | const arr=[5,6,7]<br>const sum = arr.reduce((a,x)=>a+=x,0)<br>//a的初始值指定为0,可缺省，默认为0 |

### Array2

字典

- 定义

```
var dict = new Array()
```

- 设定值

```
dict['q'] = "q1"  # 若存在则修改，若不存在则创建
```

- 遍历

```
for(var key in dict){
    cosole.log(key + ":" + dict[key])
}
```

- 删除

```
delete dict['r]
delete dict.r
```

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
### JSON

json对象

````
{
    a:1,
    b:2
}
````

json对象属性

```
# 判断是否有xxx这个key
jsonObj.hasOwnProperty('xxx')
```

将json对象转换成json字符串

```
JSON.stringify(object [, replacer] [, space])

// 参数：
object  必须；通常为对象或数组。
replacer 可选，用于转换结果的函数或者数组。
space 可选。向返回值 JSON 文本添加缩进、空格和换行符以使其更易于读取。 
//返回值：
一个包含JSON文本的字符串。
```

将json字符串转换成json对象

```
JSON. parse(text[,reviver])

// 参数
text  必须；一个有效的json字符串
reviver  可选
// 返回值
一个对象或数组
```

`eval()`

只能在非严格模式中进行使用，在use strict中是不允许使用的

```
eval(string)

// 参数
string  必须，需要计算的字符串，其中含有要计算的javascript表达式或要执行的语句。
// 返回值
如果s不是字符串，则直接返回s。否则执行s语句。如果s语句执行结果是一个值，则返回此值，否则返回undefined。 需要特别注意的是对象声明语法“{}”并不能返回一个值，需要用括号括起来才会返回值


// json字符串转换为对象
textObjString = '{a:1,b:2}'
textObj = eval('('+ textObjString +')')
```

### Maps

ES6之前，若要把键和值映射起来，需要用对象，但存在如下问题

```
对象原型中可能存在不需要的映射
不清楚对象中有多少映射
由于键必须是字符串或符号，无法把对象映射到值
对象不能保证自身属性的顺序
```

ES6之后，Maps对象接解决了这些问题

```
// 将user对象映射到role
const u1 = {name:"a"};
const u2 = {name:"b"};
// 映射方法一
const userRoles = new Map();
userRoles.set(u1, "User");
userRoles.set(u2, "Admin")
// 映射方法二
const userRoles = new Map(
	[
        [u1, "User"],
        [u2, "Admin"],
	]
)
```

使用

```
//设定角色
userRoles.set(u1,"Admin")
// 获取角色
userRoles.get(u1)
// 判断是否包含key
userRoles.has(u4)
// map中元素个数
userRoles.size;
// 删除某个
userRoles.delete(u1);
// 删除所有
userRoles.clear()
// 遍历
userRoles.keys() //所有键
userRoles.values() //所有值
userRoles.entries()//以数组形式或额所有键值对
```

### Weak maps

与maps本质上相同,除了

```
key必须是对象
WeakMap中的key可以被垃圾回收
WeakMap不能迭代或者清空
```

常用来存放私有key

```
// 创建
const SecretHolder = (fucntion(){
    const secrets = new WeakMap();
    return class {
    	// 赋值
        setSecret(secret){
            secrets.set(this, secret);
        }
        // 取值
        getSecret(){
            return secrets.get(this);
        }
    }
})();

// 使用
cosnt a = new SecretHolder();
cosnt b = new SecretHolder();
a.setSecret('secret A')
b.setSecret('secret B')
a.getSecret();
b.getSecret();
```

### Sets

ES6之后增加了集合,存放不重复的数据

```
// 创建
cosnt roles = new Set();
// 添加，当重复时，不操作
roles.add("User")
// 个数
roles.size;
// 删除某个值
roles.delete("Admin")
```

## 类

### 创建类

ES5创建类

```
// 构造函数
function Car(make, model){
    this.make = make;
    this.model = model;
    this._userGears = ['P','N','R','D'];
    this._userGear = this._userGears[0];
}
```

ES6引入创建类的语法

```
// 创建类
class Car{
	// 构造器
    constructor(make, model){
		this.make = make;//车牌号
		this.model = model;//型号
		this.userGears = ['P','N','R','D'];//档位
		this.userGear = this.userGears[0];//当前档位
	}
	// 方法
	shift(gear){
        if(this.userGears.isindexOf(gear)<0)
        	throw new Error('Invalid gear:${gear}');
        this.userGear = gear;
	}
}

// 创建类的实例
const car1 = new Car("Tesla", "Model S");
const car2 = new Car("Mazda", "3i");
// 调用方法
car1.shift('D');//this跟car1绑定
car1.shift('R');//this跟car2绑定
// 调用属性
car1.userGear
car2.userGear
```

**注意**

ES5和ES6对类的实现底层相同，都是需要创建一个函数充当类的构造方法。

```
class Es6Car()
fucntion ES5Car()
>typeOf Es6Car //function
>typeOf Es5Car //function
```

### 动态属性

Car中的shift函数能防止选择一个无效档位。但是可以直接赋值car1.userGear ="X"

动态属性可以具有属性的语义，但同时可以向方法一样被调用

```
class Car{
	// 构造器
    constructor(make, model){
		this.make = make;//车牌号
		this.model = model;//型号
		this._userGears = ['P','N','R','D'];//档位
		this._userGear = this._userGears[0];//当前档位
	}
	get userGear(){ return this._userGear; }
	set userGear(value){
        if(this._userGears.indexOf(value) < 0){
        	throw new Error('Invalid gear: $(value));
			};
		this._userGear = value;
	}
	// 方法
	shift(gear){ this.userGear = gear; }
}
```

以上只是采用了约定俗成的做法，告诉哪些代码访问了被保护的属性，并没有屏蔽直接car1._userGear = "X"

若要强制私有化，则使用WeakMap实例

```
const Car = (
	function(){
		// 使用即时调用函数表达式将WeakMap()隐藏至闭包中，阻止外界访问。
		// 这个WeakMap可以安全地存储任何不想被Car类外部访问的属性
        const carProps = new WeakMap();
        class Car{
			// 构造器
    		constructor(make, model){
				this.make = make;//车牌号
				this.model = model;//型号
				this._userGears = ['P','N','R','D'];//档位
				carProps.set(this, {userGear:this._userGears[0]});//当前档位
			}
			get userGear(){ return carProps.get(this).userGear;}
			set userGear(value){
        		if(this._userGears.indexOf(value) < 0){
        			throw new Error('Invalid gear: $(value));
				};
				carProps.get(this).userGear = value;
				}
			// 方法
			shift(gear){ this.userGear = gear; }
		}
	return Car;
	}
)();
```

### 原型

在类的实例中，当引用一个方法时，实际上是在引用原型方法

```
使用#描述原型方法。Car.prototype.shift 可表示为Car#shift
```

每个函数都有一个叫做prototype的特殊属性。一般的函数不需要使用原型。

当使用关键字new创建一个新的实例时，新创建的对象可以访问其构造器的原型对象。对象实例会将它存储在自己的`__proto__`属性中

当试图访问对象的某个属性或方法时，若它不存在于当前对象中，js会检查它是否在对象原型中。因为同一个类的所有实例共用同一个原型，若原型中存在某个属性或方法，则该类的所有实例都可以访问这个属性或方法。

在实例中定义的方法或属性会覆盖掉原型中的定义。JS的检查顺序是先实例后原型，若原型中有，则不再检查原型

### 静态方法

实例方法只针对每个具体的实例才有用。

静态方法(类方法)，不与实例绑定。在静态方法中，this绑定类本身，但通常使用类名代替this

惊天方法通常用来执行一些与类相关的任务，而非具体的实例相关

```
calss Car{
	// 静态方法
    static getNextVin(){
        return Car.nextVin++;// 也可this.nextVin++
    }
    // 构造器
    constructor(make, model){
        this.make = make;
        this.model = model;
        this.vin = Car.getNextVin();//车辆识别码
    }
    static areSimilar(car1, car2){
        return car1.make===car2.make && car1.model===car2.model;
    }
    static areSame(car1, car2){
        return car1.vin===car2.vin;
    }
}
Car.nextVin = 0;

cosnt car1 = new Car("Tesla", "S");
cosnt car2 = new Car("Mazda", "3");
cosnt car3 = new Car("Mazda", "3");

car1.vin; //0
car2.vin; //1
car3.vin; //2

Car.areSimilar(car1, car2); // false
Car.areSimilar(car2, car3); // true
Car.areSame(car2, car3); // false
Car.areSame(car2, car2); // true
```

### 对象属性

- 枚举

```
class Super{
    constructot(){
        this.name = 'Super';
        this.isSuper = true;
    }
}
// 合法，但不建议
Super.prototype.sneaky = 'not recomended!'
class Sub extends Super{
    constructor(){
        super();
        this.name = 'Sub';
        this.isSub = true;
    }
}
const obj = new Sub();

for(let p in obj){
    console.log('${p}:${obj[p]}'+
    	(obj.hasOwnProperty(p) ? '' : '(inherited)')
    );
}
```

运行之后

```
name:Sub
isSuper:true
isSub:true
sneaky: not recomended! (inherited)
```

name, isSuper,isSub都被定义在实例中，而不在原型链中。sneaky被手动添加到父类的原型中

Objects.key只包含了原型中定义的属性

- 字符串表示

由于每个对象都继承于Object,有共用方法toString()，默认返回`[object][object]`

调试时，添加一个英语返回对象的描述信息的toString()很有用

```
class Car{
    toString(){
        return '${this.make} ${this.model} ${this.vin}';
    }
}
```

## 继承

ES5的继承，实质是先创造子类的实例对象`this`，然后再将父类的方法添加到`this`上面（`Parent.apply(this)`）。

ES6的继承机制完全不同，实质是先创造父类的实例对象`this`（所以必须先调用`super`方法），然后再用子类的构造函数修改`this`

父类

```
// 定义一个动物类
function Animal (name) {
  // 属性
  this.name = name || 'Animal';
  // 实例方法
  this.sleep = function(){
    console.log(this.name + '正在睡觉！');
  }
}
// 原型方法
Animal.prototype.eat = function(food) {
  console.log(this.name + '正在吃：' + food);
};
```

### 原型链继承

**核心：** 将父类的实例作为子类的原型

```
function Cat(){ 
}
Cat.prototype = new Animal();
Cat.prototype.name = 'cat';

//　Test Code
var cat = new Cat();
console.log(cat.name);
console.log(cat.eat('fish'));
console.log(cat.sleep());
console.log(cat instanceof Animal); //true 
console.log(cat instanceof Cat); //true
```

优缺点

```
特点：
非常纯粹的继承关系，实例是子类的实例，也是父类的实例
父类新增原型方法/原型属性，子类都能访问到
简单，易于实现

缺点：
要想为子类新增属性和方法，必须要在new Animal()这样的语句之后执行，不能放到构造器中
无法实现多继承
来自原型对象的引用属性是所有实例共享的
创建子类实例时，无法向父类构造函数传参

推荐指数：★★（3、4两大致命缺陷）
```

### 构造继承

**核心：**使用父类的构造函数来增强子类实例，等于是复制父类的实例属性给子类（没用到原型）

 ```
function Cat(name){
  Animal.call(this);
  this.name = name || 'Tom';
}

// Test Code
var cat = new Cat();
console.log(cat.name);
console.log(cat.sleep());
console.log(cat instanceof Animal); // false
console.log(cat instanceof Cat); // true
 ```

优缺点

```
特点：
解决了1中，子类实例共享父类引用属性的问题
创建子类实例时，可以向父类传递参数
可以实现多继承（call多个父类对象）

缺点：
实例并不是父类的实例，只是子类的实例
只能继承父类的实例属性和方法，不能继承原型属性/方法
无法实现函数复用，每个子类都有父类实例函数的副本，影响性能

推荐指数：★★（缺点3）
```

### 实例继承

**核心：**为父类实例添加新特性，作为子类实例返回

```
function Cat(name){
  var instance = new Animal();
  instance.name = name || 'Tom';
  return instance;
}

// Test Code
var cat = new Cat();
console.log(cat.name);
console.log(cat.sleep());
console.log(cat instanceof Animal); // true
console.log(cat instanceof Cat); // false
```

优缺点

```
特点：
不限制调用方式，不管是new 子类()还是子类(),返回的对象具有相同的效果

缺点：
实例是父类的实例，不是子类的实例
不支持多继承

推荐指数：★★
```

### 拷贝继承

```
function Cat(name){
  var animal = new Animal();
  for(var p in animal){
    Cat.prototype[p] = animal[p];
  }
  Cat.prototype.name = name || 'Tom';
}

// Test Code
var cat = new Cat();
console.log(cat.name);
console.log(cat.sleep());
console.log(cat instanceof Animal); // false
console.log(cat instanceof Cat); // true
```

优缺点

```
特点：
支持多继承

缺点：
效率较低，内存占用高（因为要拷贝父类的属性）
无法获取父类不可枚举的方法（不可枚举方法，不能使用for in 访问到）

推荐指数：★（缺点1)
```

### 组合继承

**核心：**通过调用父类构造，继承父类的属性并保留传参的优点，然后通过将父类实例作为子类原型，实现函数复用

```
function Cat(name){
  Animal.call(this);
  this.name = name || 'Tom';
}
Cat.prototype = new Animal();

// 组合继承也是需要修复构造函数指向的。
Cat.prototype.constructor = Cat;

// Test Code
var cat = new Cat();
console.log(cat.name);
console.log(cat.sleep());
console.log(cat instanceof Animal); // true
console.log(cat instanceof Cat); // true
```

优缺点

```
特点：
弥补了方式2的缺陷，可以继承实例属性/方法，也可以继承原型属性/方法
既是子类的实例，也是父类的实例
不存在引用属性共享问题
可传参
函数可复用

缺点：
调用了两次父类构造函数，生成了两份实例（子类实例将子类原型上的那份屏蔽了）

推荐指数：★★★★（仅仅多消耗了一点内存)
```

### 寄生组合继承

**核心：**通过寄生方式，砍掉父类的实例属性，这样，在调用两次父类的构造的时候，就不会初始化两次实例方法/属性，避免的组合继承的缺点

```
function Cat(name){
  Animal.call(this);
  this.name = name || 'Tom';
}
(function(){
  // 创建一个没有实例方法的类
  var Super = function(){};
  Super.prototype = Animal.prototype;
  //将实例作为子类的原型
  Cat.prototype = new Super();
})();

Cat.prototype.constructor = Cat; // 需要修复下构造函数

// Test Code
var cat = new Cat();
console.log(cat.name);
console.log(cat.sleep());
console.log(cat instanceof Animal); // true
console.log(cat instanceof Cat); //true
```

优缺点

```
特点：
堪称完美

缺点：
实现较为复杂

推荐指数：★★★★（实现复杂，扣掉一颗星）
```

### 实际使用

> ES5


- 原型

`javaScript`通过一种被称为原型继承的方法提供对继承的支持。这意味着一个原型可以拥有`prototype`属性，也可以拥有一个原型。称为原型链

```
//这个函数可以理解为克隆一个对象
function clone(object) {
    function F() {}
    F.prototype = object;
    return new F();
}

var Person = {
    name: 'Default Name',
    getName: function() {
        return this.name;
    }
}

//接下来让Author变为Person的克隆体
var Author = clone(Person);
Author.books = [];
Author.getBooks = function() {
    return this.books.join(', ');
}

//增加一个作者Smith
var Smith = clone(Author);
Smith.name = 'Smith';
Smith.books = [];
Smith.books.push('<<Book A>>', '<<Book B>>'); //作者写了两本书
console.log(Smith.getName(), Smith.getBooks()); //Smith <<Book A>>, <<Book B>>

//再增加一个作者Jacky
var Jacky = clone(Author);
Jacky.name = 'Jacky';
Jacky.books = [];
Jacky.books.push('<<Book C>>', '<<Book D>>');
console.log(Jacky.getName(), Jacky.getBooks()); // Jacky <<Book C>>, <<Book D>>
```

- 类构造继承

```
//定义类的构造函数
function Person(name) {
    this.name = name || '默认姓名';
}
//定义该类所有实例的公共方法
Person.prototype.getName = function() {
    return this.name;
}
function Author(name, books) {
    //继承父类构造函数中定义的属性
    //通过改变父类构造函数的执行上下文来继承
    Person.call(this, name);
    this.books = books;
}

//继承父类对应的方法
function inherit(subClass, superClass){
    function F() {}  
    F.prototype = superClass.prototype;  
    subClass.prototype = new F();  
    subClass.prototype.constructor = subClass.constructor; //修正修改原型链时造成的constructor丢失
}
inherit(Author, Person)

Author.prototype.getBooks = function() {
    return this.books;
};

//测试
var smith = new Person('Smith');
var jacky = new Author('Jacky', ['BookA', 'BookB']);

console.log(smith.getName()); //Smith
console.log(jacky.getName()); //Jacky
console.log(jacky.getBooks().join(', ')); //BookA, BookB
console.log(smith.getBooks().join(', ')); //Uncaught TypeError: smith.getBooks is not a function
```

> ES6

- 使用类关键字

对于ES5来说，原生构造函数无法继承，由于建立子类实例对象this后，无法获得父类实例对象，故无法继承。

但是ES6是先创建父类实例对象this，再用子类的构造函数修饰this，使得父类的所有行为都可以继承

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

// extends 表示左边继承右边
class SpecialItem extends Item{
    constructor(name, color, count){
    	// 调用父类构造器
        super(color, count);
        this.name = name;
        this.describe = function(){
            console.log(this.name + ": color=" + this.color);
        }
    }
}

```

## 多态

一个实例不仅仅是它自身类的实例，也是它的任何父类的实例。在JS中，所编写的代码是“鸭子”类型。

js中的所有对象都是基类Object的实例。

```
// 使用instanceof运算符判断对象是否属于某个给定类。
```



