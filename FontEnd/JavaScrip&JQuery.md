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

# JQuery

```
jquery是一个函数库，一个js文件，页面用script标签引入这个js文件就可以使用。

jQuery的版本分为1.x系列和2.x、3.x系列，1.x系列兼容低版本的浏览器，2.x、3.x系列放弃支持低版本浏览器

1、http://jquery.com/ 官方网站
2、https://code.jquery.com/ 版本下载
```

## 文档加载完再执行

```javascript
将获取元素的语句写到页面头部，会因为元素还没有加载而出错，jquery提供了ready方法解决这个问题，它的速度比原生的 window.onload 更快。

$(document).ready(function(){     ......
});
//简写                             
$(function(){
......                               
})
```

## 选择元素

### 选择器

```
jquery选择器可以快速地选择元素，选择规则和css样式相同.
$('#myId') //选择id为myId的网页元素
$('.myClass') // 选择class为myClass的元素
$('li') //选择所有的li元素
$('#ul1 li span') //选择id为为ul1元素下的所有li下的span元素
$('input[name=first]') // 选择name属性等于first的input元素
```

### 选择集过滤

```
$('div').has('p'); // 选择包含p元素的div元素
$('div').not('.myClass'); //选择class不等于myClass的div元素
$('div').eq(5); //选择第6个div元素
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

## 操作样式

```
# 操作行间样式
// 获取div的样式
$("div").css("width");
$("div").css("color");
//设置div的样式
$("div").css("width","30px");
$("div").css({fontSize:"30px",color:"red"});

注意：选择器获取的多个元素，获取信息获取的是第一个

# 操作样式类名
$("#div1").addClass("divClass2") //为id为div1的对象追加样式divClass2
$("#div1").removeClass("divClass")  //移除id为div1的对象的class名为divClass的样式
$("#div1").removeClass("divClass divClass2") //移除多个样式，中间空格
$("#div1").toggleClass("anotherClass") //重复切换anotherClass样式
```
## 属性操作

```
# html() 取出或设置html内容
// 取出html内容
var $htm = $('#div1').html();
// 设置html内容
$('#div1').html('<span>添加文字</span>');

# prop() 取出或设置元素除了css之外的某个属性的值
// 取出图片的地址
var $src = $('#img1').prop('src');
// 设置图片的地址和alt属性
$('#img1').prop({src: "test.jpg", alt: "Test Image" });
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

## 绑定事件

```
# 给元素绑定click事件，可以用如下方法：
$('#btn1').click(function(){
    // 内部的this指的是原生对象
    // 使用jquery对象用 $(this)
})

# 获取元素的索引值 
获得匹配元素相对于其同胞元素的索引位置，此时用index()

# jquery事件
blur() 				元素失去焦点
focus() 			元素获得焦点
click() 			鼠标单击
mouseover() 		鼠标进入（进入子元素也触发）
mouseout() 			鼠标离开（离开子元素也触发）
mouseenter() 		鼠标进入（进入子元素不触发）
mouseleave() 		鼠标离开（离开子元素不触发）
hover() 			同时为mouseenter和mouseleave事件指定处理函数
ready() 			DOM加载完成
submit() 			用户递交表单
```

## 动画与特殊效果

```
通过animate方法可以设置元素某属性值上的动画，可以设置一个或多个属性值，动画执行完成后会执行一个函数。
$('#div1').animate({
    width:300,
    height:300
},1000,'swing',function(){
    alert('done!');
});
animate参数：
参数一：要改变的样式属性值，写成字典的形式
参数二：动画持续的时间，默认400，单位为毫秒，一般不写单位
参数三：动画曲线，默认为‘swing’，缓冲运动，还可设置‘linear’，匀速运动
参数四：动画回调函数，动画完成后执行的匿名函数

# 特殊效果是对常用的动画进行了函数的封装，参数取animate的后三个
fadeIn() 淡入
fadeOut() 淡出
fadeToggle() 切换淡入淡出
hide() 隐藏元素
show() 显示元素
toggle() 切换元素的可见状态
slideDown() 向下展开
slideUp() 向上卷起
slideToggle() 依次展开或卷起某个元素
```

## 循环

```
对jquery选择的对象集合分别进行操作，需要用到jquery循环操作，此时可以用对象上的each方法：
$(function(){
    $('.list li').each(function(){
        $(this).html($this.index());
    })
})
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
    $ali.click(function() {        		$(this).css({background:'red'});
    });
})		
# 委托写法
$(function(){
    $list = $('#list');
    $list.delegate('li', 'click', function() {        				$(this).css({background:'red'});
    });
})
```

## 节点操作

```
元素节点操作指的是改变html的标签结构，它有两种情况：
1、移动现有标签的位置
2、将新创建的标签插入到现有的标签中

创建新标签
var $div = $('<div>'); //创建一个空的div
var $div2 = $('<div>这是一个div元素</div>');

移动或者插入标签的
父元素.append(子元素)：当前元素的内部后面放入另外一个元素
子元素.appendTo(父元素)：当前元素放置到另一元素的内部的后面
父元素.prepend(子元素):当前元素的内部的前面放入另外一个元素
子元素.prepend(父元素)：当前元素放置到另一元素的内部的前面
元素.after(元素)：当前元素的后面放入另一个元素
元素.insertafter(元素)：当前元素放置到另一元素的后面
元素.before(元素)：当前元素的前面放入另一个元素
元素.insertbefore(元素)：当前元素放置到另一元素的前面

删除元素
元素.remove()
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

AJAX = 异步 JavaScript 和 XML（Asynchronous JavaScript and XML）
一种发送http请求与后台进行异步通讯的技术
在不重载整个网页的情况下，AJAX从后台加载数据，并在网页上进行局部刷新

```
$.ajax方法使用：
$.ajax({
    url:'/js/data.json',
    type:'POST', 
    dataType: json,
    data:{name:'wang',age:25},
    async: true
})
.success(function(data){
     alert(data)
})
.fail(function(){
    alert("出错")
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

封装方法：

- load

```javascript
# 从服务器加载数据，并把返回的数据放入被选元素中。
$(selector).load(URL,data,callback)
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
$.get(URL,callback)
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







