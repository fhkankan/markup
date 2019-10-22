# let与const

ES2015(ES6) 新增加了两个重要的 JavaScript 关键字: **let** 和 **const**。

let 声明的变量只在 let 命令所在的代码块内有效。

const 声明一个只读的常量，一旦声明，常量的值就不能改变。

------

## let 命令

基本用法:
```javascript
{  
  let a = 0;  
  a   // 0 
} a   // 报错 ReferenceError: a is not defined
```
**代码块内有效**

let 是在代码块内有效，var 是在全局范围内有效:
```javascript
{  
  let a = 0; 
  var b = 1; 
} 
a  // ReferenceError: a is not defined
b  // 1
```
**不能重复声明**

let 只能声明一次 var 可以声明多次:
```javascript
let a = 1; 
let a = 2;
var b = 3;
var b = 4;
a  // Identifier 'a' has already been declared 
b  // 4
```
for 循环计数器很适合用 let
```javascript
for (var i = 0; i < 10; i++) { 
  setTimeout(function(){ 
    console.log(i);
  })
} 
// 输出十个 10

for (let j = 0; j < 10; j++) { 
  setTimeout(function(){ 
    console.log(j); 
  }) 
}
// 输出 0123456789
```
变量 i 是用 var 声明的，在全局范围内有效，所以全局中只有一个变量 i, 每次循环时，setTimeout 定时器里面的 i 指的是全局变量 i ，而循环里的十个 setTimeout 是在循环结束后才执行，所以此时的 i 都是 10。

变量 j 是用 let 声明的，当前的 i 只在本轮循环中有效，每次循环的 j 其实都是一个新的变量，所以 setTimeout 定时器里面的 j 其实是不同的变量，即最后输出12345。（若每次循环的变量 j 都是重新声明的，如何知道前一个循环的值？这是因为 JavaScript 引擎内部会记住前一个循环的值）。

### 不存在变量提升

let 不存在变量提升，var 会变量提升:
```javascript
console.log(a);  //ReferenceError: a is not defined
let a = "apple";

console.log(b);  //undefined
var b = "banana";
```
变量 b 用 var 声明存在变量提升，所以当脚本开始运行的时候，b 已经存在了，但是还没有赋值，所以会输出 undefined。

变量 a 用 let 声明不存在变量提升，在声明变量 a 之前，a 不存在，所以会报错。

------

## const 命令

const 声明一个只读变量，声明之后不允许改变。意味着，一旦声明必须初始化，否则会报错。

基本用法:

```javascript
const PI = "3.1415926";
PI  // 3.1415926

const MY_AGE;  // SyntaxError: Missing initializer in const declaration    
```

暂时性死区:

```javascript
var PI = "a";
if(true){
  console.log(PI);  // ReferenceError: PI is not defined
  const PI = "3.1415926";
}
```

ES6 明确规定，代码块内如果存在 let 或者 const，代码块会对这些命令声明的变量从块的开始就形成一个封闭作用域。代码块内，在声明变量 PI 之前使用它会报错。

### 注意要点

const 如何做到变量在声明初始化之后不允许改变的？其实 const 其实保证的不是变量的值不变，而是保证变量指向的内存地址所保存的数据不允许改动。此时，你可能已经想到，简单类型和复合类型保存值的方式是不同的。是的，对于简单类型（数值 number、字符串 string 、布尔值 boolean）,值就保存在变量指向的那个内存地址，因此 const 声明的简单类型变量等同于常量。而复杂类型（对象 object，数组 array，函数 function），变量指向的内存地址其实是保存了一个指向实际数据的指针，所以 const 只能保证指针是固定的，至于指针指向的数据结构变不变就无法控制了，所以使用 const 声明复杂类型对象时要慎重。