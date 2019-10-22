# class类

## 概述

在ES6中，class (类)作为对象的模板被引入，可以通过 class 关键字定义类。

class 的本质是 function。

它可以看作一个语法糖，让对象原型的写法更加清晰、更像面向对象编程的语法。

## 基础用法

### 类定义

类表达式可以为匿名或命名。

```javascript
// 匿名类
let Example = class {
    constructor(a) {
        this.a = a;
    }
}
// 命名类
let Example = class Example {
    constructor(a) {
        this.a = a;
    }
}
```

### 类声明

```javascript
class Example {
    constructor(a) {
        this.a = a;
    }
}
```

注意要点：不可重复声明。

```javascript
class Example{}
class Example{}
// Uncaught SyntaxError: Identifier 'Example' has already been 
// declared
 
let Example1 = class{}
class Example{}
// Uncaught SyntaxError: Identifier 'Example' has already been 
// declared
```

- 注意要点

类定义不会被提升，这意味着，必须在访问前对类进行定义，否则就会报错。

类中方法不需要 function 关键字。

方法间不能加分号。
```javascript
new Example();  class Example {}
```
### 类主体

#### 属性

- prototype

ES6 中，prototype 仍旧存在，虽然可以直接自类中定义方法，但是其实方法还是定义在 prototype 上的。 覆盖方法 / 初始化时添加方法
```javascript
Example.prototype={  
  //methods 
}
```
添加方法
```javascript
Object.assign(Example.prototype,{  
  //methods
})
```
- 静态属性

静态属性：class 本身的属性，即直接定义在类内部的属性（ Class.propname ），不需要实例化。 ES6 中规定，Class 内部只有静态方法，没有静态属性。
```javascript
class Example { 
  // 新提案  
  static a = 2;
} 

// 目前可行写法
Example.b = 2;
```
- 公共属性

```javascript
class Example{} 
Example.prototype.a = 2;
```
- 实例属性

实例属性：定义在实例对象（ this ）上的属性。
```javascript
class Example {   
  a = 2;   
	constructor () {   
  	console.log(this.a);   
  }
}
```
- name 属性

返回跟在 class 后的类名(存在时)。
```javascript
let Example=class Exam {
    constructor(a) {
        this.a = a;
    }
}
console.log(Example.name); // Exam
 
let Example=class {
    constructor(a) {
        this.a = a;
    }
}
console.log(Example.name); // Example
```

#### 方法

- constructor 

constructor 方法是类的默认方法，创建类的实例化对象时被调用。
```javascript
class Example{
    constructor(){
      console.log('我是constructor');
    }
}
new Example(); // 我是constructor
```

返回对象

```javascript
class Test {
    constructor(){
        // 默认返回实例对象 this
    }
}
console.log(new Test() instanceof Test); // true
 
class Example {
    constructor(){
        // 指定返回对象
        return new Test();
    }
}
console.log(new Example() instanceof Example); // false
```
- 静态方法

```javascript
class Example{
    static sum(a, b) {
        console.log(a+b);
    }
}
Example.sum(1, 2); // 3
```

- 原型方法

```javascript
class Example {
    sum(a, b) {
        console.log(a + b);
    }
}
let exam = new Example();
exam.sum(1, 2); // 3
```

- 实例方法

```javascript
class Example {
    constructor() {
        this.sum = (a, b) => {
            console.log(a + b);
        }
    }
}
```

### 类的实例化

**new**

class 的实例化必须通过 new 关键字。
```javascript
class Example {}  

let exam1 = Example();  // Class constructor Example cannot be invoked without 'new'
```
**实例化对象**

共享原型对象

```javascript
class Example {
    constructor(a, b) {
        this.a = a;
        this.b = b;
        console.log('Example');
    }
    sum() {
        return this.a + this.b;
    }
}
let exam1 = new Example(2, 1);
let exam2 = new Example(3, 1);
console.log(exam1._proto_ == exam2._proto_); // true
 
exam1._proto_.sub = function() {
    return this.a - this.b;
}
console.log(exam1.sub()); // 1
console.log(exam2.sub()); // 2
```

## decorator

decorator 是一个函数，用来修改类的行为，在代码编译时产生作用。

### 类修饰

一个参数

第一个参数 target，指向类本身。
```javascript
function testable(target) {
    target.isTestable = true;
}
@testable
class Example {}
Example.isTestable; // true
```

多个参数——嵌套实现

```javascript
function testable(isTestable) {
    return function(target) {
        target.isTestable=isTestable;
    }
}
@testable(true)
class Example {}
Example.isTestable; // true
```
实例属性

上面两个例子添加的是静态属性，若要添加实例属性，在类的 prototype 上操作即可。

### 方法修饰

3个参数：target（类的原型对象）、name（修饰的属性名）、descriptor（该属性的描述对象）。

```javascript
class Example {
    @writable
    sum(a, b) {
        return a + b;
    }
}
function writable(target, name, descriptor) {
    descriptor.writable = false;
    return descriptor; // 必须返回
}
```

修饰器执行顺序

由外向内进入，由内向外执行。
```javascript
class Example {
    @logMethod(1)
    @logMthod(2)
    sum(a, b){
        return a + b;
    }
}
function logMethod(id) {
    console.log('evaluated logMethod'+id);
    return (target, name, desctiptor) => console.log('excuted         logMethod '+id);
}
// evaluated logMethod 1
// evaluated logMethod 2
// excuted logMethod 2
// excuted logMethod 1
```


## 封装与继承

### getter / setter

定义
```javascript
class Example{
    constructor(a, b) {
        this.a = a; // 实例化时调用 set 方法
        this.b = b;
    }
    get a(){
        console.log('getter');
        return this.a;
    }
    set a(a){
        console.log('setter');
        this.a = a; // 自身递归调用
    }
}
let exam = new Example(1,2); // 不断输出 setter ，最终导致 RangeError
class Example1{
    constructor(a, b) {
        this.a = a;
        this.b = b;
    }
    get a(){
        console.log('getter');
        return this._a;
    }
    set a(a){
        console.log('setter');
        this._a = a;
    }
}
let exam1 = new Example1(1,2); // 只输出 setter , 不会调用 getter 方法
console.log(exam._a); // 1, 可以直接访问
```


getter 不可单独出现

```javascript
class Example {
    constructor(a) {
        this.a = a; 
    }
    get a() {
        return this.a;
    }
}
let exam = new Example(1); // Uncaught TypeError: Cannot set property // a of #<Example> which has only a getter
```

getter 与 setter 必须同级出现

```javascript
class Father {
    constructor(){}
    get a() {
        return this._a;
    }
}
class Child extends Father {
    constructor(){
        super();
    }
    set a(a) {
        this._a = a;
    }
}
let test = new Child();
test.a = 2;
console.log(test.a); // undefined
 
class Father1 {
    constructor(){}
    // 或者都放在子类中
    get a() {
        return this._a;
    }
    set a(a) {
        this._a = a;
    }
}
class Child1 extends Father1 {
    constructor(){
        super();
    }
}
let test1 = new Child1();
test1.a = 2;
console.log(test1.a); // 2
```

### extends

通过 extends 实现类的继承。
```javascript
class Child extends Father { ... }
```
### super

子类 constructor 方法中必须有 super ，且必须出现在 this 之前。

```javascript
class Father {
    constructor() {}
}
class Child extends Father {
    constructor() {}
    // or 
    // constructor(a) {
        // this.a = a;
        // super();
    // }
}
let test = new Child(); // Uncaught ReferenceError: Must call super 
// constructor in derived class before accessing 'this' or returning 
// from derived constructor
```

调用父类构造函数,只能出现在子类的构造函数。
```javascript
class Father {
    test(){
        return 0;
    }
    static test1(){
        return 1;
    }
}
class Child extends Father {
    constructor(){
        super();
    }
}
class Child1 extends Father {
    test2() {
        super(); // Uncaught SyntaxError: 'super' keyword unexpected     
        // here
    }
}
```

调用父类方法, super 作为对象，在普通方法中，指向父类的原型对象，在静态方法中，指向父类

```javascript
class Child2 extends Father {
    constructor(){
        super();
        // 调用父类普通方法
        console.log(super.test()); // 0
    }
    static test3(){
        // 调用父类静态方法
        return super.test1+2;
    }
}
Child2.test3(); // 3
```
### 注意要点

不可继承常规对象。
```javascript
var Father = {
    // ...
}
class Child extends Father {
     // ...
}
// Uncaught TypeError: Class extends value #<Object> is not a constructor or null
 
// 解决方案
Object.setPrototypeOf(Child.prototype, Father);
```