# Promise

## 概述

是异步编程的一种解决方案。

从语法上说，Promise 是一个对象，从它可以获取异步操作的消息。

## Promise 状态

### 状态的特点

Promise 异步操作有三种状态：pending（进行中）、fulfilled（已成功）和 rejected（已失败）。除了异步操作的结果，任何其他操作都无法改变这个状态。

Promise 对象只有：从 pending 变为 fulfilled 和从 pending 变为 rejected 的状态改变。只要处于 fulfilled 和 rejected ，状态就不会再变了即 resolved（已定型）。

```javascript
const p1 = new Promise(function(resolve,reject){
    resolve('success1');
    resolve('success2');
}); 
const p2 = new Promise(function(resolve,reject){  
    resolve('success3'); 
    reject('reject');
});
p1.then(function(value){  
    console.log(value); // success1
});
p2.then(function(value){ 
    console.log(value); // success3
});
```

### 状态的缺点

无法取消 Promise ，一旦新建它就会立即执行，无法中途取消。

如果不设置回调函数，Promise 内部抛出的错误，不会反应到外部。

当处于 pending 状态时，无法得知目前进展到哪一个阶段（刚刚开始还是即将完成）。

## then 方法

then 方法接收两个函数作为参数，第一个参数是 Promise 执行成功时的回调，第二个参数是 Promise 执行失败时的回调，两个函数只会有一个被调用。

### then 方法的特点

在 JavaScript 事件队列的当前运行完成之前，回调函数永远不会被调用。

```javascript
const p = new Promise(function(resolve,reject){
  resolve('success');
});
 
p.then(function(value){
  console.log(value);
});
 
console.log('first');
// first
// success
```

通过 **.then** 形式添加的回调函数，不论什么时候，都会被调用。

通过多次调用

```javascript
.then
```

，可以添加多个回调函数，它们会按照插入顺序并且独立运行。

```javascript
const p = new Promise(function(resolve,reject){
  resolve(1);
}).then(function(value){ // 第一个then // 1
  console.log(value);
  return value * 2;
}).then(function(value){ // 第二个then // 2
  console.log(value);
}).then(function(value){ // 第三个then // undefined
  console.log(value);
  return Promise.resolve('resolve'); 
}).then(function(value){ // 第四个then // resolve
  console.log(value);
  return Promise.reject('reject'); 
}).then(function(value){ // 第五个then //reject:reject
  console.log('resolve:' + value);
}, function(err) {
  console.log('reject:' + err);
});
```

then 方法将返回一个 resolved 或 rejected 状态的 Promise 对象用于链式调用，且 Promise 对象的值就是这个返回值。

### then 方法注意点

简便的 Promise 链式编程最好保持扁平化，不要嵌套 Promise。

注意总是返回或终止 Promise 链。

```javascript
const p1 = new Promise(function(resolve,reject){
  resolve(1);
}).then(function(result) {
  p2(result).then(newResult => p3(newResult));
}).then(() => p4());
```

创建新 Promise 但忘记返回它时，对应链条被打破，导致 p4 会与 p2 和 p3 同时进行。

大多数浏览器中不能终止的 Promise 链里的 rejection，建议后面都跟上 `.catch(error => console.log(error));`