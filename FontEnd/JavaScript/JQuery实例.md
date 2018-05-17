[TOC]

# 键盘事件

```javascript
keydown() 
# keydown事件会在键盘按下时触发. 
keyup() 
# keyup事件会在按键释放时触发,也就是你按下键盘起来后的事件 
keypress() 
# keypress事件会在敲击按键时触发,我们可以理解为按下并抬起同一个按键 
```

获取键盘上对应的ASCII码

```javascript
$(document).keydown(function(event){ 
	console.log(event.keyCode); 
    alert(event.keyCode);
}); 
```

常用键对应的编码

```
0键值48..9键值57 
a键值97..z键值122
A键值65..Z键值90 
+键值43;-键值45
.键值46;退格8;
tab键值9;Enter键值13
```

回车触发按钮事件

```javascript
$("body").keydown(function(event) {
     if (event.keyCode == 13) {
         $('#submit').click();
         //阻止默认事件
         event.preventDefault();
         // 阻止冒泡事件
         //event.stopPropagation();
         // 阻止默认和冒泡
         //return false;
         }
});
$('#submit').click(function(){
    alert("test")
})
```



