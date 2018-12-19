# JS移动客户端

## Touch事件

- 概述

移动端触屏滑动的效果其实就是图片轮播，在PC的页面上很好实现，绑定click和mouseover等事件来完成。但是在移动设备上，要实现这种轮播的效果，就需要用到核心的touch事件。处理touch事件能跟踪到屏幕滑动的每根手指。

touch事件

```javascript
touchstart  //当手指触摸屏幕时候触发，即使已经有一个手指放在屏幕上也会触发
touchmove   //当手指在屏幕上滑动的时候连续地触发。在这个事件发生期间，调用preventDefault()事件可以阻止滚动。
touchend    //手指离开屏幕时触发
touchcancel //可由系统进行的触发，比如手指触摸屏幕的时候，突然alert了一下，或者系统中其他打断了touch的行为，则可以触发该事件
```

上面的这些事件都会冒泡，也都可以取消(stopPropagation())。虽然这些触摸事件没有在DOM规范中定义，但是它们却是以兼容DOM的方式实现的。所以，每个触摸事件的event对象都提供了在鼠标实践中常见的属性：

```javascript
bubbles		//起泡事件的类型
cancelable	//是否用 preventDefault()方法可以取消与事件关联的默认动作
clientX		//返回当事件被触发时，鼠标指针的水平坐标
clientY		//返回当事件触发时，鼠标指针的垂直坐标
screenX		//当某个事件被触发时，鼠标指针的水平坐标
screenY		//返回当某个事件被触发时，鼠标指针的垂直坐标
```

还包含下面三个用于跟踪触摸的属性

```javascript
touches     //表示当前跟踪的触摸操作的touch对象的数组(当前屏幕上所有手指的列表)

targetTouches  //特定于事件目标的Touch对象的数组(当前dom元素上手指的列表)，尽量使用这个代替touches

changedTouches //自上次触摸以来发生了什么改变的Touch对象的数组(涉及当前事件的手指的列表)，尽量使用这个代替touches
```

每个Touch对象包含的属性如下

```javascript
clientX 	// 触摸目标在视口中的x坐标
clientY   	//触摸目标在视口中的y坐标
identifier	//标识触摸的唯一ID
pageX  		//触摸目标在页面中的x坐标
pageY:      //触摸目标在页面中的y坐标
screenX  	//触摸目标在屏幕中的x坐标 
screenY: 	//触摸目标在屏幕中的y坐标
target:     //当前的DOM元素
```

- 执行先后顺序

```
Touchstart > toucheend > mousemove > mousedown > mouseup > click
```

很多情况下触摸事件跟鼠标事件会同时触发（目的是为了让没有对触摸设备优化的代码仍然可以在触摸设备上正常工作），如果使用了触摸事件，可以调用event.preventDefault()来阻止鼠标事件被触发。而手指在屏幕上移动touchmove则不会触发鼠标事件和单击事件,在touchmove事件中加入preventDefault, 可以禁止浏览器滚动屏幕，也不会影响单击事件的触发。

- 注意

手指在滑动整个屏幕时，会影响浏览器的行为，比如滚动和缩放。所以在调用touch事件时，要注意禁止缩放和滚动。

> 禁止缩放

通过meta元标签来设置。

```html
<meta name="viewport" content="target-densitydpi=320,width=640,user-scalable=no">
```

> 禁止鼠标

如果使用了触摸事件，可以阻止鼠标事件被触发

```
event.preventDefault()
```
> 禁止滚动

手指在屏幕上移动touchmove则不会触发鼠标事件和单击事件，可以禁止浏览器滚动屏幕

```
event.preventDefault();
```

> 细心渲染

如果你正在编写的多点触控应用涉及了复杂的多指手势的话，要小心地考虑如何响应触摸事件，因为一次要处理这么多的事情。考虑一下前面一节中的在屏幕上画出所有触点的例子，你可以在有触摸输入的时候就立刻进行绘制：

```javascript
canvas.addEventListener('touchmove', function(event) {
   renderTouches(event.touches);
  },
```

不过这一技术并不是要随着屏幕上的手指个数的增多而扩充，替代做法是，可以跟踪所有的手指，然后在一个循环中做渲染，这样可获得更好的性能

```javascript
 var touches = []
  canvas.addEventListener('touchmove', function(event) {
    touches = event.touches;
  }, false);
  // 设置一个每秒60帧的定时器
  timer = setInterval(function() {
   renderTouches(touches);
  }, 15);
```

提示：setInterval不太适合于动画，因为它没有考虑到浏览器自己的渲染循环。现代的桌面浏览器提供了requestAnimationFrame这一函数，基于性能和电池工作时间原因，这是一个更好的选择。一但浏览器提供了对该函数的支持，那将是首选的处理事情的方式

- 案例

```javascript
var slider = {
	//判断设备是否支持touch事件
	touch:('ontouchstart' in window) || window.DocumentTouch && document instanceof DocumentTouch,
	slider:document.getElementById('slider'),
	//事件
    events:{
        index:0,     //显示元素的索引
        slider:this.slider,     //this为slider对象
        icons:document.getElementById('icons'),
        icon:this.icons.getElementsByTagName('span'),
        handleEvent:function(event){
            var self = this;     //this指events对象
            if(event.type == 'touchstart'){
                self.start(event);
            }else if(event.type == 'touchmove'){
                self.move(event);
            }else if(event.type == 'touchend'){
                self.end(event);
            }
        },
```

> 定义touchstart的事件处理函数，并绑定事件

触发touchstart事件后，会产生一个event对象，event对象里包括触摸列表，获得屏幕上的第一个touch,并记下其pageX,pageY的坐标。定义一个变量标记滚动的方向。此时绑定touchmove,touchend事件。

```javascript
// if(!!self.touch) self.slider.addEventListener('touchstart',self.events,false); 

//定义touchstart的事件处理函数
start:function(event){
　　var touch = event.targetTouches[0]; //touches数组对象获得屏幕上所有的touch，取第一个touch
　　startPos = {x:touch.pageX,y:touch.pageY,time:+new Date}; //取第一个touch的坐标值
　　isScrolling = 0; //这个参数判断是垂直滚动还是水平滚动
　　this.slider.addEventListener('touchmove',this,false); //绑定touchmove事件
　　this.slider.addEventListener('touchend',this,false); //绑定touchend事件
},
```

> 定义手指在屏幕上移动的事件，定义touchmove函数

同样首先阻止页面的滚屏行为，touchmove触发后，会生成一个event对象，在event对象中获取touches触屏列表，取得第一个touch,并记下pageX,pageY的坐标，算出差值，得出手指滑动的偏移量，使当前DOM元素滑动。

```javascript
//移动
move:function(event){
　　//当屏幕有多个touch或者页面被缩放过，就不执行move操作
　　if(event.targetTouches.length > 1 || event.scale && event.scale !== 1) return;
　　var touch = event.targetTouches[0];
　　endPos = {x:touch.pageX - startPos.x,y:touch.pageY - startPos.y};
　　isScrolling = Math.abs(endPos.x) < Math.abs(endPos.y) ? 1:0; //isScrolling为1时，表示纵向滑动，0为横向滑动
　　if(isScrolling === 0){
　　　　event.preventDefault(); //阻止触摸事件的默认行为，即阻止滚屏
　　　　this.slider.className = 'cnt';
　　　　this.slider.style.left = -this.index*600 + endPos.x + 'px';
　　}
},
```

> 定义手指从屏幕上拿起的事件，定义touchend函数

手指离开屏幕后，所执行的函数。这里先判断手指停留屏幕上的时间，如果时间太短，则不执行该函数。再判断手指是左滑动还是右滑动，分别执行不同的操作。最后很重要的一点是移除touchmove,touchend绑定事件。 

```javascript
//滑动释放
end:function(event){
　　var duration = +new Date - startPos.time; //滑动的持续时间
　　if(isScrolling === 0){ //当为水平滚动时
　　　　this.icon[this.index].className = '';
　　　　if(Number(duration) > 10){ 
　　　　　　//判断是左移还是右移，当偏移量大于10时执行
　　　　　　if(endPos.x > 10){
　　　　　　　　if(this.index !== 0) this.index -= 1;
　　　　　　}else if(endPos.x < -10){
　　　　　　　　if(this.index !== this.icon.length-1) this.index += 1;
　　　　　　}
　　　　}
　　　　this.icon[this.index].className = 'curr';
　　　　this.slider.className = 'cnt f-anim';
　　　　this.slider.style.left = -this.index*600 + 'px';
　　}
　　//解绑事件
　　this.slider.removeEventListener('touchmove',this,false);
　　this.slider.removeEventListener('touchend',this,false);
},
```

初始化

```javascript
init:function(){
        var self = this;     //this指slider对象
        if(!!self.touch) self.slider.addEventListener('touchstart',self.events,false);    //addEventListener第二个参数可以传一个对象，会调用该对象的handleEvent属性
    }
};

slider.init();
```

- 第三方插件

> JQuery

```javascript
$('#webchat_scroller').off('touchstart').on('touchstart',function(e) {
    var touch = e.originalEvent.targetTouches[0]; 
    var y = touch.pageY;
});
            
$('#webchat_scroller').off('touchmove').on('touchmove',function(e) {
    e.preventDefault(); // 防止浏览器默认行为
    e.stopPropagation(); // 防止冒泡
    if(e.originalEvent.targetTouches.length > 1) return; // 防止多点触控
    var touch = e.originalEvent.targetTouches[0]; 
    var y = touch.pageY;
});

$('#webchat_scroller').off('touchend').on('touchend',function(e) {
    var touch = e.originalEvent.changedTouches[0]; 
    var y = touch.pageY;
});
```

> Zepto

touch 模块绑定事件 touchstart, touchmove, touchend 到 document上，然后通过计算事件触发的时间差，位置差来实现自定义的tap，swipe事件

```javascript
$(document)
     .on('touchstart ...',function(e){
              ...
             ...
              now = Date.now()
             delta = now - (touch.last || now)
              if (delta > 0 && delta <= 250) touch.isDoubleTap = true
              touch.last = now
     })
     .on('touchmove ...', function(e){
     })
     .on('touchend ...', function(e){
            ...
            if (deltaX < 30 && deltaY < 30) {
                   var event = $.Event('tap')
                 
                   touch.el.trigger(event)
            }
     })
```

## Tap事件

触碰事件，一般用于代替click事件，有四种

```javascript
tap			//手指碰一下屏幕会触发
longTap		//手指长按屏幕会触发
singleTap	//手指碰一下屏幕会触发
doubleTap	//手指双击屏幕会触发
```

## Swipe事件

浏览器并没有内置swipe事件，可以通过touch事件（touchstart、touchmove和touchend）模拟swipe效果。zeptojs提供了完整的tap和swipe事件。 

滑动事件，有五种

```javascript
swipe		//手指在屏幕上滑动时会触发
swipeLeft	//手指在屏幕上向左滑动时会触发
swipeRight	//手指在屏幕上向右滑动时会触发
swipeUp		//手指在屏幕上向上滑动时会触发
swipeDown	//手指在屏幕上向下滑动时会触发
```

- 实例

> Zepto

```html
<style>.delete { display: none; }</style>

<ul id=items>
  <li>List item 1 <span class=delete>DELETE</span></li>
  <li>List item 2 <span class=delete>DELETE</span></li>
</ul>

<script>
// show delete buttons on swipe
$('#items li').swipe(function(){
  $('.delete').hide()
  $('.delete', this).show()
})

// delete row on tapping delete button
$('.delete').tap(function(){
  $(this).parent('li').remove()
})
</script>
```

> jquery.touchSwipe.js

```

```



## 手势事件

当两个手指触摸屏幕时就会产生手势，手势通常会改变显示项的大小，或者旋转显示项。有三个手势事件

```javascript
gesturestart	//当有两根或多根手指放到屏幕上的时候触发
gesturechange	//当有两根或多根手指在屏幕上，并且有手指移动的时候触发
gestureend		//任意一根手指提起的时候触发，结束gesture
```

只有两个手指都触摸到事件的接收容器时才会触发这些事件。在一个元素上设置事件处理程序，意味着两个手指必须同时位于该元素的范围之内，才能触发手势事件（这个元素就是目标）。

由于这些事件冒泡，所以将事件处理程序放在文档上也可以处理所有手势事件。此时，事件的目标就算两个手指都位于其范围内的那个元素。

每个手势事件的event对象都包含着标准的鼠标事件属性：

```javascript
bubbles		//起泡事件的类型
cancelable	//是否用 preventDefault()方法可以取消与事件关联的默认动作
clientX		//返回当事件被触发时，鼠标指针的水平坐标
clientY		//返回当事件触发时，鼠标指针的垂直坐标
screenX		//当某个事件被触发时，鼠标指针的水平坐标
screenY		//返回当某个事件被触发时，鼠标指针的垂直坐标
view,detail,altKey,shiftKey,ctrlKey和metaKey。
```

此外还有两个额外的属性

```javascript
rotation	//表示手指变化引起的旋转角度，负值表示逆时针旋转，正值表示顺时针旋转（该值从0开始）。
scale		//表示两个手指间距离的变化情况（例如向内收缩会缩短距离）；这个值从1开始，并随距离拉大而增长，随距离缩短而减小。
```

- 事件触发顺序

```
第一根手指放下，触发touchstart
第二根手指放下，触发gesturestart
触发第二根手指的touchstart
立即触发gesturechange
手指移动，持续触发gesturechange
第二根手指提起，触发gestureend，以后将不会再触发gesturechange
触发第二根手指的touchend
触发touchstart（多根手指在屏幕上，提起一根，会刷新一次全局 touch，重新触发第一根手指的touchstart）
提起第一根手指，触发touchend
```

- 实例

```javascript
function handleGestureEvent(event){
   var output=document.getElementById("output");
    switch(event.type){
         case "gesturestart":
                output.innerHTML="Gesture started ( "+event.ratation+", scale"+event.scale+")";
                break;
            case "gestureend":
                output.innerHTML+="<br/>Gesture ended ("+event.rotation+", scale"+event.scale+")";
                break;
            case "gesturechange":
                event.preventDefault(); //阻止滚动
                output.innerHTML+="<br/>Gesture changed ("+event.rotation+",scale "+event.scale+")";
    }
}
EventUtil.addHandler(document,"gesturestart",handleGestureEvent);
EventUtil.addHandler(document,"gestureend",handleGestureEvent);
EventUtil.addHandler(document,"gesturechange",handleGestureEvent);
```

# 文件处理

## 上传文件

> 获取文件流

```
<div>
    上传文件 ： <input type="file" name = "file" id = "fileId" /> 
</div>
<script>
    function getFile() {
    //js写法        
    var file=document.getElementById('fileId').files[0];//获取文件流
    var fileName =  file.name;//获取文件名
    var fileSize = file.size;// 获取文件大小byte
    //jq写法
    var file = $('#fileId')[0].files[0]; 
    var filePath = $('#fileId').val(); // 文件路径
    var arr = filePath.split("\\");
    var fileName = arr[arr.length-1].split('.')[0];
  }
```

> 文件上传

```js
//上传文件
function uploadFiles(){                                                         
       var formData = new FormData();
       formData.append("file",$("#uploadFile")[0].files[0]);//append()里面的第一个参数file对应permission/upload里面的参数file                          
        $.ajax({
           type:"post",
           async:true,  //这里要设置异步上传，才能成功调用myXhr.upload.addEventListener('progress',function(e){}),progress的回掉函数
           Accept:'text/html;charset=UTF-8',
           data:formData,
           contentType:"multipart/form-data",
           url: uploadUrl,
           processData: false, // 告诉jQuery不要去处理发送的数据
           contentType: false, // 告诉jQuery不要去设置Content-Type请求头
           xhr:function(){                        
               myXhr = $.ajaxSettings.xhr();
               // check if upload property exists
               if(myXhr.upload){ 
               		myXhr.upload.addEventListener('progress',function(e){                            
                       var loaded = e.loaded; //已经上传大小情况 
                       var total = e.total;   //附件总大小 
                       var percent = Math.floor(100*loaded/total)+"%"; //已经上传的百分比  
                       console.log("已经上传了："+percent);                 
                       $("#processBar").css("width",percent);                                                                
                   	}, false); // for handling the progress of the upload
            	}
               return myXhr;
           },                    
           success:function(data){                      
               console.log("上传成功!!!!");                        
           },
           error:function(){
               alert("上传失败！");
           }
       });                             
}
```

> 多文件上传

多文件上传可采用方案如下

```
1.单文件上传(多个input标签,多次传输)
2.同一目录下多文件一次性(一个input标签，一次传输/多次传输)
```

实例

递归

```js
//html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<link rel="stylesheet" type="text/css" href="./css/NewFile.css" rel="external nofollow" >
<script type="text/javascript" src="./js/jquery-1.9.1.min.js"></script>
<script type="text/javascript" src="./js/fileMuti.js"></script>
</head>
<body>
<div id="test">
<input type="file" id="fileMutiply" name="files" multiple="multiple" >
</div>
</body>
</html>

// js

/**
 * 
 */
var i=0;
var j=0;
$(function(){
  $("#fileMutiply").change(function eventStart(){
    var ss =this.files; //获取当前选择的文件对象
     for(var m=0;m<ss.length;m++){ 
     	//循环添加进度条
        efileName = ss[m].name ;
     	if (ss[m].size> 1024 * 1024){
      		sfileSize = (Math.round(ss[m].size /(1024 * 1024))).toString() + 'MB';
      	}
    	else{
      		sfileSize = (Math.round(ss[m].size/1024)).toString() + 'KB';
      	}
     	$("#test").append(
            "<li id="+m+"file><div class='progress'><div id="+m+"barj class='progressbar'></div></div><span class='filename'>"+efileName+"</span><span id="+m+"pps class='progressnum'>"+(sfileSize)+"</span></li>");
         }
     sendAjax();
     function sendAjax() {
     	//采用递归的方式循环发送ajax请求
        if(j>=ss.length)  { 
         	$("#fileMutiply").val("");
            j=0;
          	return; 
        }
        var formData = new FormData();
        formData.append('files', ss[j]); //将该file对象添加到formData对象中
        $.ajax({
            url:'fileUpLoad.action',
            type:'POST',
            cache: false,
            data:{},//需要什么参数，自己配置
            data: formData,//文件以formData形式传入
            processData : false, 
            //必须false才会自动加上正确的Content-Type 
            contentType : false , 
          	/*  beforeSend:beforeSend,//发送请求
            complete:complete,//请求完成   */  
     		xhr: function(){   //监听用于上传显示进度
            	var xhr = $.ajaxSettings.xhr();
              	if(onprogress && xhr.upload) {
               		xhr.upload.addEventListener("progress" , onprogress, false);
               		return xhr;
            	}
            } ,
            success:function(data){
            	$(".filelist").find("#"+j+"file").remove();//移除进度条样式
               	j++; //递归条件
              	sendAjax();
            },
            error:function(xhr){
             alert("上传出错");
            }               
          });
        } 
  })
    function onprogress(evt){
     var loaded = evt.loaded;   //已经上传大小情况 
     var tot = evt.total;   //附件总大小 
     var per = Math.floor(100*loaded/tot); //已经上传的百分比 
     $(".filelist").find("#"+j+"pps").text(per +"%");
     $(".filelist").find("#"+j+"barj").width(per+"%");
     };
})
```

多文件一次性

```js
//html
<div class="input-group">
    <input type="file" id="attachment" multiple="multiple">
    <span id="progress_bar" style="color: #1AB394;display: table-cell"></span>
</div>
<ul id="attachment_list"></ul>
<button class="btn btn-file">upload file</button>

// js
ajaxSetup({   //laravel中的request要带这个header参数
        headers: {
            'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content')
        }
    });

$('.btn-file').click(function(){
        if($('#attachment').val() == '')
            alert('请选择文件再上传');
        else{
            var path = $('#attachment')[0].files;
            var formData = new FormData();
            var names = '';
            /*
            提示：FormData不能写数组，array json都不行，能写简单的key->value键值对。
            键值对中key不能是中文，不然后台读不出来，而且要保证key的唯一性，
            那么我就用文件名path[i].name用md5加密一下好了，当然你也可以用自己喜欢的加密方式。
            因为laravel不能便利地读取所有file，只能用file('key')读取key值的value，
            是的，所以你不知道key值是读不出你要的东西的。因为文件的key是变化的，所以我这里写定一个info字段，
            然后把文件的key写成字符串，然后后台解析字符串，再根据里面的字段获取文件。
            你也可以写其他需要的数据的键值对到FormData里面，一并传到后台，当成一个虚拟form表单用就行了。
            */
            for(var i= 0,name;i<path.length;i++){
                name = $.md5(path[i].name);
                formData.append(name, path[i]);
                names += name + ',';
            }
            formData.append('info',names);
            $.ajax({
                url: "{{route('upload')}}",
                type: 'POST',
                cache: false,
                data: formData,
                processData: false,
                contentType: false,
                beforeSend: function(){
                    $('#progress_bar').css('color','#1AB394').show();
                },
                success: function(result
                {
                $('#progress_bar').html(result.info).css('color','black').fadeOut(3000,function(){$(this).html('')});
                },
                error: function (result) {

                },
                xhr: function(){
                    var xhr = $.ajaxSettings.xhr();
                   if(onprogress && xhr.upload) {
                        xhr.upload.addEventListener("progress" , onprogress, false);
                        return xhr;
                   }
                }
            });
/*
小tips：在网上查找遇到一些方法（例如function A()），没有详细介绍，不知道总共完整传多少个参数，
每个参数长什么样子的，可以写成function A(a,b,c,d,e,…………){//然后写log打印出来}，
这里只有一个event对象参数，所以我写4个形参上去，然后写日志出来，只有第一个参数写出来是一个对象，
而且里面有什么属性也会写出来，后面3个形参则输出为空，
那么这时候就能写定 function A(obj){//只有一个参数，自己写个喜欢的形参名}。
前端后台都能用这个小技巧哦~
*/
function onprogress(evt){   
        console.log(evt);
        var loaded = evt.loaded;
        var tot = evt.total;
        $('#progress_bar').html(Math.floor(100*loaded/tot)+'%');
    }
```



## 下载文件

文件格式

```js
//文件下载
var blob = new Blob([要保存的文件流], { type: 'application/octet-stream' }),
//filename，摘取了常用的部分，其实还有其他一些mimetypes = array(
//    'doc'        => 'application/msword',
//    'bin'        => 'application/octet-stream',
//    'exe'        => 'application/octet-stream',
//    'so'        => 'application/octet-stream',
//    'dll'        => 'application/octet-stream',
//    'pdf'        => 'application/pdf',
//    'ai'        => 'application/postscript',
//    'xls'        => 'application/vnd.ms-excel',
//    'ppt'        => 'application/vnd.ms-powerpoint',
//    'dir'        => 'application/x-director',
//    'js'        => 'application/x-javascript',
//    'swf'        => 'application/x-shockwave-flash',
//    'xhtml'        => 'application/xhtml+xml',
//    'xht'        => 'application/xhtml+xml',
//    'zip'        => 'application/zip',
//    'mid'        => 'audio/midi',////    'midi'        => 'audio/midi',
//    'mp3'        => 'audio/mpeg',
//    'rm'        => 'audio/x-pn-realaudio',
//    'rpm'        => 'audio/x-pn-realaudio-plugin',
//    'wav'        => 'audio/x-wav',
//    'bmp'        => 'image/bmp',
//    'gif'        => 'image/gif',
//    'jpeg'        => 'image/jpeg',
//    'jpg'        => 'image/jpeg',
//    'png'        => 'image/png',
//    'css'        => 'text/css',
//    'html'        => 'text/html',
//    'htm'        => 'text/html',
//    'txt'        => 'text/plain',
//    'xsl'        => 'text/xml',
//    'xml'        => 'text/xml',
//    'mpeg'        => 'video/mpeg',
//    'mpg'        => 'video/mpeg',
//    'avi'        => 'video/x-msvideo',
//    'movie'        => 'video/x-sgi-movie',
//);
fileName = 'filename' + path.substring(path.lastIndexOf("."), path.length);
downFile(blob, fileName);

//js下载文件流
function downFile(blob, fileName) {
    if (window.navigator.msSaveOrOpenBlob) {
        navigator.msSaveBlob(blob, fileName);
    } else {
        var link = document.createElement('a');
        link.href = window.URL.createObjectURL(blob);
        link.download = fileName;
        link.click();
        window.URL.revokeObjectURL(link.href);
    }
}
```









