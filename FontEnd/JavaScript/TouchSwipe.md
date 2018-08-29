# touchSwipe

jquery.touchSwipe一款专门为移动设备设计的jquery插件，用于监听单个和多个手指触摸等事件。

## 特点

```
1、监听滑动的4个方向：上、下、左、右；
2、监听多个手指收缩还是外张；
3、支持单手指或双手指触摸事件；
4、支持单击事件touchSwipe对象和它的子对象；
5、可定义临界值和最大时间来判断手势的实际滑动；
6、滑动事件有4个触发点：“开始”,“移动”,“结束”和“取消”；
7、结束事件可以在触摸释放或是达到临界值时触发；
8、允许手指刷和页面滚屏；
9、可禁止用户通过输入元素（如：按钮、表单、文本框等）触发事件；
```

## 安装

```javascript
//NPM
npm install jquery-touchswipe --save

//Bower
bower install jquery-touchswipe --save

//将压缩文件添加到你的项目里
<script src="js/jquery.touchSwipe.min.js" type="text/javascript"></script>
```

## 使用

### 方法

```javascript
// swipe 初始化
$("#element").swipe（{
	//给id为element的容器触摸滑动监听事件
}）;

// destroy：彻底销毁swipe插件，必须重新初始化插件才能再次使用任何方法
$("#element").swipe("destroy");

// disable：禁止插件，使插件失去作用,返回值：现在与插件绑定的DOM元素
$("#element").swipe("disable");

// enable：重新启用插件，恢复原有的配置,返回值：现在与插件绑定的DOM元素
$("#element").swipe("enable");

// option：设置或获取属性
$("#element").swipe("option", "threshold"); // 返回阈值
$("#element").swipe("option", "threshold", 100); // 设置阈值
$("#element").swipe("option", {threshold:100, fingers:3} ); // 设置多个属性
$("#element").swipe({threshold:100, fingers:3} ); // 设置多个属性 -"option"方法可省略
$("#element").swipe("option"); // 返回现有的options
```

### 事件

#### swipe

滑动事件

```javascript
swipe:function(event, direction, distance, duration, fingerCount, fingerData) {
            $(this).text("You swiped " + direction ); 
}
```

参数

| 名字               | 类型        | 说明                 |
| ------------------ | ----------- | -------------------- |
| `event`            | eventObject | 原生事件对象         |
| `direction`        | int         | 滑动方向             |
| `distance`         | int         | 滑动距离             |
| `duration`         | int         | 滑动时长(毫秒)       |
| `fingerCount`      | int         | 手指数               |
| `fingerData`       | object      | 事件中手指的坐标信息 |
| `currentDirection` | String      | 当前滑动方向         |

#### swipeDown

向下滑动事件

```javascript
swipeDown: function(event, direction, distance, duration, fingerCount, fingerData) {}
```

#### swipeUp

向上滑动事件

```javascript
swipeUp: function(event, direction, distance, duration, fingerCount, fingerData){}
```

#### swipeLeft

向左滑动事件

```javascript
swipeLeft: function(event, direction, distance, duration, fingerCount, fingerData){}
```

#### swipeRight

向右滑动事件

```javascript
swipeRight: function(event, direction, distance, duration, fingerCount, fingerData){}
```

#### swipeStatus

滑动过程会持续触发swipeStatus事件，不受阈值限制

```javascript
swipeStatus: function(event, direction, distance, duration, fingerCount, fingerData, currentDirection) {}
```

#### tap

当用户简单地点击或者轻击而不是滑动一个元素时tap/click事件将被触发

```javascript
tap:function(event,target){
	console.log($(target).attr("class"));
}
```

参数

| 参数   | 说明                  |
| ------ | --------------------- |
| event  | 原生事件对象          |
| target | 被点击的元素(DOM对象) |

#### doubleTap

当用户连续两次点击或者轻击而不是滑动一个元素时事件将被触发

```javascript
doubleTap:function(event,target){
	console.log($(target).attr("class"));
}
```



