# Zepto

## 与JQuery的异同

-  同

```
Zepto最初是为移动端开发的库，是jQuery的轻量级替代品，因为它的API和jQuery相似，而文件更小。Zepto最大的优势是它的文件大小，只有8k多，是目前功能完备的库中最小的一个，尽管不大，Zepto所提供的工具足以满足开发程序的需要。大多数在jQuery中·常用的API和方法Zepto都有，Zepto中还有一些jQuery中没有的。另外，因为Zepto的API大部分都能和jQuery兼容，所以用起来极其容易，如果熟悉jQuery，就能很容易掌握Zepto。你可用同样的方式重用jQuery中的很多方法，也可以方面地把方法串在一起得到更简洁的代码，甚至不用看它的文档。
```

- 异

> 针对移动端程序，Zepto有优化

```
Zepto有一些基本的触摸事件可以用来做触摸屏交互（tap事件、swipe事件），Zepto是不支持IE浏览器的，这不是Zepto的开发者Thomas Fucks在跨浏览器问题上犯了迷糊，而是经过了认真考虑后为了降低文件尺寸而做出的决定，就像jQuery的团队在2.0版中不再支持旧版的IE（6 7 8）一样。因为Zepto使用jQuery句法，所以它在文档中建议把jQuery作为IE上的后备库。那样程序仍能在IE中，而其他浏览器则能享受到Zepto在文件大小上的优势，然而它们两个的API不是完全兼容的，所以使用这种方法时一定要小心，并要做充分的测试。
```

>Dom操作的区别

```javascript
// 添加id时jQuery不会生效而Zepto会生效
(function($) {
     $(function() {
         var $insert = $('<p>jQuery 插入</p>', {
             id: 'insert-by-jquery'
         });
         $insert.appendTo($('body'));
     });
})(window.jQuery);   
// <p>jQuery 插入<p>

Zepto(function($) {  
    var $insert = $('<p>Zepto 插入</p>', {
        id: 'insert-by-zepto'
    });
    $insert.appendTo($('body'));
});
// <p id="insert-by-zepto">Zepto 插入</p>
```

> 事件触发的区别

```javascript
// 使用jQuery时load事件的处理函数不会执行；使用Zepto时load事件的处理函数会执行。
(function($) {
    $(function() {    
        $script = $('<script />', {
            src: 'http://cdn.amazeui.org/amazeui/1.0.1/js/amazeui.js',
            id: 'ui-jquery'
        });

        $script.appendTo($('body'));

        $script.on('load', function() {
            console.log('jQ script loaded');
        });
    });
})(window.jQuery);

Zepto(function($) {  
    $script = $('<script />', {
        src: 'http://cdn.amazeui.org/amazeui/1.0.1/js/amazeui.js',
        id: 'ui-zepto'
    });

    $script.appendTo($('body'));

    $script.on('load', function() {
        console.log('zepto script loaded');
    });
});
```

> 事件委托的区别
> 

```javascript
var $doc = $(document);
$doc.on('click', '.a', function () {
    alert('a事件');
    $(this).removeClass('a').addClass('b');
});
$doc.on('click', '.b', function () {
    alert('b事件');
});
```

在Zepto中，当a被点击后，依次弹出了内容为”a事件“和”b事件“，说明虽然事件委托在.a上可是却也触发了.b上的委托。但是在 jQuery 中只会触发.a上面的委托弹出”a事件“。Zepto中，document上所有的click委托事件都依次放入到一个队列中，点击的时候先看当前元素是不是.a，符合则执行，然后查看是不是.b，符合则执行。而在jQuery中，document上委托了2个click事件，点击后通过选择符进行匹配，执行相应元素的委托事件。

> 函数的区别

```
// width()和height()的区别
Zepto由盒模型(box-sizing)决定，用.width()返回赋值的width，用.css('width')返回加border等的结果；jQuery会忽略盒模型，始终返回内容区域的宽/高(不包含padding、border)

// offset()的区别
Zepto返回{top,left,width,height}；jQuery返回{width,height}

Zepto无法获取隐藏元素宽高，jQuery 可以

Zepto中没有为原型定义extend方法而jQuery有

Zepto 的each 方法只能遍历 数组，不能遍历JSON对象

Zepto在操作dom的selected和checked属性时尽量使用prop方法，在读取属性值的情况下优先于attr。Zepto获取select元素的选中option不能用类似jQuery的方法$('option[selected]'),因为selected属性不是css的标准属性。应该使用$('option').not(function(){ return !this.selected })
```

> 不支持的选择器

```
基本伪类:first、:not(selector) 、:even 、:odd 、:eq(index) 、:gt(index) 、:lang1.9+ 、:last 、:lt(index) 、:header、:animated 、:focus1.6+ 、:root1.9+ 、:target1.9+。
内容伪类：:contains(text) 、:empty、 :has(selector)、 :parent 。
可见性伪类：:hidden 、:visible 。
属性选择器：[attribute!=value]。
表单伪类：:input、 :text、 :password、 :radio、 :checkbox、 :submit、 :image、 :reset、 :button、 :file、 :hidden 。
表单对象属性：:selected。
```

