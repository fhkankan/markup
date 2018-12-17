[TOC]
# Vue中阶

## 路由

Vue.js 路由允许我们通过不同的 URL 访问不同的内容。

通过 Vue.js 可以实现多视图的单页Web应用（single page web application，SPA）。

Vue.js 路由需要载入 [vue-router 库](https://github.com/vuejs/vue-router)

中文文档地址：[vue-router文档](http://router.vuejs.org/zh-cn/)。

### 安装

直接下载/CDN

```
https://unpkg.com/vue-router/dist/vue-router.js
```

NPM

```
// 淘宝镜像
cnpm install vue-router
```

### 创建

Vue.js + vue-router 可以很简单的实现单页应用。

`<router-link>`是一个组件，该组件用于设置一个导航链接，切换不同 HTML 内容。 `to` 属性为目标地址， 即要显示的内容。

```html
// 将 vue-router 加进来，然后配置组件和路由映射，再告诉 vue-router 在哪里渲染它们
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/vue/2.4.2/vue.min.js"></script>
<script src="https://cdn.staticfile.org/vue-router/2.7.0/vue-router.min.js"></script>
</head>
<body>
<div id="app">
  <h1>Hello App!</h1>
  <p>
    <!-- 使用 router-link 组件来导航. -->
    <!-- 通过传入 `to` 属性指定链接. -->
    <!-- <router-link> 默认会被渲染成一个 `<a>` 标签 -->
    <router-link to="/foo">Go to Foo</router-link>
    <router-link to="/bar">Go to Bar</router-link>
  </p>
  <!-- 路由出口 -->
  <!-- 路由匹配到的组件将渲染在这里 -->
  <router-view></router-view>
</div>

<script>
// 0. 如果使用模块化机制编程，導入Vue和VueRouter，要调用 Vue.use(VueRouter)

// 1. 定义（路由）组件。
// 可以从其他文件 import 进来
const Foo = { template: '<div>foo</div>' }
const Bar = { template: '<div>bar</div>' }

// 2. 定义路由
// 每个路由应该映射一个组件。 其中"component" 可以是通过 Vue.extend() 创建的组件构造器，或者，只是一个组件配置对象。我们晚点再讨论嵌套路由。
const routes = [
  { path: '/foo', component: Foo },
  { path: '/bar', component: Bar }
]

// 3. 创建 router 实例，然后传 `routes` 配置
// 你还可以传别的配置参数, 不过先这么简单着吧。
const router = new VueRouter({
  routes // （缩写）相当于 routes: routes
})

// 4. 创建和挂载根实例。
// 记得要通过 router 配置参数注入路由，从而让整个应用都有路由功能
const app = new Vue({
  router
}).$mount('#app')

// 现在，应用已经启动了！
</script>
</body>
</html>
```

### 属性

`<router-link>` 的相关属性

> to

表示目标路由的链接。 当被点击后，内部会立刻把 to 的值传到 router.push()，所以这个值可以是一个字符串或者是描述目标位置的对象

```html
<!-- 字符串 -->
<router-link to="home">Home</router-link>
<!-- 渲染结果 -->
<a href="home">Home</a>

<!-- 使用 v-bind 的 JS 表达式 -->
<router-link v-bind:to="'home'">Home</router-link>

<!-- 不写 v-bind 也可以，就像绑定别的属性一样 -->
<router-link :to="'home'">Home</router-link>

<!-- 同上 -->
<router-link :to="{ path: 'home' }">Home</router-link>

<!-- 命名的路由 -->
<router-link :to="{ name: 'user', params: { userId: 123 }}">User</router-link>

<!-- 带查询参数，下面的结果为 /register?plan=private -->
<router-link :to="{ path: 'register', query: { plan: 'private' }}">Register</router-link>
```

> replace

设置 replace 属性的话，当点击时，会调用 router.replace() 而不是 router.push()，导航后不会留下 history 记录。

```html
<router-link :to="{ path: '/abc'}" replace></router-link>
```

> append

设置 append 属性后，则在当前 (相对) 路径前添加基路径。例如，我们从 /a 导航到一个相对路径 b，如果没有配置 append，则路径为 /b，如果配了，则为 /a/b

```html
<router-link :to="{ path: 'relative/path'}" append></router-link>
```

> tag

有时候想要 `<router-link>` 渲染成某种标签，例如 `<li>`。 于是我们使用 `tag` prop 类指定何种标签，同样它还是会监听点击，触发导航

```html
<router-link to="/foo" tag="li">foo</router-link>
<!-- 渲染结果 -->
<li>foo</li>
```

> active-class

设置 链接激活时使用的 CSS 类名。可以通过以下代码来替代

```html
//注意这里 class 使用 active_class="_active"
<style>
   ._active{
      background-color : red;
   }
</style>
<p>
   <router-link v-bind:to = "{ path: '/route1'}" active-class = "_active">Router Link 1</router-link>
   <router-link v-bind:to = "{ path: '/route2'}" tag = "span">Router Link 2</router-link>
</p>
```

> exact-active-class

配置当链接被精确匹配的时候应该激活的 class。可以通过以下代码来替代。

```html
<p>
   <router-link v-bind:to = "{ path: '/route1'}" exact-active-class = "_active">Router Link 1</router-link>
   <router-link v-bind:to = "{ path: '/route2'}" tag = "span">Router Link 2</router-link>
</p>
```

> event

声明可以用来触发导航的事件。可以是一个字符串或是一个包含字符串的数组。

```html
// 设置了 event 为 mouseover ，及在鼠标移动到 Router Link 1 上时导航的 HTML 内容会发生改变
<router-link v-bind:to = "{ path: '/route1'}" event = "mouseover">Router Link 1</router-link>
```

## 过渡动画

### 过渡

Vue 在插入、更新或者移除 DOM 时，提供多种不同方式的应用过渡效果。

Vue 提供了内置的过渡封装组件，该组件用于包裹要实现过渡效果的组件。

```html
<transition name = "nameoftransition">
   <div></div>
</transition>
```

> 实例

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/vue/2.2.2/vue.min.js"></script>
<style>
/* 可以设置不同的进入和离开动画 */
/* 设置持续时间和动画函数 */
.fade-enter-active, .fade-leave-active {
    transition: opacity 2s
}
.fade-enter, .fade-leave-to /* .fade-leave-active, 2.1.8 版本以下 */ {
    opacity: 0
}
</style>
</head>
<body>
<div id = "databinding">
<button v-on:click = "show = !show">点我</button>
<transition name = "fade">
    <p v-show = "show" v-bind:style = "styleobj">动画实例</p>
</transition>
</div>
<script type = "text/javascript">
var vm = new Vue({
el: '#databinding',
    data: {
        show:true,
        styleobj :{
            fontSize:'30px',
            color:'red'
        }
    },
    methods : {
    }
});
</script>
</body>
</html>
```

过渡其实就是一个淡入淡出的效果。Vue在元素显示与隐藏的过渡中，提供了 6 个 class 来切换：

- `v-enter`：定义进入过渡的开始状态。在元素被插入之前生效，在元素被插入之后的下一帧移除。
- `v-enter-active`：定义进入过渡生效时的状态。在整个进入过渡的阶段中应用，在元素被插入之前生效，在过渡/动画完成之后移除。这个类可以被用来定义进入过渡的过程时间，延迟和曲线函数。
- `v-enter-to`: **2.1.8版及以上** 定义进入过渡的结束状态。在元素被插入之后下一帧生效 (与此同时 `v-enter` 被移除)，在过渡/动画完成之后移除。
- `v-leave`: 定义离开过渡的开始状态。在离开过渡被触发时立刻生效，下一帧被移除。
- `v-leave-active`：定义离开过渡生效时的状态。在整个离开过渡的阶段中应用，在离开过渡被触发时立刻生效，在过渡/动画完成之后移除。这个类可以被用来定义离开过渡的过程时间，延迟和曲线函数。
- `v-leave-to`: **2.1.8版及以上** 定义离开过渡的结束状态。在离开过渡被触发之后下一帧生效 (与此同时 `v-leave` 被删除)，在过渡/动画完成之后移除。

对于这些在过渡中切换的类名来说，如果你使用一个没有名字的 `<transition>`，则 `v-` 是这些类名的默认前缀。如果你使用了 `<transition name="my-transition">`，那么 `v-enter` 会替换为 `my-transition-enter`。

`v-enter-active` 和 `v-leave-active` 可以控制进入/离开过渡的不同的缓和曲线

> css过渡

通常我们都使用 CSS 过渡来实现效果

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/vue/2.2.2/vue.min.js"></script>
<style>
/* 可以设置不同的进入和离开动画 */
/* 设置持续时间和动画函数 */
.slide-fade-enter-active {
  transition: all .3s ease;
}
.slide-fade-leave-active {
  transition: all .8s cubic-bezier(1.0, 0.5, 0.8, 1.0);
}
.slide-fade-enter, .slide-fade-leave-to
/* .slide-fade-leave-active 用于 2.1.8 以下版本 */ {
  transform: translateX(10px);
  opacity: 0;
}
</style>
</head>
<body>
<div id = "databinding">
<button v-on:click = "show = !show">点我</button>
<transition name="slide-fade">
    <p v-if="show">菜鸟教程</p>
</transition>
</div>
<script type = "text/javascript">
new Vue({
    el: '#databinding',
    data: {
        show: true
    }
})
</script>
</body>
</html>
```

> css动画

CSS 动画用法类似 CSS 过渡，但是在动画中 v-enter 类名在节点插入 DOM 后不会立即删除，而是在 animationend 事件触发时删除

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/vue/2.2.2/vue.min.js"></script>
<style>
.bounce-enter-active {
  animation: bounce-in .5s;
}
.bounce-leave-active {
  animation: bounce-in .5s reverse;
}
@keyframes bounce-in {
  0% {
    transform: scale(0);
  }
  50% {
    transform: scale(1.5);
  }
  100% {
    transform: scale(1);
  }
}
</style>
</head>
<body>
<div id = "databinding">
<button v-on:click = "show = !show">点我</button>
<transition name="bounce">
    <p v-if="show">菜鸟教程 -- 学的不仅是技术，更是梦想！！！</p>
</transition>
</div>
<script type = "text/javascript">
new Vue({
    el: '#databinding',
    data: {
        show: true
    }
})
</script>
</body>
</html>
```

> 自定义过渡类名

可以通过以下特性来自定义过渡类名：

```
- enter-class
- enter-active-class
- enter-to-class (2.1.8+)
- leave-class
- leave-active-class
- leave-to-class (2.1.8+)
```

自定义过渡的类名优先级高于普通的类名，这样就能很好的与第三方（如：animate.css）的动画库结合使用

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
<script src="https://cdn.staticfile.org/vue/2.2.2/vue.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/animate.css@3.5.1" rel="stylesheet" type="text/css">
</head>
<body>
<div id = "databinding">
<button v-on:click = "show = !show">点我</button>
<transition
    name="custom-classes-transition"
    enter-active-class="animated tada"
    leave-active-class="animated bounceOutRight"
>
    <p v-if="show">菜鸟教程 -- 学的不仅是技术，更是梦想！！！</p>
</transition>
</div>
<script type = "text/javascript">
new Vue({
    el: '#databinding',
    data: {
        show: true
    }
})
</script>
</body>
</html>
```

> 同时使用过渡和动画

Vue 为了知道过渡的完成，必须设置相应的事件监听器。它可以是 `transitionend` 或 `animationend` ，这取决于给元素应用的 CSS 规则。如果你使用其中任何一种，Vue 能自动识别类型并设置监听。

但是，在一些场景中，你需要给同一个元素同时设置两种过渡动效，比如 `animation` 很快的被触发并完成了，而 `transition` 效果还没结束。在这种情况中，你就需要使用 `type` 特性并设置 `animation` 或 `transition` 来明确声明你需要 Vue 监听的类型。

> 显性地过渡持续时间

在很多情况下，Vue 可以自动得出过渡效果的完成时机。默认情况下，Vue 会等待其在过渡效果的根元素的第一个 `transitionend` 或 `animationend` 事件。然而也可以不这样设定——比如，我们可以拥有一个精心编排的一系列过渡效果，其中一些嵌套的内部元素相比于过渡效果的根元素有延迟的或更长的过渡效果。

在这种情况下你可以用 `<transition>` 组件上的 `duration` 属性定制一个显性的过渡持续时间 (以毫秒计)：

```html
<transition :duration="1000">...</transition>
```

你也可以定制进入和移出的持续时间：

```html
<transition :duration="{ enter: 500, leave: 800 }">...</transition>
```

### 钩子

> html

```html
<transition
  v-on:before-enter="beforeEnter"
  v-on:enter="enter"
  v-on:after-enter="afterEnter"
  v-on:enter-cancelled="enterCancelled"
 
  v-on:before-leave="beforeLeave"
  v-on:leave="leave"
  v-on:after-leave="afterLeave"
  v-on:leave-cancelled="leaveCancelled"
>
  <!-- ... -->
</transition>
```

> js

```js
// ...
methods: {
  // --------
  // 进入中
  // --------
 
  beforeEnter: function (el) {
    // ...
  },
  // 此回调函数是可选项的设置
  // 与 CSS 结合时使用
  enter: function (el, done) {
    // ...
    done()
  },
  afterEnter: function (el) {
    // ...
  },
  enterCancelled: function (el) {
    // ...
  },
 
  // --------
  // 离开时
  // --------
 
  beforeLeave: function (el) {
    // ...
  },
  // 此回调函数是可选项的设置
  // 与 CSS 结合时使用
  leave: function (el, done) {
    // ...
    done()
  },
  afterLeave: function (el) {
    // ...
  },
  // leaveCancelled 只用于 v-show 中
  leaveCancelled: function (el) {
    // ...
  }
}
```

这些钩子函数可以结合 CSS transitions/animations 使用，也可以单独使用。

当只用 JavaScript 过渡的时候，**在 enter 和 leave 中必须使用 done 进行回调**。否则，它们将被同步调用，过渡会立即完成。

推荐对于仅使用 JavaScript 过渡的元素添加 `v-bind:css="false"`，Vue 会跳过 CSS 的检测。这也可以避免过渡过程中 CSS 的影响。

> 实例

```html
<div id = "databinding">
<button v-on:click = "show = !show">点我</button>
<transition
    v-on:before-enter="beforeEnter"
    v-on:enter="enter"
    v-on:leave="leave"
    v-bind:css="false"
  >
    <p v-if="show">菜鸟教程 -- 学的不仅是技术，更是梦想！！！</p>
</transition>
</div>
<script type = "text/javascript">
new Vue({
  el: '#databinding',
  data: {
    show: false
  },
  methods: {
    beforeEnter: function (el) {
      el.style.opacity = 0
      el.style.transformOrigin = 'left'
    },
    enter: function (el, done) {
      Velocity(el, { opacity: 1, fontSize: '1.4em' }, { duration: 300 })
      Velocity(el, { fontSize: '1em' }, { complete: done })
    },
    leave: function (el, done) {
      Velocity(el, { translateX: '15px', rotateZ: '50deg' }, { duration: 600 })
      Velocity(el, { rotateZ: '100deg' }, { loop: 2 })
      Velocity(el, {
        rotateZ: '45deg',
        translateY: '30px',
        translateX: '30px',
        opacity: 0
      }, { complete: done })
    }
  }
})
</script>
```

### 初始渲染

可以通过 `appear` 特性设置节点在初始渲染的过渡

```html
<transition appear>
  <!-- ... -->
</transition>
```

这里默认和进入/离开过渡一样，同样也可以自定义 CSS 类名

```html
<transition
  appear
  appear-class="custom-appear-class"
  appear-to-class="custom-appear-to-class" (2.1.8+)
  appear-active-class="custom-appear-active-class"
>
  <!-- ... -->
</transition>
```

自定义 JavaScript 钩子

```html
<transition
  appear
  v-on:before-appear="customBeforeAppearHook"
  v-on:appear="customAppearHook"
  v-on:after-appear="customAfterAppearHook"
  v-on:appear-cancelled="customAppearCancelledHook"
>
  <!-- ... -->
</transition>
```

### 多个元素

我们可以设置多个元素的过渡，一般列表与描述：

需要注意的是当有相同标签名的元素切换时，需要通过 `key` 特性设置唯一的值来标记以让 Vue 区分它们，否则 Vue 为了效率只会替换相同标签内部的内容

```html
<transition>
  <table v-if="items.length > 0">
    <!-- ... -->
  </table>
  <p v-else>抱歉，没有找到您查找的内容。</p>
</transition>
```

实例

```html
<transition>
  <button v-if="isEditing" key="save">
    Save
  </button>
  <button v-else key="edit">
    Edit
  </button>
</transition>
```

在一些场景中，也可以通过给同一个元素的 `key` 特性设置不同的状态来代替 `v-if` 和 `v-else`，上面的例子可以重写为

```html
<transition>
  <button v-bind:key="isEditing">
    {{ isEditing ? 'Save' : 'Edit' }}
  </button>
</transition>
```

使用多个 `v-if` 的多个元素的过渡可以重写为绑定了动态属性的单个元素过渡。例如

```html
<transition>
  <button v-if="docState === 'saved'" key="saved">
    Edit
  </button>
  <button v-if="docState === 'edited'" key="edited">
    Save
  </button>
  <button v-if="docState === 'editing'" key="editing">
    Cancel
  </button>
</transition>
```

可重写为

```html
<transition>
  <button v-bind:key="docState">
    {{ buttonMessage }}
  </button>
</transition>

// ...
computed: {
  buttonMessage: function () {
    switch (this.docState) {
      case 'saved': return 'Edit'
      case 'edited': return 'Save'
      case 'editing': return 'Cancel'
    }
  }
}
```

## 混入

混入 (mixins)定义了一部分可复用的方法或者计算属性。混入对象可以包含任意组件选项。当组件使用混入对象时，所有混入对象的选项将被混入该组件本身的选项。

```js
var vm = new Vue({
    el: '#databinding',
    data: {
    },
    methods : {
    },
});
// 定义一个混入对象
var myMixin = {
    created: function () {
        this.startmixin()
    },
    methods: {
        startmixin: function () {
            document.write("欢迎来到混入实例");
        }
    }
};
var Component = Vue.extend({
    mixins: [myMixin]
})
var component = new Component();
```

### 选项合并

当组件和混入对象含有同名选项时，这些选项将以恰当的方式混合。

比如，数据对象在内部会进行浅合并 (一层属性深度)，在和组件的数据发生冲突时以组件数据优先。

以下实例中，Vue 实例与混入对象包含了相同的方法。从输出结果可以看出两个选项合并了

```js
var mixin = {
    created: function () {
        document.write('混入调用' + '<br>')
    }
}
new Vue({
    mixins: [mixin],
    created: function () {
        document.write('组件调用' + '<br>')
    }
});
```

如果 methods 选项中有相同的函数名，则 Vue 实例优先级会较高。如下实例，Vue 实例与混入对象的 methods 选项都包含了相同的函数

```js
var mixin = {
    methods: {
        hellworld: function () {
            document.write('HelloWorld 方法' + '<br>');
        },
        samemethod: function () {
            document.write('Mixin：相同方法名' + '<br>');
        }
    }
};
var vm = new Vue({
    mixins: [mixin],
    methods: {
        start: function () {
            document.write('start 方法' + '<br>');
        },
        samemethod: function () {
            document.write('Main：相同方法名' + '<br>');
        }
    }
});
vm.hellworld();
vm.start();
vm.samemethod();
```

### 全局混入

也可以全局注册混入对象。注意使用！ 一旦使用全局混入对象，将会影响到 所有 之后创建的 Vue 实例。使用恰当时，可以为自定义对象注入处理逻辑。

谨慎使用全局混入对象，因为会影响到每个单独创建的 Vue 实例 (包括第三方模板)

```js
// 为自定义的选项 'myOption' 注入一个处理器。
Vue.mixin({
  created: function () {
    var myOption = this.$options.myOption
    if (myOption) {
      console.log(myOption)
    }
  }
})
 
new Vue({
  myOption: 'hello!'
})
// => "hello!"
```

## vue-resource

Vue.js是数据驱动的，这使得我们并不需要直接操作DOM，如果我们不需要使用jQuery的DOM选择器，就没有必要引入jQuery。vue-resource是Vue.js的一款插件，它可以通过XMLHttpRequest或JSONP发起请求并处理响应。也就是说，$.ajax能做的事情，vue-resource插件一样也能做到，而且vue-resource的API更为简洁。另外，vue-resource还提供了非常有用的inteceptor功能，使用inteceptor可以在请求前和请求后附加一些行为，比如使用inteceptor在ajax请求时显示loading界面。

> 优点

```
1.体积小
vue-resource非常小巧，在压缩以后只有大约12KB，服务端启用gzip压缩后只有4.5KB大小，这远比jQuery的体积要小得多。

2.支持主流的浏览器
和Vue.js一样，vue-resource除了不支持IE 9以下的浏览器，其他主流的浏览器都支持。

3.支持Promise API和URI Templates
Promise是ES6的特性，Promise的中文含义为“先知”，Promise对象用于异步计算。
URI Templates表示URI模板，有些类似于ASP.NET MVC的路由模板。

4.支持拦截器
拦截器是全局的，拦截器可以在请求发送前和发送请求后做一些处理。
拦截器在一些场景下会非常有用，比如请求发送前在headers中设置access_token，或者在请求失败时，提供共通的处理方式
```

### 引用

```html
<script src="js/vue.js"></script>
<script src="js/vue-resource.js"></script>
```

### 使用

> Get/Post请求

```js
// 基于全局Vue对象使用http
Vue.http.get('/someUrl', [options]).then(successCallback, errorCallback);
Vue.http.post('/someUrl', [body], [options]).then(successCallback, errorCallback);

// 在一个Vue实例内使用$http
this.$http.get('/someUrl', [options]).then(successCallback, errorCallback);
this.$http.post('/someUrl', [body], [options]).then(successCallback, errorCallback);
```

> Then响应

在发送请求后，使用then方法来处理响应结果，then方法有两个参数，第一个参数是响应成功时的回调函数，第二个参数是响应失败时的回调函数。
then方法的回调函数也有两种写法，第一种是传统的函数写法，第二种是更为简洁的ES 6的Lambda写法：

```js
// 传统写法
this.$http.get('/someUrl', [options]).then(function(response){
  // 响应成功回调
}, function(response){
  // 响应错误回调
});

// Lambda写法
this.$http.get('/someUrl', [options]).then((response) => {
  // 响应成功回调
}, (response) => {
  // 响应错误回调
});
```

### API

vue-resource的请求API是按照REST风格设计的，它提供了7种请求API

```
get(url, [options])
head(url, [options])
delete(url, [options])
jsonp(url, [options])
post(url, [body], [options])
put(url, [body], [options])
patch(url, [body], [options])
```

除了jsonp以外，另外6种的API名称是标准的HTTP方法。当服务端使用REST API时，客户端的编码风格和服务端的编码风格近乎一致，这可以减少前端和后端开发人员的沟通成本。

| 客户端请求方法         | 服务端处理方法 |
| ---------------------- | -------------- |
| this.$http.get(...)    | Getxxx         |
| this.$http.post(...)   | Postxxx        |
| this.$http.put(...)    | Putxxx         |
| this.$http.delete(...) | Deletexxx      |

> options对象

| 参数        | 类型                    | 描述                                                         |
| ----------- | ----------------------- | ------------------------------------------------------------ |
| url         | string                  | 请求的URL                                                    |
| method      | string                  | 请求的HTTP方法，例如：'GET', 'POST'或其他HTTP方法            |
| body        | Object, FormData string | request body                                                 |
| params      | Object                  | 请求的URL参数对象                                            |
| headers     | Object                  | request header                                               |
| timeout     | number                  | 单位为毫秒的请求超时时间 (0 表示无超时时间)                  |
| before      | function(request)       | 请求发送前的处理函数，类似于jQuery的beforeSend函数           |
| progress    | function(event)         | ProgressEvent回调处理函数                                    |
| credentials | boolean                 | 表示跨域请求时是否需要使用凭证                               |
| emulateHTTP | boolean                 | 发送PUT, PATCH, DELETE请求时以HTTP POST的方式发送，并设置请求头的X-HTTP-Method-Override |
| emulateJSON | boolean                 | 将request body以application/x-www-form-urlencoded content type发送 |

emulateHTTP

```
如果Web服务器无法处理PUT, PATCH和DELETE这种REST风格的请求，你可以启用enulateHTTP现象。
启用该选项后，请求会以普通的POST方法发出，并且HTTP头信息的X-HTTP-Method-Override属性会设置为实际的HTTP方法。
Vue.http.options.emulateHTTP = true;
```

emulateJSON

```
如果Web服务器无法处理编码为application/json的请求，你可以启用emulateJSON选项。
启用该选项后，请求会以application/x-www-form-urlencoded作为MIME type，就像普通的HTML表单一样。
Vue.http.options.emulateJSON = true;
```

> response对象

response对象包含以下方法、属性

| 方法       | 类型    | 描述                                          |
| ---------- | ------- | --------------------------------------------- |
| text()     | string  | 以string形式返回response body                 |
| json()     | Object  | 以JSON对象形式返回response body               |
| blob()     | Blob    | 以二进制形式返回response body                 |
| 属性       | 类型    | 描述                                          |
| ok         | boolean | 响应的HTTP状态码在200~299之间时，该属性为true |
| status     | number  | 响应的HTTP状态码                              |
| statusText | string  | 响应的状态文本                                |
| headers    | Object  | 响应头                                        |

### 实例

> get

这段程序的then方法只提供了successCallback，而省略了errorCallback。
catch方法用于捕捉程序的异常，catch方法和errorCallback是不同的，errorCallback只在响应失败时调用，而catch则是在整个请求到响应过程中，只要程序出错了就会被调用。

```js
var demo = new Vue({
  el: '#app',
  data: {
    gridColumns: ['customerId', 'companyName', 'contactName', 'phone'],
    gridData: [],
    apiUrl: 'http://211.149.193.19:8080/api/customers'
  },
  ready: function() {
    this.getCustomers()
  },
  methods: {
    getCustomers: function() {
      this.$http.get(this.apiUrl)
        .then((response) => {
          this.$set('gridData', response.data)
        })
        .catch(function(response) {
          console.log(response)
        })
    }
  }
})
```

在then方法的回调函数内，你也可以直接使用this，this仍然是指向Vue实例的：

```js
getCustomers: function() {
  this.$http.get(this.apiUrl)
    .then((response) => {
      this.$set('gridData', response.data)
    })
    .catch(function(response) {
      console.log(response)
    })
}
```

> jsonp

```js
getCustomers: function() {
  this.$http.jsonp(this.apiUrl).then(function(response){
    this.$set('gridData', response.data)
  })
}
```

> post

```js
var demo = new Vue({
  el: '#app',
  data: {
    show: false,
    gridColumns: [{
      name: 'customerId',
      isKey: true
    }, {
      name: 'companyName'
    }, {
      name: 'contactName'
    }, {
      name: 'phone'
    }],
    gridData: [],
    apiUrl: 'http://211.149.193.19:8080/api/customers',
    item: {}
  },
  ready: function() {
    this.getCustomers()
  },
  methods: {
    closeDialog: function() {
      this.show = false
    },
    getCustomers: function() {
      var vm = this
      vm.$http.get(vm.apiUrl)
        .then((response) => {
          vm.$set('gridData', response.data)
        })
    },
    createCustomer: function() {
      var vm = this
      vm.$http.post(vm.apiUrl, vm.item)
        .then((response) => {
          vm.$set('item', {})
          vm.getCustomers()
        })
      this.show = false
    }
  }
})
```

> put

```js
updateCustomer: function() {
  var vm = this
  vm.$http.put(this.apiUrl + '/' + vm.item.customerId, vm.item)
    .then((response) => {
      vm.getCustomers()
    })
}
```

> delete

```js
deleteCustomer: function(customer){
  var vm = this
  vm.$http.delete(this.apiUrl + '/' + customer.customerId)
    .then((response) => {
      vm.getCustomers()
    })
}
```

vue-resource是一个非常轻量的用于处理HTTP请求的插件，它提供了两种方式来处理HTTP请求：
1、使用Vue.http或this.$http
2、使用Vue.resource或this.$resource

```js
data(){
    return{
      toplist:[],
      alllist:[]
    }
  },
  //vue-router
  route:{
    data({to}){
      //并发请求，利用 Promise 
      return Promise.all([
        //简写
        this.$http.get('http://192.168.30.235:9999/rest/knowledge/list',{'websiteId':2,'pageSize':5,'pageNo':1,'isTop':1}),
        //this.$http.get('http://192.168.30.235:9999/rest/knowledge/list',{'websiteId':2,'pageSize':20,'pageNo':1,'isTop':0})
        //不简写
        this.$http({
          method:'GET',
          url:'http://192.168.30.235:9999/rest/knowledge/list',
          data:{'websiteId':2,'pageSize':20,'pageNo':1,'isTop':0},
          headers: {"X-Requested-With": "XMLHttpRequest"},
          emulateJSON: true
          })
        ]).then(function(data){//es5写法
           return{
            toplist:data[0].data.knowledgeList,
            alllist:data[1].data.knowledgeList
          }
        //es6写法 .then()部分
        //.then(([toplist,alllist])=>({toplist,alllist})) 
      },function(error){
        //error
      })
    }
  }
```

## vue axios

vue2.0之后，就不再对vue-resource更新，而是推荐使用axios。基于 Promise 的 HTTP 请求客户端，可同时在浏览器和 Node.js 中使用。

> 特点

```
1、在浏览器中发送 XMLHttpRequests 请求
2、在 node.js 中发送 http请求
3、支持 Promise API
4、拦截请求和响应
5、转换请求和响应数据
6、取消请求
7、自动转换 JSON 数据
8、客户端支持保护安全免受 CSRF/XSRF 攻击
```

### 安装

> npm

```
// 安装
$ npm install axios
// 在要使用的文件中引入axios
import axios from 'axios'
```

> 单文件

```html
<script src="/static/js/axios.js"></script>
```

### 使用

> get

```js
// 向具有指定ID的用户发出请求
axios.get('/user?ID=12345')
.then(function (response) {
	console.log(response);
})
.catch(function (error) {
	console.log(error);
});
 
// 也可以通过 params 对象传递参数
axios.get('/user', {
	params: {
		ID: 12345
	}
})
.then(function (response) {
	console.log(response);
})
.catch(function (error) {
	console.log(error);
});
```

> post

```js
axios.post('/user', {
	firstName: 'Fred',
	lastName: 'Flintstone'
})
.then(function (response) {
	console.log(response);
})
.catch(function (error) {
	console.log(error);
});
```

> 多个并发

```js
function getUserAccount() {
	return axios.get('/user/12345');
}
 
function getUserPermissions() {
	return axios.get('/user/12345/permissions');
}
 
axios.all([getUserAccount(), getUserPermissions()])
.then(axios.spread(function (acct, perms) {
	//两个请求现已完成
}));
```

### API

> 常规使用

```js
axios(config)
// 发送一个 POST 请求
axios({
method: 'post',
url: '/user/12345',
data: {
firstName: 'Fred',
lastName: 'Flintstone'
}
});

axios(url[, config])
// 发送一个 GET 请求 (GET请求是默认请求模式)
axios('/user/12345');
```

> 请求方法别名

```js
//为了方便起见，已经为所有支持的请求方法提供了别名。
axios.request（config）
axios.get（url [，config]）
axios.delete（url [，config]）
axios.head（url [，config]）
axios.post（url [，data [，config]]）
axios.put（url [，data [，config]]）
axios.patch（url [，data [，config]]）
//注意:当使用别名方法时，不需要在config中指定url，method和data属性。
```

> 并发

帮助函数处理并发请求。

```
axios.all（iterable）
axios.spread（callback）
```

> 创建实例

也可以使用自定义配置创建axios的新实例。
`axios.create（[config]）`

```
var instance = axios.create({
baseURL: 'https://some-domain.com/api/',
timeout: 1000,
headers: {'X-Custom-Header': 'foobar'}
});
```

> 实例方法

可用的实例方法如下所示。 指定的配置将与实例配置合并

```
axios＃request（config）
axios＃get（url [，config]）
axios＃delete（url [，config]）
axios＃head（url [，config]）
axios＃post（url [，data [，config]]）
axios＃put（url [，data [，config]]）
axios＃patch（url [，data [，config]]）
```

> 请求配置

这些是用于发出请求的可用配置选项。 只有url是必需的。 如果未指定方法，请求将默认为GET

```js
{
// `url`是将用于请求的服务器URL
url: '/user',
 
// `method`是发出请求时使用的请求方法
method: 'get', // 默认
 
// `baseURL`将被添加到`url`前面，除非`url`是绝对的。
// 可以方便地为 axios 的实例设置`baseURL`，以便将相对 URL 传递给该实例的方法。
baseURL: 'https://some-domain.com/api/',
 
// `transformRequest`允许在请求数据发送到服务器之前对其进行更改
// 这只适用于请求方法'PUT'，'POST'和'PATCH'
// 数组中的最后一个函数必须返回一个字符串，一个 ArrayBuffer或一个 Stream
 
transformRequest: [function (data) {
// 做任何你想要的数据转换
 
return data;
}],
 
// `transformResponse`允许在 then / catch之前对响应数据进行更改
transformResponse: [function (data) {
// Do whatever you want to transform the data
 
return data;
}],
 
// `headers`是要发送的自定义 headers
headers: {'X-Requested-With': 'XMLHttpRequest'},
 
// `params`是要与请求一起发送的URL参数
// 必须是纯对象或URLSearchParams对象
params: {
ID: 12345
},
 
// `paramsSerializer`是一个可选的函数，负责序列化`params`
// (e.g. https://www.npmjs.com/package/qs, http://api.jquery.com/jquery.param/)
paramsSerializer: function(params) {
return Qs.stringify(params, {arrayFormat: 'brackets'})
},
 
// `data`是要作为请求主体发送的数据
// 仅适用于请求方法“PUT”，“POST”和“PATCH”
// 当没有设置`transformRequest`时，必须是以下类型之一：
// - string, plain object, ArrayBuffer, ArrayBufferView, URLSearchParams
// - Browser only: FormData, File, Blob
// - Node only: Stream
data: {
firstName: 'Fred'
},
 
// `timeout`指定请求超时之前的毫秒数。
// 如果请求的时间超过'timeout'，请求将被中止。
timeout: 1000,
 
// `withCredentials`指示是否跨站点访问控制请求
// should be made using credentials
withCredentials: false, // default
 
// `adapter'允许自定义处理请求，这使得测试更容易。
// 返回一个promise并提供一个有效的响应（参见[response docs]（＃response-api））
adapter: function (config) {
/* ... */
},
 
// `auth'表示应该使用 HTTP 基本认证，并提供凭据。
// 这将设置一个`Authorization'头，覆盖任何现有的`Authorization'自定义头，使用`headers`设置。
auth: {
username: 'janedoe',
password: 's00pers3cret'
},
 
// “responseType”表示服务器将响应的数据类型
// 包括 'arraybuffer', 'blob', 'document', 'json', 'text', 'stream'
responseType: 'json', // default
 
//`xsrfCookieName`是要用作 xsrf 令牌的值的cookie的名称
xsrfCookieName: 'XSRF-TOKEN', // default
 
// `xsrfHeaderName`是携带xsrf令牌值的http头的名称
xsrfHeaderName: 'X-XSRF-TOKEN', // default
 
// `onUploadProgress`允许处理上传的进度事件
onUploadProgress: function (progressEvent) {
// 使用本地 progress 事件做任何你想要做的
},
 
// `onDownloadProgress`允许处理下载的进度事件
onDownloadProgress: function (progressEvent) {
// Do whatever you want with the native progress event
},
 
// `maxContentLength`定义允许的http响应内容的最大大小
maxContentLength: 2000,
 
// `validateStatus`定义是否解析或拒绝给定的promise
// HTTP响应状态码。如果`validateStatus`返回`true`（或被设置为`null` promise将被解析;否则，promise将被
  // 拒绝。
validateStatus: function (status) {
return status >= 200 && status < 300; // default
},
 
// `maxRedirects`定义在node.js中要遵循的重定向的最大数量。
// 如果设置为0，则不会遵循重定向。
maxRedirects: 5, // 默认
 
// `httpAgent`和`httpsAgent`用于定义在node.js中分别执行http和https请求时使用的自定义代理。
// 允许配置类似`keepAlive`的选项，
// 默认情况下不启用。
httpAgent: new http.Agent({ keepAlive: true }),
httpsAgent: new https.Agent({ keepAlive: true }),
 
// 'proxy'定义代理服务器的主机名和端口
// `auth`表示HTTP Basic auth应该用于连接到代理，并提供credentials。
// 这将设置一个`Proxy-Authorization` header，覆盖任何使用`headers`设置的现有的`Proxy-Authorization` 自定义 headers。
proxy: {
host: '127.0.0.1',
port: 9000,
auth: : {
username: 'mikeymike',
password: 'rapunz3l'
}
},
 
// “cancelToken”指定可用于取消请求的取消令牌
// (see Cancellation section below for details)
cancelToken: new CancelToken(function (cancel) {
})
}
```

使用 then 时，将收到如下响应

```js
axios.get('/user/12345')
.then(function(response) {
console.log(response.data);
console.log(response.status);
console.log(response.statusText);
console.log(response.headers);
console.log(response.config);
});
```

> 配置默认值

全局axios默认值

```js
axios.defaults.baseURL = 'https://api.example.com';
axios.defaults.headers.common['Authorization'] = AUTH_TOKEN;
axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded';
```

自定义实例默认值

```js
//在创建实例时设置配置默认值
var instance = axios.create（{
   baseURL：'https://api.example.com'
}）;
//在实例创建后改变默认值
instance.defaults.headers.common ['Authorization'] = AUTH_TOKEN;
```

配置优先级顺序

```js
// 配置将与优先顺序合并。 顺序是lib / defaults.js中的库默认值，然后是实例的defaults属性，最后是请求的config参数。 后者将优先于前者。 这里有一个例子。

//使用库提供的配置默认值创建实例
//此时，超时配置值为`0`，这是库的默认值
var instance = axios.create（）;
 
//覆盖库的超时默认值
//现在所有请求将在超时前等待2.5秒
instance.defaults.timeout = 2500;
 
//覆盖此请求的超时，因为它知道需要很长时间
instance.get（'/ longRequest'，{
   timeout：5000
}）;
```

> 拦截器

可以截取请求或响应在被 then 或者 catch 处理之前

```js
//添加请求拦截器
axios.interceptors.request.use（function（config）{
     //在发送请求之前做某事
     return config;
   }，function（error）{
     //请求错误时做些事
     return Promise.reject（error）;
   }）;
 
//添加响应拦截器
axios.interceptors.response.use（function（response）{
     //对响应数据做些事
     return response;
   }，function（error）{
     //请求错误时做些事
     return Promise.reject（error）;
   }）;
```

如果你以后可能需要删除拦截器。

```js
var myInterceptor = axios.interceptors.request.use(function () {/*...*/});
axios.interceptors.request.eject(myInterceptor);
```

你可以将拦截器添加到axios的自定义实例

```js
var instance = axios.create();
instance.interceptors.request.use(function () {/*...*/});
```

>处理错误

```
axios.get（'/ user / 12345'）
   .catch（function（error）{
     if（error.response）{
       //请求已发出，但服务器使用状态代码进行响应
       //落在2xx的范围之外
       console.log（error.response.data）;
       console.log（error.response.status）;
       console.log（error.response.headers）;
     } else {
       //在设置触发错误的请求时发生了错误
       console.log（'Error'，error.message）;
     }}
     console.log（error.config）;
   }）;
```

您可以使用validateStatus配置选项定义自定义HTTP状态码错误范围。

```js
axios.get（'/ user / 12345'，{
   validateStatus：function（status）{
     return status < 500; //仅当状态代码大于或等于500时拒绝
   }}
}）
```

> 消除

可以使用取消令牌取消请求

axios cancel token API基于可取消的promise提议，目前处于阶段1，可以使用CancelToken.source工厂创建一个取消令牌，如下所示

```js
var CancelToken = axios.CancelToken;
var source = CancelToken.source（）;
 
axios.get('/user/12345', {
cancelToken: source.token
}).catch(function(thrown) {
if (axios.isCancel(thrown)) {
console.log('Request canceled', thrown.message);
} else {
// 处理错误
}
});
 
//取消请求（消息参数是可选的）
source.cancel（'操作被用户取消。'）;
```

还可以通过将执行器函数传递给CancelToken构造函数来创建取消令牌

```js
var CancelToken = axios.CancelToken;
var cancel;
 
axios.get（'/ user / 12345'，{
   cancelToken：new CancelToken（function executor（c）{
     //一个执行器函数接收一个取消函数作为参数
     cancel = c;
   }）
}）;
 
// 取消请求
clear();
```

注意：您可以使用相同的取消令牌取消几个请求

> 使用application / x-www-form-urlencoded格式

默认情况下，axios将JavaScript对象序列化为JSON。 要以应用程序/ x-www-form-urlencoded格式发送数据，您可以使用以下选项之一

浏览器中

```js
// 在浏览器中，您可以使用URLSearchParams API
var params = new URLSearchParams();
params.append('param1', 'value1');
params.append('param2', 'value2');
axios.post('/foo', params);
// 请注意，所有浏览器都不支持URLSearchParams，但是有一个polyfill可用（确保polyfill全局环境）。

// 或者，您可以使用qs库对数据进行编码
var qs = require('qs');
axios.post('/foo', qs.stringify({ 'bar': 123 });
```

Node.js

```js
// 在node.js中，可以使用querystring模块
var querystring = require('querystring');
axios.post('http://something.com/', querystring.stringify({ foo: 'bar' });
```

TypeScript

```
// axios包括TypeScript定义
import axios from 'axios';
axios.get('/user?ID=12345');
```

## 响应接口

Vue 可以添加数据动态响应接口。

通过使用 $watch 属性来实现数据的监听，$watch 必须添加在 Vue 实例之外才能实现正确的响应。

```html
// 实例中通过点击按钮自动加 1。setTimeout 设置两秒后计算器的值加上 20 
<div id = "app">
    <p style = "font-size:25px;">计数器: {{ counter }}</p>
    <button @click = "counter++" style = "font-size:25px;">点我</button>
</div>
<script type = "text/javascript">
var vm = new Vue({
    el: '#app',
    data: {
        counter: 1
    }
});
vm.$watch('counter', function(nval, oval) {
    alert('计数器值的变化 :' + oval + ' 变为 ' + nval + '!');
});
setTimeout(
    function(){
        vm.counter = 20;
    },2000
);
</script>
```

Vue 不允许在已经创建的实例上动态添加新的根级响应式属性。

Vue 不能检测到对象属性的添加或删除，最好的方式就是在初始化实例前声明根级响应式属性，哪怕只是一个空值。

如果我们需要在运行过程中实现属性的添加或删除，则可以使用全局 Vue，Vue.set 和 Vue.delete 方法。

### Vue.set

Vue.set 方法用于设置对象的属性，它可以解决 Vue 无法检测添加属性的限制

```js
// 语法格式
Vue.set( target, key, value )

// 参数说明
target: 可以是对象或数组
key : 可以是字符串或数字
value: 可以是任何类型
```

 实例

```js
<div id = "app">
   <p style = "font-size:25px;">计数器: {{ products.id }}</p>
   <button @click = "products.id++" style = "font-size:25px;">点我</button>
</div>
<script type = "text/javascript">
	// 创建了一个变量 myproduct,
	var myproduct = {"id":1, name:"book", "price":"20.00"};
   var vm = new Vue({
   el: '#app',
   data: {
      counter: 1,
      // myproduct变量在赋值给了 Vue 实例的 data 对象
      products: myproduct
   }
});
// myproduct数组添加一个或多个属性
vm.products.qty = "1";
console.log(vm);
vm.$watch('counter', function(nval, oval) {
   alert('计数器值的变化 :' + oval + ' 变为 ' + nval + '!');
});
</script>
```

在产品中添加了数量属性 qty，但是 get/set 方法只可用于 id，name 和 price 属性，却不能在 qty 属性中使用。不能通过添加 Vue 对象来实现响应。 Vue 主要在开始时创建所有属性。

 如果我们要实现这个功能，可以通过 Vue.set 来实现

```js
<div id = "app">
<p style = "font-size:25px;">计数器: {{ products.id }}</p>
<button @click = "products.id++" style = "font-size:25px;">点我</button>
</div>
<script type = "text/javascript">
var myproduct = {"id":1, name:"book", "price":"20.00"};
var vm = new Vue({
   el: '#app',
   data: {
      counter: 1,
      products: myproduct
   }
});
Vue.set(myproduct, 'qty', 1);
console.log(vm);
vm.$watch('counter', function(nval, oval) {
   alert('计数器值的变化 :' + oval + ' 变为 ' + nval + '!');
});
</script>
```

### Vue.delete

Vue.delete 用于删除动态添加的属性

```js
// 语法格式
Vue.delete( target, key )

// 参数说明
target: 可以是对象或数组
key : 可以是字符串或数字
```

实例

```html
<div id = "app">
   <p style = "font-size:25px;">计数器: {{ products.id }}</p>
   <button @click = "products.id++" style = "font-size:25px;">点我</button>
</div>
<script type = "text/javascript">
var myproduct = {"id":1, name:"book", "price":"20.00"};
var vm = new Vue({
   el: '#app',
   data: {
      counter: 1,
      products: myproduct
   }
});
Vue.delete(myproduct, 'price');
console.log(vm);
vm.$watch('counter', function(nval, oval) {
   alert('计数器值的变化 :' + oval + ' 变为 ' + nval + '!');
});
</script>
```



