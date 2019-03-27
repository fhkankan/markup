# Vue.js概述

Vue.js（读音 /vjuː/, 类似于 view） 是一套构建用户界面的渐进式框架。

Vue 只关注视图层， 采用自底向上增量开发的设计。

Vue 的目标是通过尽可能简单的 API 实现响应的数据绑定和组合的视图组件。

## 安装

- 独立版本

```
在 Vue.js 的官网上直接下载 vue.min.js 并用 <script> 标签引入。
<script src="/static/js/vue.js"></script>
```

- 网络版本

```
BootCDN（国内） : https://cdn.bootcss.com/vue/2.2.2/vue.min.js

unpkg：https://unpkg.com/vue/dist/vue.js, 会保持和 npm 发布的最新的版本一致。

cdnjs : https://cdnjs.cloudflare.com/ajax/libs/vue/2.1.8/vue.min.js
```

- NPM方法

> 安装Node版本管理工具

```
// 版本管理工具
brew install nvm
nvm ls-remote // 查看可安装版本
nvm ls // 列出可安装版本
nvm install xxx // 安装可用版本
nvm use xxx // 切换版本
nvm run 4.2.2 --version // 直接运行特定版本的 Node
nvm exec 4.2.2 node --version //在当前终端的子进程中运行特定版本的 Node
nvm which 4.2.2 //确认某个版本Node的路径

//从特定版本导入到我们将要安装的新版本 Node：
nvm install v5.0.0 --reinstall-packages-from=4.2
```

> 安装vue

```
// 安装node.js
brew install node
// 使用淘宝镜像
$ npm install -g cnpm --registry=https://registry.npm.taobao.org

// 安装vue
$ cnpm install vue
```

## 命令行

Vue.js 提供一个官方命令行工具，可用于快速搭建大型单页应用。

```
# 全局安装 vue-cli
$ cnpm install --global vue-cli
# 创建一个基于 webpack 模板的新项目(项目名不能大写)
$ vue init webpack my-project
# 这里需要进行一些配置，默认回车即可
...
```

进入项目，安装并运行

```
$ cd my-project
$ cnpm install
$ cnpm run dev
```

## 目录结构

使用npm安装项目，则目录结构汇总包含如下

| 目录/文件    | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| build        | 项目构建(webpack)相关代码                                    |
| config       | 配置目录，包括端口号等                                       |
| node_modules | npm加载的项目依赖模块                                        |
| src          | 开发目录，包括：<br>assets:放置一些图片，如logo等<br>components:目录里面放了一个组件文件，可以不用<br>App.vue：项目入口文件，可以直接将组件写在这里，二不适用components目录<br>main.js:项目的核心文件 |
| static       | 静态资源目录，如图片、字体等                                 |
| test         | 初始测试目录，可删除                                         |
| .xxx文件     | 配置文件，包括语法配置，git配置等                            |
| index.html   | 首页入口文件，可添加一些meta信息或统计代码                   |
| package.json | 项目配置文件                                                 |
| README.md    | 项目说明文档                                                 |

## 起步

每个Vue应用都需要通过实例化Vue来实现

当一个 Vue 实例被创建时，它向 Vue 的响应式系统中加入了其 data 对象中能找到的所有的属性。当这些属性的值发生改变时，html 视图将也会产生相应的变化。

```
var vm = new Vue({
    //选项
})
```

> 构造器

```html
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
	<script src="https://cdn.staticfile.org/vue/2.4.2/vue.min.js"></script>
</head>
<body>
	<div id="vue_det">
		<h1>site : {{site}}</h1>
		<h1>url : {{url}}</h1>
		<h1>{{details()}}</h1>
	</div>
	<script type="text/javascript">
		var vm = new Vue({
			el: '#vue_det',
			data: {
				site: "菜鸟教程",
				url: "www.runoob.com",
				alexa: "10000"
			},
			methods: {
				details: function() {
					return  this.site + " - 学的不仅是技术，更是梦想！";
				}
			}
		})
	</script>
```

构造器参数

```
el
# 是DOM元素中的id,表示接下来的改动全部在以上指定的div内，div外部不受影响
data
# 用于定义属性
methods
# 用于定义的函数，可以通过return来返回函数值
{{}}
# 用于输出对象属性和函数返回值
```

> 其他属性

除了数据属性，Vue 实例还提供了一些有用的实例属性与方法。它们都有前缀 $，以便与用户定义的属性区分开来

```html
<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<title>Vue 测试实例 - 菜鸟教程(runoob.com)</title>
	<script src="https://cdn.staticfile.org/vue/2.4.2/vue.min.js"></script>
</head>
<body>
	<div id="vue_det">
		<h1>site : {{site}}</h1>
		<h1>url : {{url}}</h1>
		<h1>Alexa : {{alexa}}</h1>
	</div>
	<script type="text/javascript">
	// 我们的数据对象
	var data = { site: "菜鸟教程", url: "www.runoob.com", alexa: 10000}
	var vm = new Vue({
		el: '#vue_det',
		data: data
	})
	// vue实例的属性与方法
	document.write(vm.$data === data) // true
	document.write("<br>") // true
	document.write(vm.$el === document.getElementById('vue_det')) // true
	document.write("<br>") // true
	// 它们引用相同的对象！
	document.write(vm.site === data.site) // true
	document.write("<br>")
	// 设置属性也会影响到原始数据
	vm.site = "Runoob"
	document.write(data.site + "<br>") // Runoob
 
	// ……反之亦然
	data.alexa = 1234
	document.write(vm.alexa) // 1234	
	</script>
</body>
</html>
```





