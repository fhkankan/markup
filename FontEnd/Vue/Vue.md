# Vue.js概述

Vue.js（读音 /vjuː/, 类似于 view） 是一套构建用户界面的渐进式框架。

Vue 只关注视图层， 采用自底向上增量开发的设计。

Vue 的目标是通过尽可能简单的 API 实现响应的数据绑定和组合的视图组件。

## 安装

- 独立版本

```
在 Vue.js 的官网上直接下载 vue.min.js 并用 <script> 标签引入。
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
# 创建一个基于 webpack 模板的新项目
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

