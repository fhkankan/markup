# Node简介

简单的说 Node.js 就是运行在服务端的 JavaScript。

Node.js 是一个基于Chrome JavaScript 运行时建立的一个平台。

Node.js是一个事件驱动I/O服务端JavaScript环境，基于Google的V8引擎，V8引擎执行Javascript的速度非常快，性能非常好。

检查版本

```
node -v
```

简单使用

```javascript
// 1.脚本模式
// 保存如下内容至helloworld.js中
console.log("hello world");
// 在命令行通过node命令执行
$ node helloworld.js

// 2.交互模式
$ node
> console.log('hello world')
```

## 安装

### nodejs

方法一：官网，适用于各平台

[官网](https://nodejs.org/en/)

官网下载安装包，执行安装

方法二：包管理器

ubuntu

```
sudo apt-get install nodejs-legacy nodejs
sudo apt-get install npm
```

mac

```
brew install node
```

### npm

NPM是随同NodeJS一起安装的包管理工具，能解决NodeJS代码部署上的很多问题，常见的使用场景有以下几种：

```
- 允许用户从NPM服务器下载别人编写的第三方包到本地使用。
- 允许用户从NPM服务器下载并安装别人编写的命令行程序到本地使用。
- 允许用户将自己编写的包或命令行程序上传到NPM服务器供别人使用。
```

新版的nodejs已经集成了npm

```
# 检查版本
node -v
npm -v
```

- 常用命令

本地安装

将安装包放在 ./node_modules 下（运行 npm 命令时所在的目录），如果没有 node_modules 目录，会在当前执行 npm 命令的目录下生成 node_modules 目录。

可以通过 require() 来引入本地安装的包。

```javascript
$ npm install [name]
```

全局安装

将安装包放在 /usr/local 下或者你 node 的安装目录。

可以直接在命令行里使用

```
$ npm install [name] -g
```

查看版本信息

```
npm -v
```

查看安装包信息

```javascript
// 全局安装
npm list -g
// 本地安装
npm list
// 某个模块
npm list [name]
// 查看在层级0下的安装模块
npm list --depth 0
```

卸载

```
npm uninstall name
```

清空本地缓存

```
npm cache clear
```

更新

```
npm update name
```

搜索

```
npm search name
```

创建模块

```javascript
// 1.用npm生成package.json,填写相应信息
npm init
// 2.在npm资源库中注册用户(使用邮箱)
npm adduser
```

发布

```
npm publish
```

帮助

```
npm help
```

- 使用package.json

package.json 位于模块的目录下，用于定义包的属性。

```
name - 包名。
version - 包的版本号。
description - 包的描述。
homepage - 包的官网 url 。
author - 包的作者姓名。
contributors - 包的其他贡献者姓名。
dependencies - 依赖包列表。如果依赖包没有安装，npm 会自动将依赖包安装在 node_module 目录下。
repository - 包代码存放的地方的类型，可以是 git 或 svn，git 可在 Github 上。
main - main 字段指定了程序的主入口文件，require('moduleName') 就会加载这个文件。这个字段的默认值是模块根目录下面的 index.js。
keywords - 关键字
```

### cnpm

由于国内npm速度过慢，采用[cnpm](http://npm.taobao.org)替代

- 替换

你可以使用我们定制的 cnpm (gzip 压缩支持) 命令行工具代替默认的 npm:

```
$ npm install -g cnpm --registry=https://registry.npm.taobao.org
```

或者你直接通过添加 npm 参数 alias 一个新命令:

```shell
# 1.拷贝命令至～/.zshrc
#alias for cnpm
alias cnpm="npm --registry=https://registry.npm.taobao.org \
--cache=$HOME/.npm/.cache/cnpm \
--disturl=https://npm.taobao.org/dist \
--userconfig=$HOME/.cnpmrc"
# 2.更新加载
source ~/.zshrc
```

- 使用

> 安装模块

从 [registry.npm.taobao.org](http://registry.npm.taobao.org/) 安装所有模块. 当安装的时候发现安装的模块还没有同步过来, 淘宝 NPM 会自动在后台进行同步, 并且会让你从官方 NPM [registry.npmjs.org](http://registry.npmjs.org/) 进行安装. 下次你再安装这个模块的时候, 就会直接从 淘宝 NPM 安装了.

```javascript
$ cnpm install [name]
```

> 同步模块

直接通过 `sync` 命令马上同步一个模块, 只有 `cnpm` 命令行才有此功能:

```
$ cnpm sync connect
```

当然, 你可以直接通过 web 方式来同步: [/sync/connect](http://npm.taobao.org/sync/connect)

```
$ open https://npm.taobao.org/sync/connect
```

> 其他命令

支持 `npm` 除了 `publish` 之外的所有命令, 如:

```
$ cnpm info connect
```

### nvm

在一台机器上进行多个node版本之间切换管理

- 安装

> mac

安装包

```shell
# 方法一：
brew install nvm
# 方法二
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | bash
或
wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
```

配置环境变量

```shell
mkdir .nvm
vim ~/.bash_profile, ~/.zshrc, ~/.profile, or ~/.bashrc
# 添加
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
```

- 使用

安装node

```shell
nvm install node  # 最新版本
nvm install stable  # 安装稳定版本
nvm install 6.14.4  # 特定版本
```

展示列表

```shell
nvm ls-remote  # 展示远程版本列表
nvm ls  # 展示本地列表
```

切换版本

```shell
nvm use node  # 使用最新版本
nvm use 6.14.4  # 使用特定版本
nvm run 6.14.4 --version  # 运行特定版本
nvm exec 6.14.4 node --version  # 在当前终端子进程中运行特定版本
```

确认路径

```shell
nvm which 7.3.0
```

别名

```shell
# 设置别名
nvm alias latest 11.1.0
nvm alias default 11.1.0
# 使用别名
nvm user latest
# 取消别名
nvm unalias latest
```

卸载

```
nvm uninstall 11.1.0  # 卸载指定版本
```

## 创建应用

使用 Node.js 时，我们不仅仅在实现一个应用，同时还实现了整个 HTTP 服务器。

创建一个应用需要如下几步

- 引入required模块

使用require指令载入http模块，并将实例化的HTTP赋值给变量http

```javascript
var http = require('http');
```

- 创建服务器

服务器可以监听客户端的请求，类似于 Apache 、Nginx 等 HTTP 服务器

使用`http.createServer()`方法创建服务器，并使用 listen 方法绑定 8888 端口。 函数通过 request, response 参数来接收和响应数据。

如：在你项目的根目录下创建一个叫 server.js 的文件，并写入以下代码完成一个http服务器

```javascript
// 请求（require）Node.js 自带的 http 模块，并且把它赋值给 http 变量
var http = require('http');
// createServer会返回一个对象，这个对象的isten方法通过传参指定这个 HTTP 服务器监听的端口号。
http.createServer(function (request, response) {
    // 发送 HTTP 头部 
    // HTTP 状态值: 200 : OK
    // 内容类型: text/plain
    response.writeHead(200, {'Content-Type': 'text/plain'});
    // 发送响应数据 "Hello World"
    response.end('Hello World\n');
}).listen(8888);

// 终端打印如下信息
console.log('Server running at http://127.0.0.1:8888/');
```

使用node命令执行以上代码

```javascript
node server.js
```

## REPL

Node.js REPL(Read Eval Print Loop:交互式解释器) 表示一个电脑的环境，类似 Window 系统的终端或 Unix/Linux shell，我们可以在终端中输入命令，并接收系统的响应。

Node 自带了交互式解释器，可以执行以下任务：

```
- 读取  读取用户输入，解析输入了Javascript 数据结构并存储在内存中。
- 执行  执行输入的数据结构
- 打印  输出结果
- 循环  循环操作以上步骤直到用户两次按下 **ctrl-c** 按钮退出。
```

- 启动

```
$ node
>
```

- 使用

表达式运算

```
$ node
> 1 +4
5
> 5 / 2
2.5
```

使用变量

```
$ node
> x = 10
10
> var y = 10
undefined
> x + y
20
> console.log("Hello World")
Hello World
undefined
```

多行表达式

```
$ node
> var x = 0
undefined
> do {
... x++;
... console.log("x: " + x);
... } while ( x < 5 );
x: 1
x: 2
x: 3
x: 4
x: 5
undefined
>
```

下划线变量

```
$ node
> var x = 10
undefined
> var y =10
undefined
> x + y
20
> var sum = _
undefined
> console.log(sum)
20
undefined
```

- 命令

```
ctrl + c - 退出当前终端。

ctrl + c 按下两次 - 退出 Node REPL。

ctrl + d - 退出 Node REPL.

向上/向下 键 - 查看输入的历史命令

tab 键 - 列出当前命令

.help - 列出使用命令

.break - 退出多行表达式

.clear - 退出多行表达式

.save filename - 保存当前的 Node REPL 会话到指定文件

.load filename - 载入当前 Node REPL 会话的文件内容。
```

