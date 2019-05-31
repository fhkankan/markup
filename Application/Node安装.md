# 安装

## nodejs

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

## npm

NPM是随同NodeJS一起安装的包管理工具，能解决NodeJS代码部署上的很多问题，常见的使用场景有以下几种：

- 允许用户从NPM服务器下载别人编写的第三方包到本地使用。
- 允许用户从NPM服务器下载并安装别人编写的命令行程序到本地使用。
- 允许用户将自己编写的包或命令行程序上传到NPM服务器供别人使用。

新版的nodejs已经集成了npm

```
# 检查版本
node -v
npm -v
```

# cnpm

由于国内npm速度过慢，采用[cnpm](http://npm.taobao.org)替代

## 替换

你可以使用我们定制的 cnpm (gzip 压缩支持) 命令行工具代替默认的 npm:

```
$ npm install -g cnpm --registry=https://registry.npm.taobao.org
```

或者你直接通过添加 npm 参数 alias 一个新命令:

```shell
alias cnpm="npm --registry=https://registry.npm.taobao.org \
--cache=$HOME/.npm/.cache/cnpm \
--disturl=https://npm.taobao.org/dist \
--userconfig=$HOME/.cnpmrc"

# Or alias it in .bashrc or .zshrc
$ echo '\n#alias for cnpm\nalias cnpm="npm --registry=https://registry.npm.taobao.org \
  --cache=$HOME/.npm/.cache/cnpm \
  --disturl=https://npm.taobao.org/dist \
  --userconfig=$HOME/.cnpmrc"' >> ~/.zshrc && source ~/.zshrc
```

## 使用

- 安装模块

从 [registry.npm.taobao.org](http://registry.npm.taobao.org/) 安装所有模块. 当安装的时候发现安装的模块还没有同步过来, 淘宝 NPM 会自动在后台进行同步, 并且会让你从官方 NPM [registry.npmjs.org](http://registry.npmjs.org/) 进行安装. 下次你再安装这个模块的时候, 就会直接从 淘宝 NPM 安装了.

```
$ cnpm install [name]
```

- 同步模块

直接通过 `sync` 命令马上同步一个模块, 只有 `cnpm` 命令行才有此功能:

```
$ cnpm sync connect
```

当然, 你可以直接通过 web 方式来同步: [/sync/connect](http://npm.taobao.org/sync/connect)

```
$ open https://npm.taobao.org/sync/connect
```

- 其他命令

支持 `npm` 除了 `publish` 之外的所有命令, 如:

```
$ cnpm info connect
```

# nvm

在一台机器上进行多个node版本之间切换管理

## 安装

- mac

安装包

```shell
# 方法一：
brew install nvm
# 方法二
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
或
wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.34.0/install.sh | bash
```

配置环境变量

```shell
mkdir .nvm
vim ~/.bash_profile
# 添加
export NVM_DIR="$HOME/.nvm"
  [ -s "/usr/local/opt/nvm/nvm.sh" ] && . "/usr/local/opt/nvm/nvm.sh"  # This loads nvm
  [ -s "/usr/local/opt/nvm/etc/bash_completion" ] && . "/usr/local/opt/nvm/etc/bash_completion"  # This loads nvm bash_completio
```

## 使用

安装node

```shell
nvm install node  # 最新版本
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
# 使用别名
nvm user latest
# 取消别名
nvm unalias latest
```

