# ApiDoc

## 概述

apidoc是一款可以有源代码中的注释直接自动生成api接口文档的工具，它几乎支持目前主流的所有风格的注释。可在C#, Go, Dart, Java, JavaScript, PHP, TypeScript等语言中使用)

## 安装

```
1. 安装node.js
brew install node

2. 安装apiDoc
npm install apidoc -g
```

## 运行

```
apidoc -i myapp/ -o apidoc/ -t mytemplate/
```

命令参数

| 参数                  | 描述                                                         |
| --------------------- | ------------------------------------------------------------ |
| -h, --help            | 查看帮助文档                                                 |
| -f, --file-filters    | 指定读取文件的文件名过滤正则表达式(可指定多个) <br/>例如: `apidoc -f ".*\\.js$" -f ".*\\.ts$"` 意为只读取后缀名为js和ts的文件 默认值:`.clj .cls .coffee .cpp .cs .dart .erl .exs?`  `.go .groovy .ino? .java .js .jsx .kt .litcoffee lua .p .php? .pl .pm .py .rb .scala .ts .vue` |
| -e, --exclude-filters | 指定不读取的文件名过滤正则表达式(可指定多个)<br/>例如:`apidoc -e ".*\\.js$"` 意为不读取后缀名为js的文件
默认:`''` |
| -i, --input           | 指定读取源文件的目录<br/>例如：`apidoc -i myapp/` 意为读取`myapp/`目录下面的源文件
默认值:`./` |
| -o, --output          | 指定输出文档的目录<br/> 例如：`apidoc -o doc/` 意为输出文档到`doc`目录下 默认值:`./doc/` |
| -t, --template        | 指定输出的模板文件<br/>例如:`apidoc -t mytemplate/`
默认:`path.join(__dirname, '../template/')(使用默认模板)` |
| -c, --config          | 指定包含配置文件(apidoc.json)的目录<br/>例如:`apidoc -c config/`
默认:`./` |
| -p, --private         | 输出的文档中是否包含私有api<br/>例如:`apidoc -p true` 
默认:`false` |
| -v, --verbose         | 是否输出详细的debug信息<br/>例如:`apidoc -v true`
默认:`false` |

