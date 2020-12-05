# gitbook+hexo
前置条件
- Node.js
- Git

## gitbook

### 安装

```shell
npm install -g gitbook-cli
```

### 命令

````shell
gitbook build [options] [source_dir]  # 编译指定目录，输出Web格式(_book文件夹中)
gitbook serve [options] [source_dir]  # 监听文件变化并编译指定目录，同时会创建一个服务器用于预览Web，默认http://localhost:4000/
gitbook pdf [options] [source_dir]  # 编译指定目录，输出PDF
gitbook epub [options] [source_dir]  # 编译指定目录，输出epub
gitbook mobi [options] [source_dir]  # 编译指定目录，输出mobi
gitbook init [source_dir]  # 通过SUMMARY.md生成作品目录
````

## Hexo

[官方文档](https://hexo.io/zh-cn/docs/)

### 安装

全局安装安装

```shell
npm install -g hexo-cli
```

局部安装

```shell
npm install hexo

# 配置命令行，则可使用hexo [command]
echo 'PATH="$PATH:./node_modules/.bin"' >> ~/.profile
# 不配置命令行，需使用npx hexo [command]
```

### 建站

在指定文件夹中新建所需要的文件。

```shell
hexo init <folder>  # 初始化
cd <folder>
npm install
```

新建完成后，指定文件夹的目录如下：

```
.
├── _config.yml  # 博客的配置文件
├── package.json  # 应用程序的信息
├── scaffolds  # 模版文件夹
├── source  # 资源文件夹
|   ├── _drafts
|   └── _posts
└── themes  # 主题文件夹
```

相关命令

```shell
hexo new [layout] <title>  # 新建一篇文章
hexo g  # 生成静态文件
hexo publish [layout] <filename>  # 发表草稿
hexo server # 启动服务器，默认http://localhost:4000/
hexo d  # 部署网站
hexo render <file1> [file2] ...  # 渲染文件
hexo migrate <type>  # 从其他博客系统 迁移内容
hexo clean  # 清除缓存文件 (db.json) 和已生成的静态文件 (public)
hexo list <type>  # 列出网站资料
hexo version  # 显示hexo版本
```

