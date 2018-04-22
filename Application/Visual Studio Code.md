# Visual Studio Code（python）
## 插件安装

python

```
- command + P 打开命令输入界面
- 输入ext install python 安装python插件
```

安装配置flake8（自动错误检查工具）

```
- python环境中安装flake8    pip install flake8   
- 用户-首选项-工作区设置中修改配置（用户设置也可以）  "python.linting.flake8Enabled": true

```

安装配置yapf（自动格式化代码工具）

```
- python环境安装yapf    pip install yapf
- 用户-首选项-工作区设置中修改配置（用户设置也可以）  "python.formatting.provider": "yapf"
- Command + shift + F 格式化代码

```

配置Command + Shift + B 运行代码

```
打开或新建一个python源文件，按下快捷键Ctrl+Shift+B运行，VSC会提示No task runner configured.，点击“Configure Task Runner”，选择“Others”，输入以下内容并保存：
```

```
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "0.1.0",
    "command": "${workspaceRoot}/venv/bin/python",
    "isShellCommand": true,
    "args": ["${file}"],
    "showOutput": "always"
}
```
安装Linting（代码格式检查工具）

```
- python环境中安装linting    pip install pylint
- 安装完python插件后pylint是默认开启的
```

配置显示空格字符

```
用户-首选项-工作区设置中修改配置（用户设置也可以）"editor.renderWhitespace": "all"
```

配置忽略非代码文件显示

```
用户-首选项-工作区设置中修改配置（用户设置也可以）
"files.exclude":{
"**/.git": true,
"**/.svn": true,
"**/.hg": true,
"**/.DS_Store": true,
"*/.pyc":true
}
```

完整工作区配置文件
```
1. //将设置放入此文件中以覆盖默认值和用户设置。
2. {
3. "python.pythonPath":"${workspaceRoot}/venv/bin/python",
4. "editor.renderWhitespace":"all",
5. "python.linting.pylintEnabled": false,
6. "python.linting.flake8Enabled": true,
7. "python.formatting.provider":"yapf",
8. //配置 glob 模式以排除文件和文件夹。
9. "files.exclude":{
10."**/.git": true,
11."**/.svn": true,
12."**/.hg": true,
13."**/.DS_Store": true,
14."**/*.pyc":true
15.}
16.}
```
配置代码片段

```
Code—首选项—用户代码片段，选择python
在配置文件中，输入想要定义的内容，字段含义如下：
prefix      :这个参数是使用代码段的快捷入口,比如这里的log在使用时输入log会有智能感知.
body        :这个是代码段的主体.需要设置的代码放在这里,字符串间换行的话使用\r\n换行符隔开.注意如果值里包含特殊字符需要进行转义.
$1          :这个为光标的所在位置.
$2          :使用这个参数后会光标的下一位置将会另起一行,按tab键可进行快速切换
description :代码段描述,在使用智能感知时的描述
```

完整配置文件实例如下：
```
1. {
2. /*
3. //Place your snippets forPython here.Each snippet is defined under a snippet name and has a prefix, body and
4. // description.The prefix is what is used to trigger the snippet and the body will be expanded and inserted.Possible variables are:
5. // $1, $2 for tab stops, $0 for the final cursor position,and ${1:label},${2:another}for placeholders.Placeholderswith the
6. // same ids are connected.
7. //Example:
8."Print to console":{
9. "prefix":"log",
10. "body":[
11. "console.log('$1');",
12. "$2"
13. ],
14. "description":"Log output to console"
15. }
16. */
17. "Input Note":{
18. "prefix":"itne",
19. "body":[
20. "'''",
21. "Function Name : diff_Json_Same",
22. "Function : 通用比较xx方法",
23. "Input Parameters: jsonastr,jsonbstr",
24. "Return Value : None",
25. "'''"
26. ],
27. "description":"Input the class or function notes!"
28. }
29. }

- - 调用方式：在python文件中输入itne回车，则输入定义代码片段
```

配置快捷键
```
Code—首选项—键盘映射拓展
```
配置主题
```
Code—首选项—颜色主题
Code—首选项—文件图标主题
```

## 设置信息

```
# 设定代码格式化调用工具
"python.formatting.provider": "yapf",
# 控制已更新文件的自动保存
"files.autoSave": "afterDelay",   
# path-intellisense插件可调用路径
"path-intellisense.extensionOnImport": true,
# 默认浏览器
"open-in-browser.default": "Google Chrome",
# 控制编辑器是否应自动设置粘贴内容的格式。格式化程序必须可用并且能设置文档中某一范围的格式
"editor.formatOnPaste": true,
# 控制是否将代码段与其他建议一起显示以及它们的排序方式。
"editor.snippetSuggestions": "top",
# git 路径
"git.path": "D:/Program Files/Git/cmd/git.exe",
# 控制命令面板中保留最近使用命令的数量。设置为 0 时禁用命令历史功能。
"workbench.commandPalette.history": 5,
# 设置终端启动时的路径
"terminal.integrated.cwd": "E:/Python/demos",
# 设置linting的启用插件
"python.linting.pylintEnabled": false,
"python.linting.flake8Enabled": true,
```

 ctrl+shift+B运行时输出中文乱码的问题

```
在tasks.json中配置

"options": {
        "env": {
            "PYTHONIOENCODING": "UTF-8"
        }
    }
```

按F5调试时，代码的开始处停止

```
在lanuch.json中配置

“stopOnEntry”: false
```

识别虚拟环境

```
 "python.venvFolders": [
        "envs",
        ".virtualenvs",
        ".pyenv",
        ".direnv"
    ]s
```

设置忽略文件

```
 "files.exclude": {
        "**/.git": true,
        "**/.svn": true,
        "**/.hg": true,
        "**/CVS": true,
        "**/.DS_Store": true,
        "**/*.pyc": true,
        "**/.idea": true,
    },
```



## 快捷键

```
VsCode 快捷键有五种组合方式（科普）

Ctrl + Shift + ? : 这种常规组合按钮
Ctrl + V Ctrl +V : 同时依赖一个按键的组合
Shift + V c : 先组合后单键的输入
Ctrl + Click: 键盘 + 鼠标点击
Ctrl + DragMouse : 键盘 + 鼠标拖动

通用快捷键

快捷键	作用
Ctrl+Shift+P,F1	展示全局命令面板
Ctrl+P	快速打开最近打开的文件
Ctrl+Shift+N	打开新的编辑器窗口
Ctrl+Shift+W	关闭编辑器

基础编辑

快捷键	作用
Ctrl + X	剪切
Ctrl + C	复制
Alt + up/down	移动行上下
Shift + Alt up/down	在当前行上下复制当前行
Ctrl + Shift + K	删除行
Ctrl + Enter	在当前行下插入新的一行
Ctrl + Shift + Enter	在当前行上插入新的一行
Ctrl + Shift + | 匹配花括号的闭合处，跳转	
Ctrl + ] / [	行缩进
Home	光标跳转到行头
End	光标跳转到行尾
Ctrl + Home	跳转到页头
Ctrl + End	跳转到页尾
Ctrl + up/down	行视图上下偏移
Alt + PgUp/PgDown	屏视图上下偏移
Ctrl + Shift + [	折叠区域代码
Ctrl + Shift + ]	展开区域代码
Ctrl + K Ctrl + [	折叠所有子区域代码
Ctrl + k Ctrl + ]	展开所有折叠的子区域代码
Ctrl + K Ctrl + 0	折叠所有区域代码
Ctrl + K Ctrl + J	展开所有折叠区域代码
Ctrl + K Ctrl + C	添加行注释
Ctrl + K Ctrl + U	删除行注释
Ctrl + /	添加关闭行注释
Shift + Alt +A	块区域注释
Alt + Z	添加关闭词汇包含

导航

快捷键	作用
Ctrl + T	列出所有符号
Ctrl + G	跳转行
Ctrl + P	跳转文件
Ctrl + Shift + O	跳转到符号处
Ctrl + Shift + M	打开问题展示面板
F8	跳转到下一个错误或者警告
Shift + F8	跳转到上一个错误或者警告
Ctrl + Shift + Tab	切换到最近打开的文件
Alt + left / right	向后、向前
Ctrl + M	进入用Tab来移动焦点

查询与替换

快捷键	作用
Ctrl + F	查询
Ctrl + H	替换
F3 / Shift + F3	查询下一个/上一个
Alt + Enter	选中所有出现在查询中的
Ctrl + D	匹配当前选中的词汇或者行，再次选中-可操作
Ctrl + K Ctrl + D	移动当前选择到下个匹配选择的位置(光标选定)
Alt + C / R / W	

多行光标操作于选择

快捷键	作用
Alt + Click	插入光标-支持多个
Ctrl + Alt + up/down	上下插入光标-支持多个
Ctrl + U	撤销最后一次光标操作
Shift + Alt + I	插入光标到选中范围内所有行结束符
Ctrl + I	选中当前行
Ctrl + Shift + L	选择所有出现在当前选中的行-操作
Ctrl + F2	选择所有出现在当前选中的词汇-操作
Shift + Alt + right	从光标处扩展选中全行
Shift + Alt + left	收缩选择区域
Shift + Alt + (drag mouse)	鼠标拖动区域，同时在多个行结束符插入光标
Ctrl + Shift + Alt + (Arrow Key)	也是插入多行光标的[方向键控制]
Ctrl + Shift + Alt + PgUp/PgDown	也是插入多行光标的[整屏生效]

丰富的语言操作

快捷键	作用
Ctrl + Space	输入建议[智能提示]
Ctrl + Shift + Space	参数提示
Tab	Emmet指令触发/缩进
Shift + Alt + F	格式化代码
Ctrl + K Ctrl + F	格式化选中部分的代码
F12	跳转到定义处
Alt + F12	代码片段显示定义
Ctrl + K F12	在其他窗口打开定义处
Ctrl + .	快速修复部分可以修复的语法错误
Shift + F12	显示所有引用
F2	重命名符号
Ctrl + Shift + . / ,	替换下个值
Ctrl + K Ctrl + X	移除空白字符
Ctrl + K M	更改页面文档格式

编辑器管理

快捷键	作用
Ctrl + F4, Ctrl + W	关闭编辑器
Ctrl + k F	关闭当前打开的文件夹
Ctrl + |切割编辑窗口	
Ctrl + 1/2/3	切换焦点在不同的切割窗口
Ctrl + K Ctrl <-/->	切换焦点在不同的切割窗口
Ctrl + Shift + PgUp/PgDown	切换标签页的位置
Ctrl + K <-/->	切割窗口位置调换

文件管理

快捷键	作用
Ctrl + N	新建文件
Ctrl + O	打开文件
Ctrl + S	保存文件
Ctrl + Shift + S	另存为
Ctrl + K S	保存所有当前已经打开的文件
Ctrl + F4	关闭当前编辑窗口
Ctrl + K Ctrl + W	关闭所有编辑窗口
Ctrl + Shift + T	撤销最近关闭的一个文件编辑窗口
Ctrl + K Enter	保持开启
Ctrl + Shift + Tab	调出最近打开的文件列表，重复按会切换
Ctrl + Tab	与上面一致，顺序不一致
Ctrl + K P	复制当前打开文件的存放路径
Ctrl + K R	打开当前编辑文件存放位置【文件管理器】
Ctrl + K O	在新的编辑器中打开当前编辑的文件

显示

快捷键	作用
F11	切换全屏模式
Shift + Alt + 1	切换编辑布局【目前无效】
Ctrl + =/-	放大 / 缩小
Ctrl + B	侧边栏显示隐藏
Ctrl + Shift + E	资源视图和编辑视图的焦点切换
Ctrl + Shift + F	打开全局搜索
Ctrl + Shift + G	打开Git可视管理
Ctrl + Shift + D	打开DeBug面板
Ctrl + Shift + X	打开插件市场面板
Ctrl + Shift + H	在当前文件替换查询替换
Ctrl + Shift + J	开启详细查询
Ctrl + Shift + V	预览Markdown文件【编译后】
Ctrl + K v	在边栏打开渲染后的视图【新建】

调试

快捷键	作用
F9	添加解除断点
F5	启动调试、继续
F11 / Shift + F11	单步进入 / 单步跳出
F10	单步跳过
Ctrl + K Ctrl + I	显示悬浮

集成终端

快捷键	作用
Ctrl + `	打开集成终端
Ctrl + Shift + `	创建一个新的终端
Ctrl + Shift + C	复制所选
Ctrl + Shift + V	复制到当前激活的终端
Shift + PgUp / PgDown	页面上下翻屏
Ctrl + Home / End	滚动到页面头部或尾部

```

