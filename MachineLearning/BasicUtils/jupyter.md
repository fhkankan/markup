# jupyter

## 概述

安装

```shell
pip install jupyter
```

启动

```shell
jupyter notebook
```

## 使用

快捷键

```python
# mac快捷键及对应含义
F					find and replace
enter				enter edit mode
shift+Enter			run cell, select below
contorl+Enter		run selected cell
option+Enter		run cell, insert below
Y					to code
M					to markdown
R					to raw
A					add cell up the selected cell
B					add cell bellow the selected cell
```

魔法命令

```python
%run	path
# 执行path路径所代表的脚本文件

%timeit 表达式
# 单行表达式代码运行时间(多次求平均)
%%timeit
# 单个cell中所有代码运行时间(多次求平均)

%time 表达式
# 一次测量单行表达式代码运行时间
%%time
# 一次单个cell中所有代码运行时间

%lsmagic
# 查看所有魔法命令
%run?
# %run的帮助信息
```



