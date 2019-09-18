#  配置

文件头

> 变量

```
${PROJECT_NAME} - the name of the current project.
${NAME} - the name of the new file which you specify in the New File dialog box during the file creation.
${USER} - the login name of the current user.
${DATE} - the current system date.
${TIME} - the current system time.
${YEAR} - the current year.
${MONTH} - the current month.
${DAY} - the current day of the month.
${HOUR} - the current hour.
${MINUTE} - the current minute.
${PRODUCT_NAME} - the name of the IDE in which the file will be created.
${MONTH_NAME_SHORT} - the first 3 letters of the month name. Example: Jan, Feb, etc.
${MONTH_NAME_FULL} - full name of a month. Example: January, February, etc.
```

> 样例

```
# -*- coding: utf-8 -*-
"""
@project:${PROJECT_NAME}
@file: ${NAME}.py
@time: ${DATE} ${TIME}
@author: fuhang
@contact: fu.hang.2008@163.com
@desc: 
"""
```

# 快捷方式

alt + 回车			    导包  /  创建方法   【很有用】
ctrl + win + 空格		快速提示  【很有用】  按两次
ctrl + win + o			导入类方法
ctrl + shift + f12		窗口最大化
ctrl + d			    复制行 （搜：duplicate entry line 设置一个快捷键，可复制选中的多行）
ctrl + y			    删除选中的代码
ctrl + o			    重写父类的方法

ctrl + alt + l			自动格式化
ctrl + /			    单行注释
ctrl + shift + /		多行注释

ctrl + shift + backspace     	返回到上一次编辑的地方
ctrl + w			    选中内容(按多次扩大选中范围)
ctrl + e			    查看最近打开过的一些文件
ctrl + shift + o		删除没有用到的包
ctrl + 回车			    在当前行上方创建一行
shift + 回车			在当前行下方创建一行