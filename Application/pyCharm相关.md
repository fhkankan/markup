# pyCharm使用总结

[参考](https://blog.csdn.net/u013088062/article/details/50388329)

## 快捷方式

### 常用

```
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
```

### 所有

编辑类

```
Ctrl + Space 基本的代码完成（类、方法、属性）
Ctrl + Alt + Space 类名完成
Ctrl + Shift + Enter 语句完成
Ctrl + P 参数信息（在方法中调用参数）
Ctrl + Q 快速查看文档
Shift + F1 外部文档
Ctrl + 鼠标 简介
Ctrl + F1 显示错误描述或警告信息
Alt + Insert 自动生成代码
Ctrl + O 重新方法
Ctrl + Alt + T 选中
Ctrl + / 行注释
Ctrl + Shift + / 块注释
Ctrl + W 选中增加的代码块
Ctrl + Shift + W 回到之前状态
Ctrl + Shift + ]/[ 选定代码块结束、开始
Alt + Enter 快速修正
Ctrl + Alt + L 代码格式化
Ctrl + Alt + O 优化导入
Ctrl + Alt + I 自动缩进
Tab / Shift + Tab 缩进、不缩进当前行
Ctrl+X/Shift+Delete 剪切当前行或选定的代码块到剪贴板
Ctrl+C/Ctrl+Insert 复制当前行或选定的代码块到剪贴板
Ctrl+V/Shift+Insert 从剪贴板粘贴
Ctrl + Shift + V 从最近的缓冲区粘贴
Ctrl + D 复制选定的区域或行
Ctrl + Y 删除选定的行
Ctrl + Shift + J 添加智能线
Ctrl + Enter 智能线切割
Shift + Enter 另起一行
Ctrl + Shift + U 在选定的区域或代码块间切换
Ctrl + Delete 删除到字符结束
Ctrl + Backspace 删除到字符开始
Ctrl + Numpad+/- 展开折叠代码块
Ctrl + Numpad+ 全部展开
Ctrl + Numpad- 全部折叠
Ctrl + F4 关闭运行的选项卡
```

运行类：

```
Alt + Shift + F10 运行模式配置
Alt + Shift + F9 调试模式配置
Shift + F10 运行
Shift + F9 调试
Ctrl + Shift + F10 运行编辑器配置
Ctrl + Alt + R 运行manage.py任务
```

调试类：

```
F8 跳过
F7 进入
Shift + F8 退出
Alt + F9 运行游标
Alt + F8 验证表达式
Ctrl + Alt + F8 快速验证表达式
F9 恢复程序
Ctrl + F8 断点开关
Ctrl + Shift + F8 查看断点
```

导航类：

```
Ctrl + N 跳转到类
Ctrl + Shift + N 跳转到符号
Alt + Right/Left 跳转到下一个、前一个编辑的选项卡
F12 回到先前的工具窗口
Esc 从工具窗口回到编辑窗口
Shift + Esc 隐藏运行的、最近运行的窗口
Ctrl + Shift + F4 关闭主动运行的选项卡
Ctrl + G 查看当前行号、字符号
Ctrl + E 当前文件弹出
Ctrl+Alt+Left/Right 后退、前进
Ctrl+Shift+Backspace 导航到最近编辑区域
Alt + F1 查找当前文件或标识
Ctrl+B / Ctrl+Click 跳转到声明
Ctrl + Alt + B 跳转到实现
Ctrl + Shift + I查看快速定义
Ctrl + Shift + B跳转到类型声明
Ctrl + U跳转到父方法、父类
Alt + Up/Down跳转到上一个、下一个方法
Ctrl + ]/[跳转到代码块结束、开始
Ctrl + F12弹出文件结构
Ctrl + H类型层次结构
Ctrl + Shift + H方法层次结构
Ctrl + Alt + H调用层次结构
F2 / Shift + F2下一条、前一条高亮的错误
F4 / Ctrl + Enter编辑资源、查看资源
Alt + Home显示导航条F11书签开关
Ctrl + Shift + F11书签助记开关
Ctrl + #[0-9]跳转到标识的书签
Shift + F11显示书签
```

查找/替换类：

```
F3 下一个
Shift + F3 前一个
Ctrl + R 替换
Ctrl + Shift + F 全局查找
Ctrl + Shift + R 全局替换
```

搜索类：

```
Alt + F7/Ctrl + F7文件中查询用法
Ctrl + Shift + F7文件中用法高亮显示
Ctrl + Alt + F7显示用法
```

重构类：

```
F5复制F6剪切
Alt + Delete安全删除
Shift + F6重命名
Ctrl + F6更改签名
Ctrl + Alt + N内联
Ctrl + Alt + M提取方法
Ctrl + Alt + V提取属性
Ctrl + Alt + F提取字段
Ctrl + Alt + C提取常量
Ctrl + Alt + P提取参数
```

一般：

```
Alt + #[0-9]打开相应的工具窗口
Ctrl + Alt + Y同步
Ctrl + Shift + F12最大化编辑开关
Alt + Shift + F添加到最喜欢
Alt + Shift + I根据配置检查当前文件
Ctrl + BackQuote(’)快速切换当前计划
Ctrl + Alt + S　打开设置页
Ctrl + Shift + A查找编辑器里所有的动作
Ctrl + Tab在窗口间进行切换
```

## 使用技巧

### 使用docker环境

[参考](https://blog.csdn.net/buxianghejiu/article/details/84839338)

方法

```
File/Settings/Project:xxx/Project Interpreter/Add/Docker
```

### 添加个人信息

[参考](https://blog.csdn.net/m0_37886429/article/details/76977691)

方法

```
File/Settings/Editor/File and Code Templates/Python Script
```

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
