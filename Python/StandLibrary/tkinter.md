# tkinter
创建windows窗口
```
import tkinter      # 导入Tkinter模块
win = tkinter.Tk()  # 创建windows窗口对象
win.title('my first GUI program') # 设置窗口标题
win.geomertry('800*600') # 初始化窗口大小
win.minsize('400*600')  # 窗口最小尺寸 
win.maxsize('1440*800') # 窗口最大尺寸
win.mainloop()  # 进入消息循环，也就是显示窗口
```
## 几何布局管理器

###  pack 
```
# 块的方式组织组件，根据组件创建生成的顺序
# pack(option = value,...)
# side      --->停靠在父组件的哪一边              --->top(默认),bottom,left,right
# anchor    --->停靠位置，对应东南西北及四角      ---> n,s,e,w,nw,sw,se,ne,center
# fill      --->填充空间                          --->x,y,both,none
# expand    --->扩展空间                          --->0或1
# ipadx,ipady--->组件内部在x/y方向上填充的空间大小--->单位为c(厘米),m(毫米),i(英寸),p(打印机的点)
# padx,pady --->组件外部在x/y方向上填充的空间大小 --->单位为c(厘米),m(毫米),i(英寸),p(打印机的点)
```
###  grid
表格 结构组织组件，子组件的位置由行/列确定的单元格决定
```
# grid(option = value)
# sticky        --->组件紧贴所在单元格的某一角，对应于东南西北及四角    --->n,s,e,w,nw,sw,se,ne,center(默认)
# row           --->单元格行号                                          --->整数
# column        --->单元格列号                                          --->整数
# rowspan       --->行跨度                                              --->整数
# columnpan     --->列跨度                                              --->整数
# ipadx,ipady   --->组件内部在x/y方向上填充的空间大小                   --->单位为c(厘米),m(毫米),i(英寸),p(打印机的点)
# padx,pady     --->组件内部在x/y方向上填充的空间大小                   --->单位为c(厘米),m(毫米),i(英寸),p(打印机的点)
```
### place
指定组件的大小和位置。可精确控制组件的位置，但是子组件不能随窗口灵活改变
```
# place(option = value)
# x,y           --->将组件放到指定位置的绝对坐标       --->从0开始的整数
# relx,rely     --->将组件放到指定位置的相对坐标       --->0~1.0
# height,width  --->高度和宽度，单位为像素             --->
# anchor        --->对齐方式，对应于东南西北及四角     --->n,s,e,w,nw,sw,se,ne,center(默认)
```
## 组件
```
# Button            --->按钮控件，在程序中显示按钮
# Canvas            --->画布控件，显示图形元素，如线条或文本
# Checkbutton       --->多选框空间，用于在程序中提供多项选择框
# Entry             --->输入控件，用于显示简单的文本内容
# Frame             --->框架控件，在屏幕上显示一个矩形区域，用来做容器
# Label             --->标签控件，可以显示文本和位图
# Listbox           --->列表框控件，listbox窗口小部件，用来显示一个字符串列表给用户
# Menubutton        --->菜单按钮控件，显示菜单项
# Menu              --->菜单控件，显示菜单栏，下拉菜单和弹出菜单
# Message           --->消息控件，显示多行文本，与Lable比较类似
# Radiobutton       --->单选按钮控件，显示一个单选的按钮状态
# Scale             --->范围控件，显示一个数值刻度，为输出限定范围的数字区间
# Scrollbar         --->滚动条控件，当内容超过可视化区域时使用，如列表框
# Text              --->文本控件，用于显示多行文本
# Toplevel          --->容器控件，用来提供一个单独的对话框，和Frame类似
# Spinbox           --->输入控件，与Entry类似，但可以指定输入范围值
# PanedWindow       --->窗口布局管理的插件，可以包含一个或者多个子控件
# LabelFrame        --->简单的容器控件，常用于复杂的窗口布局
# tkMessageBox      --->用于显示应用程序的消息框
```
### 通过构造函数创建对象
```
from thinter import *
root = Tk()
button1 = Button(root,text = 'ok')
```
### 标准属性
```
# dimension         --->控件大小
# color             --->控件颜色
# font              --->控件字体
# anchor            --->锚点(内容停靠位置)，对应于东南西北及四个角
# relief            --->控件样式
# bitmap            --->位图，内置位图包括：error,gray75,gray50,gry25,gray12,info,questhead,hourglass,questtion,warning,自定义位图为.xbm格式文件
# cursor            --->光标
# text              --->显示文本内容
# state             --->设置组件状态，normal,active,disabled
```
### 通过下列方式之一设置组件属性
```
# button1 = Button(root,text = 'ok')  # --->按钮组件的构造函数
# button1.config(text = 'ok')         # --->组件对象的config方法的命名参数
# button1['text'] = 'ok'              # --->组件对象的属性赋值
```



