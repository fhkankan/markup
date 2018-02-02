"""
Button按钮
"""
# 用于实现各种按钮
# text          --->显示文本内容
# command       --->指定button的事件处理函数
# compound      --->指定文本与图像的位置关系
# bitmap        --->指定位图
# focus_set     --->设置当前组件得到的焦点
# master        --->代表了父窗口
# bg            --->设置背景颜色
# fg            --->设置前景颜色
# font          --->设置字体大小
# height        --->设置显示高度，若未设置，其大小以适应内容标签
# relief        --->指定外观装饰边界附近的标签，默认是平的--->flat,groove,raised,ridge,solid,sunken
# width         --->设置显示宽度，若未设置，其大小以适应内容标签
# wraplength    --->将此选项设置为所需的数量限制每行额字符数，默认为0
# state         --->设置组件状态--->normal,active,disabled
# anchor        --->设置button文本在控件上的显示位置--->n,s,e,w,nw,sw,se,ne,center
# bd            --->设置button的边框大小，bd缺省为1或2个像素
# textvariable  --->设置button可变的文本内容对应的变量
# flash()       --->按钮在active color和normal color颜色之间闪烁几次，disabled状态无效 
# invoke()      --->调用按钮的command指定的回调函数