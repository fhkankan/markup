"""
事件类型
"""
# 通用格式
# <[modifier-]...type[-detail]>
# 事件类型必须放置于尖括号<>内，type描述了类型，如键盘按键，鼠标单击
# modifier 用于组合键定义，如Ctrl,Alt;detail用于明确定义是哪一个键或按钮的事件

# 键盘事件
# KyePress          --->按下键盘某键时触发，可以在detail部分指定是哪个键
# KeyRelease        --->释放键盘某键时触发，可以在detail部分指定是哪个键

# 鼠标事件
# ButtonPress/Button--->按下鼠标某键,可以在detail部分指定是哪个键
# ButtonRelease     --->释放鼠标某键,可以在detail部分指定是哪个键
# Motion            --->点中组件的同时拖拽组件移动时触发
# Enter             --->当鼠标指针移进某组件时触发
# Leave             --->当鼠标指针移出某组件时触发
# MouseWheel        --->当鼠标滚轮滚动时触发

# 窗体事件
# Visibility        --->当组件变为可视状态时触发
# Unmap             --->当组件由显示状态变为隐藏状态时触发
# Map               --->当组件由隐藏状态变为显示状态时触发
# Expose            --->当组件从原本被其他组件遮盖的状态中暴露出来时触发
# FocusIn           --->组件获得焦点时触发
# FocusOut          --->组件失去焦点时触发
# Configure         --->当改变组件大小时触发，例如拖拽窗体边缘
# Property          --->当窗体的属性被删除或改变时触发，属于Tk的核心事件
# Destroy           --->当组件被销毁时触发
# Activate          --->与组件选项中的state项有关，表示组件由不可用转为可用
# Deactivate        --->与组件选项中的state项有关，表示组件由可用转为不可用

# 组合键定义中的修饰符
# Alt               --->当Alt键按下
# Any               --->在任何按键按下，如<Any-KeyPress>
# Control           --->Control键按下
# Double            --->两个事件在短时间内发生，如双击鼠标左键<Double-Button-1>
# Lock              --->当CapsLk键按下
# Shift             --->当shift键按下
# Triple            --->类似Double，三个事件短时间内发生

# 对于大多数的单字符按键，可以忽略<>符号，但是空格键和尖括号键需保留，如<space>,<less>



"""
事件绑定
"""
# 程序建立一个处理某一事件的事件处理函数

# 创建组件对象时指定
# 创建对象实例时，通过其命名参数command指定事件处理函数

# 实例绑定
# 组件对象实例名.bind('<事件类型>'，事件处理函数)

# 类绑定
# 组件实例名.bind_class("组件类",'<事件类型>',事件处理函数)

# 程序界面绑定
# 组件实例名.bind_all('<事件类型>',事件处理函数)

# 标识绑定
# Canvas对象.tag_bind('标识','<事件类型>',事件处理函数)


"""
事件处理
"""
# 定义事件处理函数
# def callback(event):
    # showinfo()
# 触发事件调用事件处理函数时，将传递event对象实例


# 参数属性
# x,.y              --->鼠标相对于组件对象左上角的坐标
# x_root,y_root     --->鼠标相对与屏幕左上角的坐标
# keysym            --->字符串命名按键，如Escape,F1...F12,...
# keysym_num        --->数字代码命名按键
# keycode           --->键码，但不能反映事件前缀Alt,Control,Shift,Lock,并且不区分大小写
# time              --->时间
# type              --->事件类型
# widget            --->触发事件的对应组件
# char              --->字符

# 按键详细信息
# keysym        --->keycode         --->keysym_num      --->说明
# Alt_L         --->64              --->65513           --->左手边的Alt键
# Alt_R         --->113             --->65514           --->右手边的Alt键
# BackSpace     --->22              --->65288           --->BackSpace
# Cancel        --->110             --->65387           --->Pause Break
# F1~F11        --->67~77           --->65470~65480     --->功能键F1~F12
# Print         --->111             --->65377           --->打印屏幕键








 









