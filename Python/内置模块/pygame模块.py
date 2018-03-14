import pygame

# 专为电子游戏设计，包含图像、声音功能和网络支持

"""
模块
"""
# pygame.cdrom ----》 访问光驱
# pygame.cursors ----》加载光标
# pygame.display  ----》访问显示设备
# pygame.draw  ----》绘制形状、线和点
# pygame.event  ----》管理事件
# pygame.font  ----》使用字体
# pygame.image  ----》加载和存储图片
# pygame.joystick  ----》使用游戏手柄或类似的东西
# pygame.key  ----》读取键盘按键
# pygame.movie  ----》播放视频
# pygame.music  ----》播放音频
# pygame.mixer  ----》声音
# pygame.mouse  ----》鼠标
# pygame.overlay  ----》访问高级视频叠加
# pygame.rect  ----》管理矩形区域
# pygame.sndarray  ----》操作声音数据
# pygame.sprite  ----》操作移动图像
# pygame.surface  ----》管理图像和屏幕
# pygame.surfarray  ----》管理点阵图像数据
# pygame.time  ----》管理时间和帧信息
# pygame.transform  ----》缩放和移动图像
# pygame.locals  ---->> pygame中各种常量，包括事件类型、按键和视频模式等的名字

"""
检测模块是否正常
"""
# 由于硬件和游戏的兼容新或是请求的驱动没有安装，有些模块可能在某些平台无法使用，可用None来测试
if pygame.font is None:
    print('The font module is not available!')
    # 若没有，则退出应用环境
    pygame.quit()
    
"""
surface
"""
pygame.surface((width,height),flag=0,depth=0,masks=none)
# 返回一个新surface对象。是一个有确定大小尺寸的空图像，用来进行图像绘制与移动

"""
display
"""
# flip/update更新显示
# 一般来说，修改当前屏幕额时候要经过两步，首先需要对get_surface函数返回的surface对象进行修改，然后调用pygame.display.flip()更新显示以反映所做的修改。只更新屏幕一部分的时候调用update()函数，而不是flip()函数

# set_mode(resolution,flags,depth)
# 建立游戏窗口，返回surface对象
#第一个参数是元组,指定窗口尺寸，第三参数为色深，指定窗口的色彩位数，第二参数是标志位，含义如下：
# FULLSCREEN --->>  创建一个全屏窗口
# DOUBLEBUY  --->>  创建一个'双缓冲'窗口，建议在HWSURFACE或者OPENGL时使用
# HWSURFACE  --->>  创建一个硬件加速的窗口，必须和FULLSCREEN同时使用
# OPENGL     --->>  创建一个OPENGL渲染的窗口
# RESIZABLE  --->>  创建一个可以改变大小的窗口
# NOFRAME    --->>  创建一个没有边框的窗口

# set_caption 设定游戏程序标题
# 当游戏以窗口模式(对应于全屏)运行时尤其有用，因为该标题会作为窗口的标题

# get_surface 
# 返回一个可用来画图的surface对象

"""
sprite
"""
# sprite精灵类
# 是所有可视游戏的基类。为了实现自己的游戏对象，需要子类化sprite,覆盖它的构造函数以设定image和rect属性(决定sprite的外观和放置的位置)，再覆盖update()方法。在sprite需要更新时可调用update()方法。

# group精灵组
# group实例用作精灵sprite对象的容器。在一些简单的游戏中，只要创建名为sprites\allsprite或是其他类似的组，然后将所有sprite精灵对象添加到上面即可。group精灵组对象的update()方法被调用时，就会自动调用所有sprite精灵对象的update()方法。group精灵组对象的clear()方法用于清理它包含的所有sprite对象(使用回调函数实现清理)，group精灵组对象draw()方法用于绘制所有的sprite对象。

"""
mouse
"""
# set_visible(false/true)
# 隐藏/显示鼠标光标

# get_pos()
# 获取鼠标位置

"""
event
"""
# get()
# 获取最近事件列表

"""
image
"""
# load()
# 读取图像文件










