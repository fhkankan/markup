"""
Canvas画布组件
"""
# 长方形区域，用于绘制或复杂的图形界面布局。
# 可以在画布上绘制图形，文字，放置各种组件和框架

# 创建对象
# Canvas对象 = Canvas(窗口对象, 选项,...)

# 显示对象
# Canvas对象.pack()

# 属性
# bd                    --->指定画布的边框宽度，单位是像素
# bg                    --->指定画布的背景颜色
# confie                --->指定画布在滚动区域外是否可以滚动。默认True,表示不能滚动
# cursor                --->指定画布中的鼠标指针，如arrow,circle,dot
# height                --->指定画布的高度
# highlightcolor        --->选中画布时的背景颜色
# relief                --->指定画布的边框样式，如SUNKEN,RAISED,GROOVE,RIDGE
# scrolregion           --->指定画布的滚动区域的元组(w,n,e,s)


"""
绘制图形
"""

# 绘制图形
# creat_arc()           --->圆弧
# creat_line()          --->直线
# creat_bitmap()        --->位图
# creat_image()         --->位图图像
# creat_oval()          --->椭圆
# creat_polyon()        --->多边形
# creat_window()        --->子窗口
# creat_text()          --->文字对象

# 每个绘制对象都有一个标识id,使用绘制函数创建绘制对象时，返回id
# 在创建图形对象时可以使用属性tags设置图形对象的标记(tag)
# 指定标记后，使用find_withtag()方法可获取到指定tag的图形对象，然后设置对象的属性
# Canvas对象.find_withtag(tag名)
# 返回一个图形对象数组，其中包含所有具有tag名的图形对象
# Canvas对象.itemconfig(图形对象, 属性1 = 值1, 属性2 = 值2...)


# 绘制圆弧
# Canvas对象.creat_arc(弧外框矩形左上角的x坐标,弧外框矩形左上角的y坐标,弧外框矩形右下角的x坐标,弧外框矩形右下角的y坐标,选项,...)
# outline,边框颜色；fill,填充颜色；width,边框宽度；start,起始角度；extent,指定角度偏移量而不是终止角度

# 绘制线条
# line = Canvas对象.creat_line(x0,yo,x1,y1,...,xn,yn)
# xn,yn,指线段的端点
# width,宽度；arrow,指定是否使用箭头(none,无；first,起点有箭头；last,终点有箭头；both,两端有箭头)

# 绘制矩形
# Canvas对象.create_rectangle(矩形左上角的x坐标,矩形左上角的y坐标,矩形右下角的x坐标,矩形右下角的y坐标,选项,...)
# outline,边框颜色；fill,填充颜色；width,边框宽度；dash,指定边框为虚线；stipple,使用指定自定义画刷填充矩形

# 绘制多边形
# Canvas对象.creat_polygon(顶点1的x坐标，顶点1的y坐标，...,顶点n的x坐标，顶点n的y坐标，选项，...)
# outline,边框颜色；fill,填充颜色；width,边框宽度；smooth,指定多变形的平滑程度(等于0表示多边形是折线，等于1表示多边形是平滑曲线)

# 绘制椭圆
# Canvas对象.creat_oval(包裹椭圆的矩形左上角x坐标，包裹椭圆的矩形左上角y坐标，包裹椭圆的矩形右下角x坐标，包裹椭圆的矩形右下角y坐标，选项，...)
# outline,边框颜色；fill,填充颜色；width,边框宽度；如果包裹椭圆的矩形是正方形则绘制一个圆形

# 绘制文字
# Canvas对象.creat_text((文本左上角的x坐标，文本左上角的y坐标),选项,...)
# text,文字对象的内容文本；fill,指定文字颜色；anchor,控制文字对象的位置(n,s,e,w,nw,sw,se,ne,center(默认))；justify,文字对象中文本的对齐方式(eft,right,center(默认))
# Canvas对象.select_from(文字对象，选中文本的起始位置)     --->指定选中文本的起始位置
# Canvas对象.select_to(文字对象，选中文本的起始位置)       --->指定选中文本的结束位置

# 绘制位图
# Canvas对象.creat_bitmap((x坐标,y坐标),bitmap = 位图字符串，选项，...)
# (x坐标,y坐标)是位图放置的中心坐标，常用选项有bitmap(正常),activebitmap(活动),disabledbitmap(禁用)

# 绘制图像
# Canvas对象.creat_image((x坐标,y坐标),bitmap = 位图字符串，选项，...)
# (x坐标,y坐标)是位图放置的中心坐标，常用选项有bitmap(正常),activebitmap(活动),disabledbitmap(禁用)
# PhotoImage(file = 图像文件)      --->函数获取图像文件对象

"""
修改图形
"""
# 修改图形对象的坐标
# Canvas对象.coords(图形对象,(图形左上角的x坐标，图形左上角y坐标，图形右下角的x坐标，图形右下角的y坐标))
# 若图形对象是图像文件，则只能指定图像中心点坐标，不能缩放图像

# 移动指定图形对象
# Canvas对象.move(图形对象,x坐标偏移量,y坐标偏移量)


# 删除图形对象
# Canvas对象.delete(图形对象)

# 缩放图形对象
# Canvas对象.scale(图形对象,x轴偏移量,y轴偏移量，x轴缩放比例，y轴缩放比例)












 