"""
Menu
"""
# 主菜单/上下文菜单

# 创建对象
# Menu对象 = Menu(Windows窗口对象)

# 显示对象
# Windows窗口对象['menu'] = Menu对象
# Windows窗口对象.mainloop()

# 添加下拉菜单
# Menu对象1.add_cascade(label = 菜单文本，menu = Menu对象2)
# Menu对象2 = Menu(Menu对象1)

# 在菜单中添加复选框
# 菜单对象.add_checkbutton(label = 复选框的显示文本, command = 菜单命令函数, variable = 与复选框绑定的变量)

# 在菜单中的当前位置添加分隔符
# 菜单对象.add_separator()

# 创建上下文菜单
# 1、创建菜单
menubar = Menu(root)
menubar.add_command(label = 'move', command = hello1)
menubar.add_command(label = 'copy', command = hello2)
menubar.add_command(label = 'nest', command = hello3)
# 2、绑定鼠标右击时间，并在事件处理函数中弹出菜单
def popup(event):
    menubar.post(event.x_root,event.y_root)
root.bind('<Burron - 3>',popup)
