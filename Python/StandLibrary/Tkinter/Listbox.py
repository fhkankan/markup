"""
Listbox列表框
"""
# 用于显示多个项目，切允许用户选择一个或多个项目

# 创建Listbox对象
# Listbox 对象 = Listbox(Thinter Windows窗口对象)

# 显示Listbox对象
# Listbox对象.pack()

# 插入文本项
# Listbox对象.insert(index,item)

# 返回选中项索引
# Listbox对象.curselection()

# 删除文本项
# Listbox对象.delete(first,last)

# 获取项目内容
# Listbox对象.get(first,last)

# 获取项目个数
# Listbox对象.size()

# 获取Listbox内容
m = StringVar()
listb = Listbox(root,listvariable = m)
listb.pack()
root.mainloop()
listb.get()


