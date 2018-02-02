"""
对话框
"""
# 子模块messagebox,filedialog,colorchoser,simpledialog包含一些通用预定义对话框；
# 也可以通过继承TopLevel创建自定义对话框


# filedialog        --->文件对话框

# 打开文件对话框
# askopenfilename(title = '标题', filetypes = [('所有文件','.*'),('文本文件','.txt')])
# title         ---> 设置打开文件对话框的标题
# filetypes     ---> 文件过滤器，可以筛选某种格式文件

# 文件保存对话框
# asksavefilename(title = '标题', initialdir = '文件路径', initialfile = '文件名')
# initialdir    ---> 默认保存路径即文件夹
# initialfile   ---> 默认保存的文件名


# colorchoser       --->颜色对话框

# 打开颜色对话框
# askcolor()


# simpledialog      --->简单对话框
# askfloat(title, prompt, 选项)           --->打开输入对话框，输入并返回浮点数
# askinteger(title, prompt, 选项)         --->打开输入对话框，输入并返回整数
# askstring(title, prompt, 选项)          --->打开输入对话框，输入并返回字符串
# title--->窗口标题，prompt--->提示文本信息，选项是指initialvalue,minvalue,maxvalue


