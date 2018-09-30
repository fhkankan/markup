# 控制shell程序

sys.vesion
# 获取解释器的版本信息

sys.path
# 获取模块的搜索路径，初始化时使用PYTHONPATH环境变量的值

sys.platform
# 获取操作系统平台的名称

sys.maxint
# 最大的int值

sys.maxunicode
# 最大的Unicode值

sys.stdin
# 获取信息到shell程序中

sys.stdout
# 向shell程序输出信息

sys.exit()
# 退出shell程序

sys._getframe(0)
# 获取当前栈信息

sys._getframe(0).f_code.co_filename
# 当前文件名

sys._getframe(0).f_code.co_name
# 当前函数名

sys._getframe(0).f_lineno
# 当前行号

sys._getframe().f_code.co_filename.split('/')[-1].split('.')[0]
# 获取文件名


