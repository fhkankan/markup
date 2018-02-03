import shutil


"""
复制文件
"""
shutil.copy(source,destianation)
# 复制文件和文件夹

shutil.copytree(source,destination)
# 赋值整个文件夹，包括其下的文件及子文件夹


"""
移动文件
"""
shutil.move(source,destination)
# 移动文件和文件夹
# 若目标出存在崇明，则会覆盖


"""
删除文件夹
"""
shutil.rmtree(path)
# 删除整个文件夹，包括所有文件及文件夹

