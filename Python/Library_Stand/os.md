# os

os模块除了提供使用操作系统功能和访问文件系统之外，还有大量文件和文件夹操作方法

## 查看

### 当前系统

```python
os.extsep
# 当前操作系统所使用的文件扩展名分隔符  

os.sep
# 当前操作系统所用的路径分隔符 

os.get_exec_path()
# 返回操作系统中可执行文件的搜索路径
```
### 当前目录

```python
os.getcwd()
# 返回当前工作目录

os.getcwsu()
# 返回一个当前工作目录的Unicode对象

os.curdir()
# 返回当前文件夹

os.chdir(path)
# 把path设为当前工作目录

os.chroot(path)
# 改变当前进程的根目录
```

### 其他

```python
os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
# 遍历目录树，该方法返回一个元组，包括3个元素，所有路径名、所有目录列表与文件列表
import os
o = os.walk('d:\new\')
print([root, dirs, files] for root, dirs, files in o)

os.stat(path)
# 获取path指定的路径的信息，功能等同于C API中的stat()系统调用。

os.statvfs(path)
# 获取指定路径的文件系统统计信息

os.listdir(path)
#返回path指定的文件夹包含的文件或文件夹的名字的列表 
# ./当前路径(相对路径)            
            
os.scandir(path='.')
# 返回包含指定文件夹中所有DirEntry对象的迭代对象，遍历文件夹时比listdir()更加高效     

os.pathconf(path,name)
# 返回相关文件的系统配置信息

os.utime(path, times)
# 返回指定的path文件的访问和修改的时间         
```

### 路径相关

```python
os.path.abspath(path)
# 返回给定路径的绝对路径
os.path.realpath(path)
# 返回给定路径的绝对路径
os.path.relpath(path)
# 返回给定路径的相对路径，不能跨越磁盘驱动器或分区

os.path.dirname(path)
# 返回path参数中的路径文件夹部分
os.path.basename(path)
# 返回path参数中的文件名
os.path.commonpath(paths)
# 返回给定的多个路径的最长公共路径
os.path.commonprefix(paths)
# 返回给定的多个路径的最长公共前缀

os.path.exists(path)
# 判断参数path的文件或文件夹是否存在。存在返回True，不存在返回False
os.path.isabs(path)
# 判断path是否为绝对路径
os.path.isfile(path)
# 判断参数path存在且是一个文件，是则返回True，都则返回False
os.path.isdir(path)
# 判断参数path存在且是一个文件夹，是则返回True，否则返回False
os.path.samefile(f1,f2)
# 测试f1和f2这两个路径是否引用同一个文件

os.path.getatime(filename)
# 返回文件的最后访问时间
os.path.getctime(filename)
# 返回文件的创建时间
os.path.getmtime()
# 获取指定路径的最后修改时间

os.path.getsize(path)
# 产看文件的大小

os.path.split(path)
# 以路径中最后一个斜线为分隔符把路径分隔成两部分，以列表形式返回
os.path.splitext(path)
# 从路径中分隔文件的扩展名
os.path.splitdrive(path)
# 从路径中分割驱动器的名称

os.path.join(path, *paths)
# 返回连接两个或多个path后的新路径
```

## 环境变量

```python
# 包含系统环境变量和值的字典
os.environ

# 设置环境变量
os.environ["JAVA_HOME"] = xxx  

# 获取环境变量种key对应的值
os.environ.get("a") 
os.getenv("a")
```

## 重命名

```python
os.rename(src,dst)
# 重命名文件或目录，从 src 到 dst，可以实现文件的移动，若目标文件已存在则抛异常，不能跨越磁盘或分区

os.renames(old,new)
# 递归地对目录进行更名，也可以对文件进行更名。

os.replace(old, new)
# 重命名文件或目录，若目标文件已存在则覆盖，不能跨越磁盘或分区
```

## 删除

文件

```python
os.remove(path)
# 删除指定的文件，要求用户拥有删除文件的权限，并且文件没有只读或其他特殊权限。如果path是一个文件夹，将抛出OSError; 查看下面的rmdir()删除一个 directory。

os.removedirs(path)
# 递归删除目录


```

文件夹

```python
os.rmdir(path)
# 删除path指定的空目录，目录中不能有文件或子文件。如果目录非空，则抛出一个OSError异常。

os.removedirs(path1/path2...)
# 删除多级目录，目录中不能有文件
```
## 权限

```python
os.access(path,mode)
# 测试是否可以按照mode指定的权限访问文件

os.chmod(path,mode)
# 改变文件的访问权限

os.chown(path,uid,gid)
# 更改文件所有者
```
## 创建

创建文件夹

```python
os.mkdir(path[,mode])
# 以数字mode的mode创建一个名为path的文件夹.默认的 mode 是 0777 (八进制)。要求上级目录必须存在

os.makedirs(path)
# 创建多级目录，会根据需要自动创建中间缺失的目录
```

其他

```python
os.stat_float_times([newvalue])
# 决定stat_result是否以float对象显示时间戳

os.chflags(path,flags)
# 设置路径的标记为数字标记

os.mkfifo(path[,mode])
# 创建命名管道，mode 为数字，默认为 0666 (八进制)

os.mknod(filename[,mode=0600,device])
# 创建一个名为filename文件系统节点(文件，设备特别文件或命名pipe)

os.open(file,flags[,mode])
# 打开一个文件，并且设置需要的打开选项，mode参数是可选的

os.openpty()
# 打开一个新的伪终端对。返回pty和tty的文件描述符

os.pipe()
# 创建一个管道，返回一对文件描述符(r,w)分别为读和写

os.popen(command[,mode[,bufsize]])
# 从一个 command 打开一个管道
	
os.tmpnam()
# 为创建一个临时文件返回一个唯一的路径

os.tempnam([dir[, prefix]])
# 返回唯一的路径名用于创建临时文件。

os.tmpfile()
# 返回一个打开的模式为(w+b)的文件对象 .这文件对象没有文件夹入口，没有文件描述符，将会自动删除
```
## 设备相关

```python
os.major(device)
# 从原始的设备号中提取设备major号码 (使用stat中的st_dev或者st_rdev field)。

os.makedev(major,minor)
# 以major和minor设备号组成一个原始设备号

os.minor(device)
# 从原始的设备号中提取设备minor号码 (使用stat中的st_dev或者st_rdev field )
```
## 连接对象
```python
os.symlink(src,dst)
# 创建一个软链接

os.readlink(path)
# 返回软链接所指向的文件

os.lstat(path)
# 像stat()，但是没有软链接

os.link(src,det)
# 创建硬链接，名为参数dst，指向参数src

os.lchflags(path,flags)
# 设置路径的标记为数字标记,类似chflags()，但是没有软链接

os.lchmod(path,mode)
# 修改连接文件权限

os.lchown(path,uid,gid)
# 更改文件所有者，类似chown,但是不追踪链接

os.lseek(fd,pos,how)
# 设置文件描述符fd当前位置为pos, how方式修改: SEEK_SET 或者 0 设置从文件开始的计算的pos; SEEK_CUR或者 1 则从当前位置计算; os.SEEK_END或者2则从文件尾部开始. 在unix，Windows中有效

os.unlink(path)
# 删除文件路径的软链接
```
## 文件描述符
```python
os.read(fd,n)
# 从文件描述符fd中读取最多 n 个字节，返回包含读取字节的字符串，文件描述符 fd对应文件已达到结尾, 返回一个空字符串。

os.write(fd, str)
# 写入字符串到文件描述符fd中.返回实际写入的字符串长度

os.close(fd)
# 关闭文件描述符fd

os.closerange(fd_low,fd_high)
# 关闭所有文件描述符，从[fd_low,fd_high),错误会忽略

os.dup(fd)
# 复制文件描述符

os.dup2(fd,fd2)
# 将一个文件描述符fd复制到另一个fd2

os.fchdir(fd)
# 通过文件描述符改变当前工作目录

os.fchmod(fd,mode)
# 改变一个文件的访问权限，该文件由参数fd指定，参数mode是Unix下的文件访问权限

os.fchown(fd,uid,gid)
# 修改一个文件的所有权，这个函数修改一个文件的用户ID和用户组ID，该文件由文件描述符fd指定

os.fdatasync(fd)
# 强制将文件谢谢如磁盘，该文件由文件描述符fd指定，但是不强制更新文件的状态信息

os.fdopen(fd[,mode[,bufsize]])
# 通过文件描述符fd创建一个文件独享，并返回这个文件对象

os.fpathconf(fd,name)
# 返回一个打开的文件的系统配置信息。name为检索的系统配置的值，它也许是一个定义系统值的字符串，这些名字在很多标准中指定（POSIX.1, Unix 95, Unix 98, 和其它）。

os.fstat(fd)
# 返回文件描述符fd的状态，像stat()

os.fstatvfs(fd)
# 返回包含文件描述符fd的文件的文件系统的信息，想stavfs()

os.fsync(fd)
# 强制将文件描述符为fd的文件写入硬盘

os.ftruncate(fd,length)
# 裁剪文件描述符fd对应的文件，所以它最大不能超过文件大小

os.isatty(fd)
# 如果文件描述符fd是打开的，同时与tty(-like)设备相连，则返回True,否则False

os.tcgetpgrp(fd)
# 返回与终端fd（一个由os.open()返回的打开的文件描述符）关联的进程组
	
os.tcsetpgrp(fd, pg)
# 设置与终端fd（一个由os.open()返回的打开的文件描述符）关联的进程组为pg。

os.ttyname(fd)
# 返回一个字符串，它表示与文件描述符fd 关联的终端设备。如果fd 没有与终端设备关联，则引发一个异常

os.truncate(path, length)
# 将文件截断，只保留指定长度的内容
```

## 执行程序

```python
startfile(filepath[, operation])
# 使用关联的应用程序打开指定文件或启动指定应用程序

system()
# 启动外部程序
```

如下函数都执行一个新的程序，然后用新的程序替换当前子进程的进程空间，而该子进程从新程序的main函数开始执行。在Unix下，该新程序的进程id是原来被替换的子进程的进程id。在原来子进程中打开的所有描述符默认都是可用的，不会被关闭。

```python
os.execl(path, arg0, arg1, ...)
os.execle(path, arg0, arg1, ..., env)
os.execlp(file, arg0, arg1, ...)
os.execlpe(file, arg0, arg1, ..., env)
os.execv(path, args)
os.execve(path, args, env)
os.execvp(file, args)
os.execvpe(file, args, env)
```

说明

```
execl*系列的函数表示其接受的参数是一个个独立的参数传递进去的。
execv*系列的函数表示其接受的参数是以一个list或者是一个tuple表示的参数表
exec*p*系列函数表示在执行参数传递过去的命令时使用PATH环境变量来查找命令
exec*e系列函数表示在执行命令的时候读取该参数指定的环境变量作为默认的环境配置，最后的env参数必须是一个mapping对象，可以是一个dict类型的对象。
```

> demo

程序重启

```python
import sys
import os
 
python = sys.executable
os.execl(python, python, *sys.argv)
```

控制台清屏

```python
os.system("cls)
```















