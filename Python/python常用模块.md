# Python

| 模块名         | 是否标准库 | 说明                                                         |
| -------------- | ---------- | ------------------------------------------------------------ |
| atexit         | 是         | 允许注册在程序退出时调用的函数                               |
| argparse       | 是         | 提供解析命令行参数的函数                                     |
| bisect         | 是         | 可为排序列表提供二分查找算法                                 |
| calendar       | 是         | 提供一组与日期相关的函数                                     |
| codecs         | 是         | 提供编解码数据的函数                                         |
| collections    | 是         | 提供一组有用的数据结构                                       |
| copy           | 是         | 提供复制数据的函数                                           |
| csv            | 是         | 提供用于读写CSV文件的函数                                    |
| datetime       | 是         | 处理日期和事件                                               |
| fnmatch        | 是         | 用于匹配Unix风格文件名模式的函数                             |
| concurrent     | 是         | 提供异步计算(python内置)                                     |
| glob           | 是         | 用于匹配Unix风格路径模式的函数                               |
| io             | 是         | 用于处理I/O流的函数，在python3中，还包含StringIO(python2中有同名的模块),可以像处理文件一样处理字符串 |
| json           | 是         | 用来读写JSON格式数据的函数                                   |
| logging        | 是         | 提供对python内置的日志功能的访问                             |
| multiprocessig | 是         | 可以在应用程序中运行多个子进程，而且提供API让这些子进程看上去像线程一样 |
| operator       | 是         | 提供实现基本的python运算符功能的函数，可以使用这些函数而不是自己写lambda表达式 |
| os             | 是         | 提供对基本的操作系统函数的访问                               |
| random         | 是         | 提供生成伪随机数的函数                                       |
| re             |            | 提供正则表达式功能                                           |
| sched          |            | 提供一个无需多线程的事件调度器                               |
| select         |            | 提供对函数select()和poll()的访问，用于创建事件循环           |
| shutil         |            | 提供对高级文件处理函数的访问                                 |
| signal         |            | 提供用于处理POSIX信号的函数                                  |
| tempfile       |            | 提供用于创建临时文件和目录的函数                             |
| threading      |            | 提供对处理高级线程功能的访问                                 |
| urllib         |            | (python2中的urllib2和urlparse)提供处理和解析url的函数        |
| uuid           |            | 可以生成全局唯一的标识符                                     |



## 时间

```
time		时间相关
calendar	日历
date	    日期与时间
datetime	日期和时间
```
## 集合类

```
collections
```
## 迭代操作

```
itertool
```
## 处理字节类型
```
struct
```
## 上下文管理

```
contextlib
```

## 系统操作

```
sys			控制shell程序
os			与操作系统相关的函数
copy		复制
keyword		关键字
shutil		提供对高级文件处理函数的访问
```

## 测试

```
pdb			调试模块
doctest		文档测试
```
## 摘要算法

```
hashlib     摘要算法
hmac        hmac算法
```
## 操作URl

```
urlib
urlib2(python2)
```
## 网络

| 协议   | 功能用处                         | 端口号 | Python 模块                |
| ------ | -------------------------------- | ------ | -------------------------- |
| HTTP   | 网页访问                         | 80     | httplib, urllib, xmlrpclib |
| NNTP   | 阅读和张贴新闻文章，俗称为"帖子" | 119    | nntplib                    |
| FTP    | 文件传输                         | 20     | ftplib, urllib             |
| SMTP   | 发送邮件                         | 25     | smtplib                    |
| POP3   | 接收邮件                         | 110    | poplib                     |
| IMAP4  | 获取邮件                         | 143    | imaplib                    |
| Telnet | 命令行                           | 23     | telnetlib                  |
| Gopher | 信息查找                         | 70     | gopherlib, urllib          |

## HTML解析
```
HTMLParser
```
## XML
```
XML
```
## 数学
```
math
```

## 数据压缩
```
zlib
gzip
bz2
zipfile
tarfile
```

## 数据获取

```
Scrapy
beautifulsoup
requests
paramiko
```

## 数据运算

```
random		随机数
NumPy		快速数组处理
Scripy		数值运算
```
## 数据存储

```
mysql
hadoop
mangodb
redis
spark
```

## 结果输出

```
matplotlib
VisPy
```

## 图形界面

```
TK
wxWidgets
Qt
GTK
```

## 其他语言交互

```python
# 调用C/C++
ctypes	
Cython
boost.python
SWIG
# 调用Qt
PyQt
# 调用R
rpy2
# 调用java
Jython
JPype
```

## 字符串编码

```
chardet
```
## 图像处理
```
pillow      图片处理库
qrcode      生成二维码
zbar        解析二维码
PyOpenGL
PyOpenCV
mayavi2
```
## 信号处理

```
PyWavelets
scipy.signal
signal
```

## 云系统支持

```
github
sourceforge
EC2
BAT
HPC
```

## 机器学习

```
scikit-learn
TensorFlow
Pytorch
caffe2
Theano
Cognitive Toolkit
Keras
```
## 加速处理

```
pypy
Cython
PyCUDA
```

## 获取系统信息

```

```

## PDF

```
pypdf2
```
## 打包

```
pyinstaller
```




