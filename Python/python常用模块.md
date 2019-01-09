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

## 密码学

```
hashlib
zlib
hmac
pycryptodome
cryptography
```

## 系统运维

```
diff		可以比较文件差异并可以生曾呢不同格式的比较结果
filecmp		用与实现文件与文件夹的差异比较
smtplib,poplib,ftplib	邮件收发与FTP空间访问
ansible-playbook		轻量级多主机部署与配置管理系统
dnspython	DNS工具包，支持几乎所有记录类型
ipy			用于管理IPv4和IPv6地址与网络的工具包
paramiko	提供了SHHv2协议的服务端和客户端功能
psutil		可以获取内存、cpu、磁盘、网络的使用情况，查看系统进程和线程信息，并具有一定的进程和线程管理功能
pyclamad	提供了免费开源杀毒软件Clam Antivirus的访问接口
pycurl		对libcur的封装，类似于标准库urlib，但功能更强大
python-rrdtool			提供了rddtool的访问接口，rddtool主要用来跟踪对象的变化情况并生曾呢走势图，例如业务的访问流量，系统性能，磁盘利用率等趋势图
scapy		交互式数据包处理工具包，支持各种网络数据包的解析和伪造
xlrd,xlwt,openpyxl		支持不同版本Excel文件的读写操作，包括数字、文本、公式、图表
```

## 系统相关

```
# windows
ctypes		提供了访问.dll或.so等不同类型动态链接库中函数的接口，很好支持与C/C++等混合编程的需求，可以调用操作系统底层API函数
os			调用window系统内部命令和外部程序，提供一定的文件与文件夹管理功能以及进程管理功能	
platform	扩平台的标准库，实现了与系统平台有关的部分功能，如查看机型、CPU、操作系统类型等信息
winreg		提供了用于操作Windows系统注册表的大部分功能
wmi			提供了windows Managerment Instrumentation的访问接口
py2exe		可以把python程序打包程脱离python解释器环境并独立运行在windows平台上的所有操作
pywin32		对windows底层API进行了封装，几乎支持windows平台上的所有操作

# ubuntu
pyinstaller
```

## PDF

```
pypdf2
```
## 软件工程

```
PyEmu	可编写脚本的模拟器，对恶意软件分析很有用
Immunity Debugger	著名调试器
Paimei	逆向工程框架
ropper	成熟的ROP Gradgets查找与可执行文件分析工具
WinAppDbg	纯python调试器，无本机代码，使用ctypes封装了很多与调试相关的Win32 API调用，并为操作线程和进程提供了有效抽象
YARA	恶意软件识别和分类引擎可以利用YARA创建规则以检测字符串、入侵序列、正则表达式、字节模式等。
pefile	读取和处理PE文件
IDAPython	IDA插件，式运行于交互式反汇编器IDA的插件，用于实现IDA的Python编程接口
Hex-Rays Decompiler	IDA插件，反编译
PatchDiff2		IDA插件，用于补丁对比
BinDiff 		IDA插件，用于二进制文件差异比较
hidedebug		Immunity Debugger插件，可以隐藏调试器的村子啊，用来对抗某些通用的反调试技术
IDAStealth		IDA插件，可隐藏IDA debugger的存在，对抗某些通用的饭调试技术
MyNav			IDA插件，帮助逆向工程师完成典型的任务，如发现特定功能或任务由哪些函数实现，找出补丁前后函数的不同之处和数据入口
Lobotomy		应用与Python的安卓渗透测试工具包，可以帮助安全研究员评估不同Android逆向工程任务
```




