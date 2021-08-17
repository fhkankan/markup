[TOC]

# IO编程

```
IO在计算机中指Input/Output，也就是输入和输出。由于程序和运行时数据是在内存中驻留，由CPU这个超快的计算核心来执行，涉及到数据交换的地方，通常是磁盘、网络等，就需要IO接口。

IO编程中，Stream（流）是一个很重要的概念，可以把流想象成一个水管，数据就是水管里的水，但是只能单向流动。Input Stream就是数据从外面（磁盘、网络）流进内存，Output Stream就是数据从内存流到外面去。对于浏览网页来说，浏览器和新浪服务器之间至少需要建立两根水管，才可以既能发数据，又能收数据。

由于CPU和内存的速度远远高于外设的速度，所以，在IO编程中，就存在速度严重不匹配的问题。有两种办法：
第一种是CPU等着，也就是程序暂停执行后续代码，等100M的数据在10秒后写入磁盘，再接着往下执行，这种模式称为同步IO；
另一种方法是CPU不等待，只是告诉磁盘，“您老慢慢写，不着急，我接着干别的事去了”，于是，后续代码可以立刻接着执行，这种模式称为异步IO。

同步和异步的区别就在于是否等待IO执行的结果,使用异步IO来编写程序性能会远远高于同步IO，但是异步IO的缺点是编程模型复杂。
```

> 文件

按照数据的组织形式，可以把文件分为文本文件和二进制文件

文本文件

```
文本文件存储的是常规字符串，由若干文本行组成，通常每行以换行符'\n'结尾。
实际上文本文件在磁盘上也是以二进制形式存储，只是在读取和查看时使用正确的编码方式进行解码还原为字符串信息，可以直接阅读和理解
```

二进制文件

```
图像文件、音视频文件、可执行文件、资源文件、各种数据库文件、office文档等都属于二进制文件。
二进制文件把信息以bytes进行存储，通常无法直接阅读和理解，需要专门的软件进行解码或反序列化
```

# 文件操作

## 读写

> 直接读写

```python
file1 = open("test.txt") 
file2 = open("output.txt","w") 

while True: 
    line = file1.readline() 
    #这里可以进行逻辑处理 
    file2.write('"'+line[:s]+'"'+",") 
    if not line: 
        break 
#记住文件处理完，关闭是个好习惯 
file1.close() 
file2.close() 
```

> 文件迭代器

从python2.2开始，文件对象是可迭代的。

```python
f_name = open(path)
for line in f_name:
	print('line is', line)
f_name.close()
```

> 文件上下文管理器

```python
#打开文件
#用with..open自带关闭文本的功能
with open('somefile.txt', 'r') as f: 
    data = f.read() 

# loop整个文档
with open('somefile.txt', 'r') as f: 
    for line in f: 
        # 处理每一行

# 写入文本 
with open('somefile.txt', 'w') as f: 
    f.write(text1) 
    f.write(text2) 
    ... 

# 把要打印的line写入文件中 
with open('somefile.txt', 'w') as f: 
    print(line1, file=f) 
    print(line2, file=f)
```

> 懒加载式

按read()和readlines()进行读取时，若不带参数则表示把文件中所有内容加载到内存中，有内存溢出风险。

考虑使用while循环和readline()方法，或使用懒加载模式

```python
import fileinput
for line in fileinput.input(path):
	print('line is', line)
```

## 对象

内置函数open()

```python
fileobj = open(file, mode='r', buffering=-1, encodeing=None, errors=None, newline=None, closefd=True, opener=None)
# 返回可迭代的文件对象(也叫文件描述符或文件流)

# 参数
# file	指定要打开或创建的文件名，可以是绝对路径，也可以是相对路径
# mode	打开文件后的处理模式，默认文本文件只读
# buffering	指定读写文件的缓存模式，默认值是-1。0(二进制模式)表示不会缓存，1(文本模式)表示使用行缓存，>1表示寄存区的缓冲大小(字节)，-1表示二进制和非交互文本文件以固定大小的块为缓存单位，等价于io.DEFAULT_BUFFER_SIZE,交互文件采用行缓存。缓存机制使得修改文件时不需要频繁地进行磁盘文件的读写操作，而是等缓存满了以后再写入文件，或者在需要的时候使用flush()方法强行将缓存中的内容写入磁盘文件，缓冲机制大幅度提高了文件操作速度，同事延长了磁盘使用寿命
# encoding	指定对文本进行编码和解码的方式。只适用文本模式
# newline	表示文件中的换行符，只适用于文本模式，取值可是None、'\n'、'\r'和'\r\n'
    
# 读取文本
# 文件存在，直接打开，文件不存在，报错
f = open('hm.txt','r')

# 写入文本
# 文件存在，直接打开，文件不存在，创建
f = open('hm.txt','w')

# 关闭文件
# 目的是释放文件占用的资源，若没有关闭，虽然python会在文件对象引用计数为0时自动关闭，但是可能会丢失输出缓冲区的数据，若不及时关闭，该文件资源被占用，无法进行其他操作 
f.close()
```

标准库codecs的open()

```python
open(file, mode='r', encoding=None, errors='strict', buffering=-1)
# 返回一个StreamReaderWriter对象
# 参数与内置open()类似，但若指定了encoding会强制使用二进制
```

> 模式

| 模式 | 描述                                                         |
| ---- | ------------------------------------------------------------ |
| r    | 以只读方式打开文件。文件的指针将会放在文件的开头。这是默认模式。 |
| rb   | 以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。这是默认模式。 |
| r+   | 打开一个文件用于读写。文件指针将会放在文件的开头。           |
| rb+  | 以二进制格式打开一个文件用于读写。文件指针将会放在文件的开头。 |
| w    | 打开一个文件只用于写入。若该文件已存在则将其覆盖。若不存在，创建新文件。 |
| wb   | 以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| w+   | 打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| wb+  | 以二进制格式打开一个文件用于读写。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。 |
| a    | 打开一个文件用于追加。若文件已存在，文件指针将会放在文件的结尾，新的内容将会被写入到已有内容之后。若文件不存在，创建新文件进行写入。 |
| ab   | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不存在，创建新文件进行写入。 |
| a+   | 打开一个文件用于读写。如果该文件已存在，文件指针将会放在文件的结尾。文件打开时会是追加模式。如果该文件不存在，创建新文件用于读写。 |
| ab+  | 以二进制格式打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件的结尾。如果该文件不存在，创建新文件用于读写。 |

## 属性

| 属性   | 说明                                     |
| ------ | ---------------------------------------- |
| buffer | 返回当前文件的缓冲区对象                 |
| closed | 判断文件是否关闭，若文件已关闭则返回True |
| fileno | 文件号，一般不需要太关心                 |
| mode   | 返回文件的打开模式                       |
| name   | 返回文件的名称                           |

## 方法

文件读写操作相关函数都会自动改变文件指针的位置

| 方法                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| close()                | 把缓冲区的内容写入文件，同时关闭文件，并释放文件对象         |
| detach()               | 分离并返回底层的缓冲，一旦底层缓冲被分离，文件对象不再可用，不允许做任何操作 |
| flush()                | 把缓冲区的内容写入文件，但不关闭文件                         |
| read([size])           | 从文本文件中读取size个字节(py2)或字符(py3)的内容作为结果返回，或从二进制文件中读取指定数量的字节并返回，若省略size表示读取所有内容 |
| readable()             | 测试当前文件是否可读                                         |
| readline()             | 从文本文件找那个读取一行内容作为结果返回                     |
| readlines()            | 把文本文件中的每行文本作为一个字符串存入列表中，返回该列表，对于大文件会占用较多内存 |
| seek(offset[, whence]) | 把文件指针移动到新的位置，offset表示相对于whence的位置。whence为0表示从文件头开始计算，1表示从当前位置开始结算，2表示从文件尾开始计算，默认0 |
| seekable()             | 测试当前文件是否支持随机访问，若文件不支持随机访问，则调用方法seek()/tell()/truncate时会抛出异常 |
| tell()                 | 返回文件指针的当前位置                                       |
| truncate([size])       | 删除从当前指针位置到文件末尾的内容。若指定了size，则不论指针在什么位置都只留下前size个字节，其余一律删除 |
| write()                | 把字符串s的内容写入文件                                      |
| writable()             | 测试当前文件是否可写                                         |
| writelines(s)          | 把字符串列表写入文本文件，不添加换行符                       |
| next()                 | 返回文件下一行                                               |
| isatty()               | 如果文件连接到一个终端设备，返回True，否则返回False          |

- 实现文件的增量读取

```python
#!/usr/bin/python
fd=open("test.txt",'r') #获得一个句柄
for i in xrange(1,3): #读取三行数据
    fd.readline()
label=fd.tell() #记录读取到的位置
fd.close() #关闭文件

#再次阅读文件
fd=open("test.txt",'r') #获得一个句柄
fd.seek(label,0)# 把文件读取指针移动到之前记录的位置
fd.readline() #接着上次的位置继续向下读取
```

## 特殊文本文件

> json

| JSON类型(UTF-8) | Python类型 |
| --------------- | ---------- |
| {}              | dict       |
| []              | list       |
| "string"        | str        |
| 1234.56         | int/float  |
| true/false      | True/False |
| null            | None       |

列表list

```python
import json

x = [1,2,3]
json.dumps(x)
json.dumps(_)
```

字典dict

```python
# 把Python对象变成一个JSON，json.dumps()/json.dump(),前者把Python中的字典序列化为str，后者把JSON写入一个file-like Object
import json
d = dict(name='Bob', age=20, score=88)
json.dumps(d)

# 把JSON反序列化为Python对象， loads()/load(),前者把JSON的字符串反序列化，后者从file-like Object中读取字符串并反序列化
json_str = '{"age": 20, "score": 88, "name": "Bob"}'
json.loads(json_str)
```

类class

```python
# 类
class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

# 转换函数--->类转字典
def student2dict(std):
    return {
        'name': std.name,
        'age': std.age,
        'score': std.score
    }
    
# 转换函数--->字典转类
def dict2student(d):
    return Student(d['name'], d['age'], d['score'])
    
# 实现转换
>>> s = Student('Bob', 20, 88)
>>> print(json.dumps(s, default=student2dict))
{"age": 20, "name": "Bob", "score": 88}

>>> json_str = '{"age": 20, "score": 88, "name": "Bob"}'
>>> print(json.loads(json_str, object_hook=dict2student))
<__main__.Student object at 0x10cd3c190>

# 把任意class的实例变为dict
print(json.dumps(s, default=lambda obj: obj.__dict__))
```

> csv

```python
import csv

with open('test.csv', 'w', newline='') as fp:
	testWriter = csv.writer(fp, delimiter=' ', quotechar='"')  # 创建writer对象
    testWriter.writerow(['good', 'bad'])  # 写入一行内容
    
with open('test.csv', 'r')
```

> fileInput

显示多个文本文件的内容

```python
import fileinput

with fileinput.input(files=('demo.py', 'test.py')) as f:
    for line in f:
        print(name)
```

> linecahe

访问文本文件指定行内容

```python
import linecache

lineNumber = (0,1,2,5,9,9999)
for line in lineNumber:
    # 使用getline()函数访问指定文件中的某行内容，指定行不存在则返回空字符串
    print(linecache.getline('demo.py', line))
linecache.clearcache()
```

> chardet

判断文本文件的编码格式

```python 
import chardet
import sys

with open(sys.argv[1], 'rb') as fp:
    print(chardet.detect(fp.read()))
```

## 二进制文件

```python
f = open('EDC.jpg', 'rb')
f.write(..., 'wb')
f.close()
```

对于二进制文件，不能使用常规软件直接进行读写，也不能通过python的文件对象直接读取和理解二进制文件的内容。必须正确理解二进制文件结构和序列化规则，然后设计正确的反序列化规则，才能正确理解二进制文件的内容

把变量从内存中变成可存储或传输的过程称之为序列化，在Python中叫pickling，在其他语言中也被称之为serialization，marshalling，flattening等等。

序列化之后，就可以把序列化后的内容写入磁盘，或者通过网络传输到别的机器上。反过来，把变量内容从序列化的对象重新读到内存里称之为反序列化，即unpickling。

python中常用的序列化模块有struct、pickle、shelve、marshal

> pickle

```python
# pickle.dumps()把一个python对象序列化,之后把bytes写入文件。
import pickle
d = dict(name='Bob', age=20, score=88)
s_b = pickle.dumps(d)

# pickle.dump()直接把文件对象序列化后写入一个file-like Object。
f = open('dump.txt', 'wb')
pickle.dump(d, f)
f.close()

# pickle.loads()方法反序列化出对象
import pickle
d = pickle.loads(s_b)

# pickle.load()从一个file-like Object中直接反序列化出对象
f = open('dump.txt', 'rb')
d = pickle.load(f)
f.close()
```

> struct

python中的struct模块对python基本数据类型与用Python字符串格式表示的C语言struct类型进行转换

读写流程

```
写入
使用struct模块需要使用pack()方法把对象按指定的格式进行序列化，然后使用文件对象的write()
方法将序列化的结果写入二进制文件；

读取
需要使用文件对象的read()方法读取二进制文件的内容，然后使用unpack()方法反序列化得到原来的信息
```

struct模块的函数

| 方法                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| pack(fmt, v1, v2, ...) | 按照给定的格式(fmt)把数据封装成字符串(实际是类似于C结构体的字节流) |
| unpack(fmt, string)    | 按照给定的格式(fmt)解析字节流string,返回解析出来的tuple      |
| calcsize(fmt)          | 计算给定的格式(fmt)占用多少字节的内存                        |

Python3 format对照表

```
Format	Ctype	PythonType	seandardSize
x	pad byte 	no value
c	char		bytes of lenght 1	1
b	signed char	 int				1
B	unsigned char int				1
?	_Bool		 bool				1
h	short		int					2
H	unsigned short	int				2
i	int			int					4
I	unsigned int	int				4
l	long		int					4
L	undigned long	int				4
q	long long	int					8
Q	undigned long long	int			8
n	ssize_t			int
N	size_t			int
e	(7)				float			2
f	float			float			4
d	double			float			8
s	char[]			bytes
P	char[]			bytes
p	void*			int
```

读写文件

```python
import struct

n = 130000
x = 96.45
b=True
s='al@中国'
sn = struct.pack('if?', n,x,b)  # 序列化，i表示整数，f表示实数，？表示逻辑值

with open('sample_struct.dat', 'wb') as f:
    f.write(sn)
    f.write(s.encode())  # 字符串需要编码为字节串再写入文件
    
 with open('sample_struct.dat', 'rb') as f:
    sn = f.read(9)
    tu = struct.unpack('if?', sn)  # 使用指定格式反序列化
    n,x,b1 = tu  # 序列解包
    print('n=', n, 'x=', x, 'b1=', b1)
    s = f.read(9)
    s = s.decode()  # 字符串解码
    print('s=', s) 
```

> shelve

可以像字典赋值一样写入和读取二进制文件

```python
import shelve

zhangsan = {'age':36, 'sex':'male', 'address':'SDIBT'}
lisi = {'age';25, 'sex':'female', 'tel':'123456'}

# 以字典形式把数据写入文件
with shelve.open('shelve_test.dat') as fp:
    fp['zahngsan'] = zhangsan  
    fp['lisi'] = lisi
    for i in range(5):
        fp[str(i)] = str(i)
 
# 读取并显示文件内容
with shelve.open('shelve_test.dat') as fp:
    print(fp['zhangsan'])
    print(fp['zhangsan']['age'])
    print(fp['lisi']['tel'])
    print(fp['3'])
```

> marshal

```python
import marshal

x1 = 30		# 待序列化的对象
x2 = 5.0
x3 = [1,2,3]
x4 = (4,5,6)
x5 = {'a':1, 'b':2, 'c':3}
x6 = {7,8,9}
x = [eval('x'+str(i)) for i in range(1,7)] # 把要序列化的对象放到一个列表中

with open('test.dat', 'wb') as fp:  # 创建二进制文件
    marshal.dump(len(x), fp)  # 先写入对象个数
    for item in x:
        marshal.dump(item, fp)  # 把列表中的对象依次序列化并写入文件
        
with open('test.dat', 'rb') as fp:  # 打开二进制文件
    n = marshal.load(fp)  # 获取对象个数
    for i in range(n):
        print(marshal.load(fp))  # 反序列化，输出结果
  

# 与pickle类似，marshal也提供了dumps()和loads()函数来实现数据的序列化和反序列化。序列化后的字节串更短，可以减少磁盘空间或网络宽带的占用
```

> Xlrd/xlwt

对Excel(2003及更低版本)文件读写

```python
from xlwt import *
import xlrd

book = Workbook()  # 创建新的Excel文件
sheet1 = book.add_sheet('First')  # 添加新的worksheet
al = Alignment()
al.horz = Alignment.HORZ_CENTER  # 对齐方式
al.vert = Alignment.VERT_CENTER
borders = Borders()
borders.bottom = Borders.THICK  # 边框样式
style = XFStyle()
style.alignment = al
style.borders = borders
row = sheet1.row(0)  # 获取第0行
row.write(0, 'test', style=style)  # 写入单元格
row = sheet1.row(1)
for i in range(5):
    roe.write(i, i, style=style)  # 写入数字
row.write(5, '=SUM(A2:E2)', style=style)  # 写入公式
book.save(r'D:\test.xls')  # 保存文件

book = xlrd.open_workbook(r'D:\test.xls')
sheet1 = book.sheet_by_name('First')
row = sheet1.row(0)
print(row[0].value)
print(sheet1.row(1)[2].value)
```

> openpyxl

读写Excel(07及以上)

```python
import openpyxl
from openpyxl import Workbook

fn = r'f:\test.xlsx'  # 文件名
wb = Workbook()  # 创建工作簿
ws = wb.create_sheet(title='你好，世界')  # 创建工作表
ws['A1'] = '这是第一个单元格'  # 单元格赋值
ws['B1'] = 3.1415926
wb.save(fn)  # 保存Excel文件

wb = openpyxl.load_workbook(fn)  # 打开已有的Excel文件
ws = wb.worksheets[1]  # 打开指定索引的工作表
print(ws['A1'].value)  # 读取并输出指定单元格的值
ws.append([1,2,3,4,5]) # 添加一行数据
ws.merge_cells('F2:F3')  # 合并单元格
ws['F2'] = "=sum(A2A2:E2)"  # 写入公式
for r in range(10, 15):
    for c in range(3, 8):
        _ = ws.cell(row=r, column=c, value=r*c)  # 写入单元格数据
wb.save(fn)
```

> Python-docx

```python
from docx import Document

doc = Document('demo.docx')  # 读取文档
contents = ''.join(p.text for p in doc.paragraphs)
words = []
# 检查连续重复字
for index,ch in enumerate(contents[:-2]):
    if ch == contents[index+1] or ch == contents[index+2]: 
        word = contents[index:index+3]
        if word not in wors:
            words.append(word)
            print(word)       
```

> zipfile

zip文件的访问与压缩

```python
import zipfile
import os

# 压缩文件名称，读写模式，压缩类型
with zipfile.ZipFile('newZipFile','w',zipfile.ZIP_DEFILATED) as f:
	f.write('oldFile1')
	f.write('oldFile2)

with zipfile.ZipFile('zipFile','r') as f:
    # 获取压缩文件中的目录和文件
    f.namelist()
    # 读取文件内容
    f.read('fileName')
    # 将压缩文件拷贝到其他目录进行解压缩
    for file in f.namelist():
        f.extract(file,'path')
        
# 将整个文件夹压缩       
startdir = './dir_a'
with zipfile.ZipFile('newZipFile','w',zipfile.ZIP_DEFILATED) as f:
    # f.write(r'./dir_a')只能写入空目录
    for dirpath, dirnames, filenames in os.walk(startdir):
        for filename in filenames:
            f.write(os.path.join(dirpath, filename))
```

> rarfile

rar文件的访问

```python
import rarfile

with rarfile.RarFile('demo.rar') as fp:
    for fn in fp.namelist():
        print(fn)
```

> gzip

gz文件的访问与压缩

```python
import os
import tarfile

# 压缩
with tarfile.open('sample.tar', 'w:gz') as tar:
    for name in [f for f in os.listdir('.') if f.endwith('.py')]:
        tar.add(name)

# 解压
with tarfile.open('sample.tar', 'r:gz') as tar:
    tar.extractall(path='sample')
```

> 判断文件是否是PE文件

PE的全称Portable Executale,即可移植的可执行文件，PE文件包括exe/com/dll/ocx/sys/src文件等windows平台上所有可执行文件类型，是windows平台上所有软件和程序能够正常运行的基础。每种文件有独特的specification用来说明文件头结构和内容组织方式，依赖specification来判断文件类型比扩展名更准确

```python
import sys
import os

if len(sys.argv) != 2:
    print('Usage:{0} anotherFile'.format(sys.argv[0]))
    sys.exit()
    
filename = sys.argv[1]  # 获取要检测的文件名
if not os.path.isfile(filename):  # 判断是否为文件
    print(filename + ' is not file.')
    sys.exit()
    
with open(filename, 'rb') as fp:
    flag1 = fp.read(2)  # 读取文件前两个字节
    fp.seek(0x3c)  # 获取PE头偏移
    offset = ord(fp.read(1))
    p.seek(offset)
    flag2 = fp.read(4)  # 获取PE头签名
# 判断是否为PE文件的特征签名
if flag1 == b'MZ' and flg2 = b'PE\x00\x00':
    print(filename + ' is a PE file.')
else:
    print(filename + ' is not a PE file.')
```

> 批量提取PDF文件中的文本转换为TXT记事本文件

安装扩展库pdfminer3k，使用pdf2txt.py对pdf文件进行转换

```python
import os
import sys
import time

pdfs = (pdfs for pdfs in os.listdir('.') if pdfs.endwith('.pdf'))

for pdf1 in pdfs:
    # 替换文件中的指定字符
    pdf = pdf1.replace('', '_').replace('-', '_').replace('&', '_')
    os.rename(pdf1, pdf)
    print('='*30+'\n', pdf)
    
    txt = pdf[:-4] + '.txt'
    exe = '"' + sys.executable + '""'
    pdf2txt = os.path.dirname(sys.executable)
    pdf2txt = pdf2txt + '\\scripts\\pdf2txt.py"-o'
    try:
        # 调用命令行工具pdf2txt.py进行转换
        # 如果pdf加密过，可以改写下面的代码
        # 在-o前面使用-P来指定密码
        cmd = exe + pdf2txt + txt + ' ' + pdf
        os.popen(cmd)
        # 转换需要一定时间，一般小文件2s
        time.sleep(2)
        # 输出转换后的文本，前200个字符
        with open(txt, encoding='utf-8') as fp:
            print(fp.read(200))
    except:
        pass
```

# 上传下载

## 文件上传

```python
header={"ct":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"}
files = {'file':open('D:\\test_data\\summer_test_data_05.txt','rb')}#此处是重点！我们操作文件上传的时候，把目标文件以open打开，然后存储到变量file里面存到一个字典里面
upload_data={"parentId":"","fileCategory":"personal","fileSize":179,"fileName":"summer_text_0920.txt","uoType":1}
upload_res=requests.post(upload_url,upload_data,files=files,headers=header)##此处是重点！我们操作文件上传的时候，接口请求参数直接存到upload_data变量里面，在请求的时候，直接作为数据传递过去
```

## 文件下载

Python开发中时长遇到要下载文件的情况，最常用的方法就是通过Http利用urllib或者urllib2模块。

当然你也可以利用ftplib从ftp站点下载文件。此外Python还提供了另外一种方法requests。

下面来看看三种方法是如何来下载zip文件的：

方法一：

```python
import urllib 
import urllib2 
import requests
print "downloading with urllib" 
url = 'http://www.pythontab.com/test/demo.zip'  
print "downloading with urllib"
urllib.urlretrieve(url, "demo.zip")
```

方法二：

```python
import urllib2
print "downloading with urllib2"
url = 'http://www.pythontab.com/test/demo.zip' 
f = urllib2.urlopen(url) 
data = f.read() 
with open("demo2.zip", "wb") as code:     
    code.write(data)
```

方法三：

```python
import requests 
print "downloading with requests"
url = 'http://www.pythontab.com/test/demo.zip' 
r = requests.get(url) 
with open("demo3.zip", "wb") as code:
     code.write(r.content)
```

看起来使用urllib最为简单，一句语句即可。当然你可以把urllib2缩写成：

```python
f = urllib2.urlopen(url) 
with open("demo2.zip", "wb") as code:
   code.write(f.read())
```

# 文件和目录

## os

os模块除了提供使用操作系统功能和访问文件系统之外，还有大量文件和文件夹操作方法

| 方法                                                    | 说明                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| access(path, mode)                                      | 测试是否可以按照mode指定的权限访问文件                       |
| chdir(path)                                             | 把path设为当前工作目录                                       |
| chmod(path, mode, *, dir_fd=None, follow_symlinks=True) | 改变文件的访问权限                                           |
| curdir                                                  | 当前文件夹                                                   |
| environ                                                 | 包含系统环境变量和值的字典                                   |
| extsep                                                  | 当前操作系统所使用的文件扩展名分隔符                         |
| get_exec_path()                                         | 返回可执行文件的搜索路径                                     |
| getcwd()                                                | 返回当前工作目录                                             |
| listdir(path)                                           | 返回path目录下的问价暖和目录列表                             |
| mkdir(path[, mode=0777])                                | 创建目录，要求上级目录必须存在                               |
| makedirs(path1/path2..., mode=511)                      | 创建多级目录，会根据需要自动创建中间缺失的目录               |
| open(path, flags, mode=0o777,*, dir_fd=None)            | 按照mode指定的权限打开文件，默认权限为可读、可写、可执行     |
| popen(cmd, mode='r', buffering=-1)                      | 创建进程，启动外部程序                                       |
| rmdir(path)                                             | 删除目录，目录中不能有文件或子文件                           |
| remove(path)                                            | 删除指定的文件，要求用户拥有删除文件的权限，并且文件没有只读或其他特殊权限 |
| removedirs(path1/path2...)                              | 删除多级目录，目录中不能有文件                               |
| rename(src, dst)                                        | 重命名文件或目录，可以实现文件的移动，若目标文件已存在则抛异常，不能跨越磁盘或分区 |
| replace(old, new)                                       | 重命名文件或目录，若目标文件已存在则覆盖，不能跨越磁盘或分区 |
| scandir(path='.')                                       | 返回包含指定文件夹中所有DirEntry对象的迭代对象，遍历文件夹时比listdir()更加高效 |
| sep                                                     | 当前操作系统所用的路径分隔符                                 |
| startfile(filepath[, operation])                        | 使用关联的应用程序打开指定文件或启动指定应用程序             |
| stat(path)                                              | 返回文件的所有属性                                           |
| system()                                                | 启动外部程序                                                 |
| truncate(path, length)                                  | 将文件截断，只保留指定长度的内容                             |
| walk(top, topdown=True, onerror=None)                   | 遍历目录树，该方法返回一个元组，包括3个元素，所有路径名、所有目录列表与文件列表 |
| write(fd, data)                                         | 将bytes对象data写入文件fd                                    |

设置环境变量

```python
import os

JAVA_HOME = '/root/jdk'
os.environ["JAVA_HOME"] = JAVA_HOME
```

## os.path

提供了大量用于路径判断、切分、连接及文件夹遍历的方法

| 方法                | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| abspath(path)       | 返回给定路径的绝对路径                                       |
| basename(path)      | 返回指定路径的最后一个组成部分                               |
| commonpath(paths)   | 返回给定的多个路径的最长公共路径                             |
| commonprefix(paths) | 返回给定的多个路径的最长公共前缀                             |
| dirname(p)          | 返回给定路径的文件夹部分                                     |
| exists(path)        | 判断文件是否存在                                             |
| getatime(filename)  | 返回文件的最后访问时间                                       |
| getctime(filename)  | 返回文件的创建时间                                           |
| getsize(filename)   | 返回文件的大小                                               |
| isabs(path)         | 判断path是否为绝对路径                                       |
| isdir(path)         | 判断path是否为文件夹                                         |
| isfile(path)        | 判断path是否为文件                                           |
| join(path, *paths)  | 连接两个或多个path                                           |
| realpath(path)      | 返回给定路径的绝对路径                                       |
| relpath(path)       | 返回给定路径的相对路径，不能跨越磁盘驱动器或分区             |
| samefile(f1,f2)     | 测试f1和f2这两个路径是否引用同一个文件                       |
| split(path)         | 以路径中最后一个斜线为分隔符把路径分隔成两部分，以列表形式返回 |
| splitext(path)      | 从路径中分隔文件的扩展名                                     |
| splitdrive(path)    | 从路径中分割驱动器的名称                                     |

## shutil

提供了较高级的文件和文件夹操作

| 方法                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| copy(src, dst)                                               | 复制文件，新文件具有同样的文件属性，若目标文件已存在则抛出异常 |
| copy2(src, dst)                                              | 复制文件，新文件具有源文件完全一致的属性，包括创建时间、修改时间和最后访问时间等，若目标文件已存在则抛异常 |
| copyfile(src, dst)                                           | 复制文件，不复制文件属性，若目标文件已存在则直接覆盖         |
| copyfileobj(fsrc, fdst)                                      | 在两个文件对象之间复制数据                                   |
| copymode(src, dst)                                           | 把src的模式位复制发哦dst上，之后两者具有相同的模式           |
| copystat(src, dst)                                           | 把src的模式位、访问时间等所有状态都复制到dst上               |
| copytree(src, dst)                                           | 递归复制文件夹                                               |
| disk_usage(path)                                             | 查看磁盘使用情况                                             |
| move(src ,dst)                                               | 移动问价拿货递归移动文件夹，也可以给文件和文件夹重命名       |
| rmtree(path)                                                 | 递归删除文件夹                                               |
| make_archive(base_name, format, root_dir=None, base_dir=None) | 创建TAR或ZIP格式的压缩文件                                   |
| unpack_archive(filename, extract_dir=None, format=None)      | 解压缩压缩文件                                               |

## glob

提供了一些与文件搜索或遍历有关的函数，并允许使用命令行的统配符进行模糊搜索，方便灵活

```python
import glob

glob.glob('*.txt')  # 返回当前文件夹中所有扩展名为txt的文件列表
glob.iglob('c:\\python3.5\\*.*')  # 返回包含指定文件夹找那个所有文件的生成器对象
glob.glob('tools\\**\*.txt', recursive=True)  # 递归查找tools文件夹中所有.txt文件
glob._rlistdir('.')  # 递归遍历当前文件夹中所有文件，返回生成器对象
glob.glob1('dlls', '*.pyd')  # 返回指定文件夹中指定类型的文件列表
for i in glob.glob2('tools', '**'):
    print(i)  # 递归遍历tools文件夹下所有文件
```

## fnmatch

提供了文件名的检查功能，支持通配符的使用

| 方法                           | 说明                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| fnmatch(filename, pattern)     | 检查文件名filename是否与模式pattern相匹配，返回True或False   |
| fnmatchcase(filename, pattern) | 检查文件名filename是否与模式pattern相匹配，返回True或False，区分大小写 |
| filter(names, pattern)         | 返回names中符合pattern的那部分元素构成的列表                 |



# 内存映射

内存映射就是把一个文件映射到内存中，并不是将文件都放入内存中，而只是加载。当程序访问文件的时候，访问到哪里，哪里的数据就被映射到内存中，不会占用太多内存，却有着内存级别的访问速度。

python提供了mmap模块来实现内存映射文件

准备工作

````python
# 创建文件
size = 1000000
with open('data','wb') as f:
    f.seek(size-1)
    f.write(b'\x00')
# 确认文件内容
$ od -x data
````

内存映射

```python
import os, mmap

# 定义工具函数
def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmpa(fd, size, access=access)

with memoery_map('data') as m:
    print(len(m))
    print(m[0:10])
    print(m[0])
    m[0:10] = b'Python-IoT'
    
with open('data', 'rb') as f:
    print(f.read(10))
```

# 内存操作

## StringIO

在内存中读写str

```
# 把str写入StringIO
>>> from io import StringIO
>>> f = StringIO()  # 创建一个StringIO
>>> f.write('hello')  
5
>>> f.write(' ')
1
>>> f.write('world!')
6
>>> print(f.getvalue())  # 获得写入后的str
hello world!

# 读取StringIO
>>> from io import StringIO
>>> f = StringIO('Hello!\nHi!\nGoodbye!')
>>> while True:
...     s = f.readline()
...     if s == '':
...         break
...     print(s.strip())
...
Hello!
Hi!
Goodbye!
```

## BytesIO

操作二进制数据，就需要使用BytesIO

```
# 在内存中读写bytes
# 写入的不是str，而是经过UTF-8编码的bytes。
>>> from io import BytesIO
>>> f = BytesIO()
>>> f.write('中文'.encode('utf-8'))
6
>>> print(f.getvalue())
b'\xe4\xb8\xad\xe6\x96\x87'

# 在内存中读取bytes
>>> from io import BytesIO
>>> f = BytesIO(b'\xe4\xb8\xad\xe6\x96\x87')
>>> f.read()
b'\xe4\xb8\xad\xe6\x96\x87'
```

# 临时文件与目录

若应用程序需要一个临时文件存储数据，但不需要同其他程序共享，则用tempfile模块提供的TemporaryFile方法创建临时文件，其他应用程序无法找到或打开这个文件，因为它并没有引用文件系统表

- 匿名临时文件

```python
import tempfile

# 普通文本模式w+t，二进制w+b
with tempfile.TemporaryFile('w+t') as f:
    f.write('Python')
    f.write('IoT')
    f.seek(0)
    f.read()
```

- 有名临时文件

若临时文件会被多个进程或主机使用，建立一个有名字的文件比较合适

```python
from tempfile import NamedTemporaryFile

with NamedTemporaryFile('w+t') as f:
    print('filename is:', f.name)
```

- 创建临时目录

```python
from tempfile import TemporaryDirectory

with TemporaryDirectory() as dirname:
    print('dirname is:', dirname)
```

# 序列化

## Base64

某些系统中只能使用ASCII字符，Base64就是用来将非ASCII字符的数据转换为ASCII字符的一种方法，特别适合在HTTP和MIME协议下快速传输数据

> `b64encode,b64decode`

```python
import base64
str = "abcd"
# 编码
en = base64.b64encode(str.encode())
# 解码
de = base64.b64decode(en).decode()
```

> `urlsafe_b64encode,urlsafe_b64decode`

对于标准Base64编码后可能有(+)(/),这两种字符不能再URL中使用，需要转换为(-)(_)

```python
import base64
str = "abcd++//"
en = base64.b64decode(str.encode())
# 编码
en_url = base64.urlsafe_b64encode(en)  # b'abcd--__'
# 解码
de_url = base64.urlsafe_b64decode(en_url)  # b'i\xb7\x1d\xfb\xef\xff'
```

## json/ujson

ujson相对json来说，效率更高。

常用类型
```python
# list
x = [1,2,3]
json.dumps(x)
json.dumps()


# dict
import json
d = dict(name='Bob', age=20, score=88)
json.dumps(d)

json_str = '{"age": 20, "score": 88, "name": "Bob"}'
json.loads(json_str)

# 序列化时不转换为unicode码
a  = {"1": "中国"}
json.dumps(a, ensure_ascii=False)
```

类class

```python
# 类
class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score

# 转换函数--->类转字典
def student2dict(std):
    return {
        'name': std.name,
        'age': std.age,
        'score': std.score
    }
    
# 转换函数--->字典转类
def dict2student(d):
    return Student(d['name'], d['age'], d['score'])
    
# 实现转换
>>> s = Student('Bob', 20, 88)
>>> print(json.dumps(s, default=student2dict))
{"age": 20, "name": "Bob", "score": 88}

>>> json_str = '{"age": 20, "score": 88, "name": "Bob"}'
>>> print(json.loads(json_str, object_hook=dict2student))
<__main__.Student object at 0x10cd3c190>

# 把任意class的实例变为dict
print(json.dumps(s, default=lambda obj: obj.__dict__))
```

## pickle

```python
# pickle.dumps()把一个python对象序列化,之后把bytes写入文件。
import pickle
d = dict(name='Bob', age=20, score=88)
s_b = pickle.dumps(d)

# pickle.dump()直接把文件对象序列化后写入一个file-like Object。
f = open('dump.txt', 'wb')
pickle.dump(d, f)
f.close()

# pickle.loads()方法反序列化出对象
import pickle
d = pickle.loads(s_b)

# pickle.load()从一个file-like Object中直接反序列化出对象
f = open('dump.txt', 'rb')
d = pickle.load(f)
f.close()
```

## struct

python中的struct模块对python基本数据类型与用Python字符串格式表示的C语言struct类型进行转换

读写流程

```
写入
使用struct模块需要使用pack()方法把对象按指定的格式进行序列化，然后使用文件对象的write()
方法将序列化的结果写入二进制文件；

读取
需要使用文件对象的read()方法读取二进制文件的内容，然后使用unpack()方法反序列化得到原来的信息
```

struct模块的函数

| 方法                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| `pack(fmt, v1, v2, ...)` | 按照给定的格式(fmt)把数据封装成字符串(实际是类似于C结构体的字节流) |
| `unpack(fmt, string)`    | 按照给定的格式(fmt)解析字节流string,返回解析出来的tuple      |
| `calcsize(fmt)`          | 计算给定的格式(fmt)占用多少字节的内存                        |

Python3 format对照表

```
Format	Ctype			PythonType			seandardSize
x	pad byte 			no value			1
c	char				string of lenght 	1
b	signed char	 		int					1
B	unsigned char 		int					1
?	_Bool		 		bool				1
h	short				int					2
H	unsigned short		int					2
i	int					int					4
I	unsigned int		int or long			4
l	long				int					4
L	undigned long		long				4
q	long long			long				8
Q	undigned long long 	long 				8
n	ssize_t				int
N	size_t				int
e	(7)					float				2
f	float				float				4
d	double				float				8
s	char[]				string				1
P	char[]				string				1
p	void*				long
```

读写文件

```python
import struct

n = 130000
x = 96.45
b=True
s='al@中国'
sn = struct.pack('!if?', n,x,b)  # 序列化，!表示表示适用于网络传输的字节顺序，i表示整数，f表示实数，？表示逻辑值

with open('sample_struct.dat', 'wb') as f:
    f.write(sn)
    f.write(s.encode())  # 字符串需要编码为字节串再写入文件
    
 with open('sample_struct.dat', 'rb') as f:
    sn = f.read(9)
    tu = struct.unpack('!if?', sn)  # 使用指定格式反序列化
    n,x,b1 = tu  # 序列解包
    print('n=', n, 'x=', x, 'b1=', b1)
    s = f.read(9)
    s = s.decode()  # 字符串解码
    print('s=', s) 
```

## shelve

可以像字典赋值一样写入和读取二进制文件

```python
import shelve

zhangsan = {'age':36, 'sex':'male', 'address':'SDIBT'}
lisi = {'age';25, 'sex':'female', 'tel':'123456'}

# 以字典形式把数据写入文件
with shelve.open('shelve_test.dat') as fp:
    fp['zahngsan'] = zhangsan  
    fp['lisi'] = lisi
    for i in range(5):
        fp[str(i)] = str(i)
 
# 读取并显示文件内容
with shelve.open('shelve_test.dat') as fp:
    print(fp['zhangsan'])
    print(fp['zhangsan']['age'])
    print(fp['lisi']['tel'])
    print(fp['3'])
```

## marshal

```python
import marshal

x1 = 30		# 待序列化的对象
x2 = 5.0
x3 = [1,2,3]
x4 = (4,5,6)
x5 = {'a':1, 'b':2, 'c':3}
x6 = {7,8,9}
x = [eval('x'+str(i)) for i in range(1,7)] # 把要序列化的对象放到一个列表中

with open('test.dat', 'wb') as fp:  # 创建二进制文件
    marshal.dump(len(x), fp)  # 先写入对象个数
    for item in x:
        marshal.dump(item, fp)  # 把列表中的对象依次序列化并写入文件
        
with open('test.dat', 'rb') as fp:  # 打开二进制文件
    n = marshal.load(fp)  # 获取对象个数
    for i in range(n):
        print(marshal.load(fp))  # 反序列化，输出结果
  

# 与pickle类似，marshal也提供了dumps()和loads()函数来实现数据的序列化和反序列化。序列化后的字节串更短，可以减少磁盘空间或网络宽带的占用
```
