# IO编程

```
IO在计算机中指Input/Output，也就是输入和输出。由于程序和运行时数据是在内存中驻留，由CPU这个超快的计算核心来执行，涉及到数据交换的地方，通常是磁盘、网络等，就需要IO接口。

IO编程中，Stream（流）是一个很重要的概念，可以把流想象成一个水管，数据就是水管里的水，但是只能单向流动。Input Stream就是数据从外面（磁盘、网络）流进内存，Output Stream就是数据从内存流到外面去。对于浏览网页来说，浏览器和新浪服务器之间至少需要建立两根水管，才可以既能发数据，又能收数据。

由于CPU和内存的速度远远高于外设的速度，所以，在IO编程中，就存在速度严重不匹配的问题。有两种办法：
第一种是CPU等着，也就是程序暂停执行后续代码，等100M的数据在10秒后写入磁盘，再接着往下执行，这种模式称为同步IO；
另一种方法是CPU不等待，只是告诉磁盘，“您老慢慢写，不着急，我接着干别的事去了”，于是，后续代码可以立刻接着执行，这种模式称为异步IO。

同步和异步的区别就在于是否等待IO执行的结果,使用异步IO来编写程序性能会远远高于同步IO，但是异步IO的缺点是编程模型复杂。
```

# 文件操作

##文件的打开

```python
#文件格式：文本文件、二进制文件
fileobj = open(filename[,mode[,buffering]])
# 返回代表连接的文件对象(也叫文件描述符或文件流)
# filename可以是绝对路径，也可以是相对路径
# mode是文件模式
# buffering，若为0则不会有寄存，为1,访问文件就会寄存行，>1表示寄存区的缓冲大小(字节)，<0表示系统默认值

    
# 读取文本
# 文件存在，直接打开，文件不存在，报错
f = open('hm.txt','r')

# 写入文本
# 文件存在，直接打开，文件不存在，创建
f = open('hm.txt','w')

# 关闭文件
f.close()

# 文件读写关
with open('路径', 'r') as f:
    print(f.read())
```

## 文件的模式

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

## 二进制文件的读写

```
# utf-8中3~4个字节代表每个字符

# 以二进制文件写入
# f = open("new.txt",'wb')
# 汉字->转码->二进制保存
# f.write("中国".encode('utf-8'))
# f.close()

# 读取二进制文件
# f = open('new.txt','rb')
# # 二进制->解码->中文字符串
# 三个字符为一个汉字
# result = f.read(3).decode('utf-8')
# print(result)
# f.close()
```

## 读写关定

### 读取

```
f.read([size])
# 读取指定的字节数，若是size未给定或为负数则读取所有
# 返回字符串
# 默认r格式以文本格式保存数据，无论是字符还是中文，都是一个字符

f.readline([size])
# 读取整行，包括”\n“字符
# 返回字符串

f.readlines([sizeint])
# 读取所有行，若sizeint>0,返回总和大约为sizeint字节的行，实际读取可能偏大，因为需要填充缓冲区
# 返回的是一个列表，其中每一行的数据为一个元素

# 文件读取到末尾，会返回空字符串，可做结束判定
```

### 写入

```
f.write(str)
# 只能写入字符串

f.writelines(sequence)
# 向文件写入一个序列字符串列表

# 写方法不能自动在字符串末尾添加换行符，需要自己添加'\n'
```

### 关闭

```
f.close()
# 关闭文本，关闭文本后不能进行读写操作
# 目的是释放文件占用的资源，若没有关闭，虽然python会在文件对象引用计数为0时自动关闭，但是可能会丢失输出缓冲区的数据，若不及时关闭，该文件资源被占用，无法进行其他操作 


# 打开关闭文件一体化
# with open
with open ('new.txt') as file_object:
    contents = file_object.read()
    print(contents)
```

### 定位

```
f.tell()
# 返回文件中光标当前位置和开始位置之间的字节偏移量

f.seek(offset[,whence])
# 设置文件中光标当前位置，offset是偏移字节数，可以取负值，whence是引用点
# offset：正值表示从前至后，负值表示从后至前
# whence：0是文件开始处(默认)，1是当前位置，2是文件结尾
# 在追加模式(a,a+)下打开文件，不能使用seek函数进行定位追加
seek(5,0):表示从文件的开头偏移5个字符
seek(3,1):表示从文件当前位置向后偏移3个字符
seek(-5,2):表示从文件的末尾向前偏移5个字符 
```

### 其他操作

```
f.next()
# 返回文件下一行

f.truncate([size])
# 从文件的首行首字符开始截断，截断文件为size个字符，无size表示从当前位置阶段；截断之后V后面的所有字符被删除，其中Windows系统戏下的换行代表2个字符大小

f.flush()
# 刷新文件内部缓存,直接把内部缓冲区的数据写入文件，而不是被动地等待输出缓冲区写入

f.fileno()
# 返回一个整型的文件描述符，可用在如os模块的read方法等一些底层操作上

f.isatty()
# 如果文件连接到一个终端设备，返回True，否则返回False
```

## 对内容迭代

- 按字节处理

```
read()方法对写入的文件的每个字符进行了循环，运行到文件末尾时，read()方法返回一个空字符串。
```

- 按行操作

```
readline()方法按行读取字符
```

- 懒加载式

按read()和readlines()进行读取时，若不带参数则表示把文件中所有内容加载到内存中，有内存溢出风险。

考虑使用while循环和readline()方法，或使用懒加载模式

```
import fileinput
for line in fileinput.input(path):
	print('line is', line)
```

- 文件迭代器

从python2.2开始，文件对象是可迭代的。

```
f_name = open(path)
for line in f_name:
	print('line is', line)
f_name.close()
```

## 文件的重命名

```
os.rename(current_file_name, new_file_name)
```

## 文件的删除

```
os.remove(path)
# 删除路径为path的文件。如果path 是一个文件夹，将抛出OSError; 查看下面的rmdir()删除一个 directory。

os.removedirs(path)
# 递归删除目录
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

# 序列化

把变量从内存中变成可存储或传输的过程称之为序列化，在Python中叫pickling，在其他语言中也被称之为serialization，marshalling，flattening等等。

序列化之后，就可以把序列化后的内容写入磁盘，或者通过网络传输到别的机器上。

反过来，把变量内容从序列化的对象重新读到内存里称之为反序列化，即unpickling。

## Pickle

Python提供了`pickle`模块来实现序列化

```
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
d

# pickle.load()从一个file-like Object中直接反序列化出对象
f = open('dump.txt', 'rb')
d = pickle.load(f)
f.close()
d
```

## JSON

| JSON类型(UTF-8) | Python类型 |
| --------------- | ---------- |
| {}              | dict       |
| []              | list       |
| "string"        | str        |
| 1234.56         | int/float  |
| true/false      | True/False |
| null            | None       |

> 字典dict

```
# 把Python对象变成一个JSON，json.dumps()/json.dump(),前者把Python中的字典序列化为str，后者把JSON写入一个file-like Object
>>> import json
>>> d = dict(name='Bob', age=20, score=88)
>>> json.dumps(d)
'{"age": 20, "score": 88, "name": "Bob"}'

# 把JSON反序列化为Python对象， loads()/load(),前者把JSON的字符串反序列化，后者从file-like Object中读取字符串并反序列化
>>> json_str = '{"age": 20, "score": 88, "name": "Bob"}'
>>> json.loads(json_str)
{'age': 20, 'score': 88, 'name': 'Bob'}

```

> 类class

```
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

## struct

python中的struct模块对python基本数据类型与用Python字符串格式表示的C语言struct类型进行转换

struct模块的函数

```
- pack(fmt, v1, v2, ...)  按照给定的格式(fmt)把数据封装成字符串(实际是类似于C结构体的字节流)
- unpack(fmt, string)  按照给定的格式(fmt)解析字节流string,返回解析出来的tuple
- calcsize(fmt)  计算给定的格式(fmt)占用多少字节的内存
```

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

## Base64

某些系统中只能使用ASCII字符，Base64就是用来将非ASCII字符的数据转换为ASCII字符的一种方法，特别适合在HTTP和MIME协议下快速传输数据

> `b64encode,b64decode`

```
import base64
str = "abcd"
# 编码
en = base64.b64encode(str.encode())
# 解码
de = base64.b64decode(en).decode()
```

> `urlsafe_b64encode,urlsafe_b64decode`

对于标准Base64编码后可能有(+)(/),这两种字符不能再URL中使用，需要转换为(-)(_)

```
import base64

# 编码
en = base64.urlsafe_b64encode(byt.encode())
# 解码
de = base64.urlsafe_b64decode(en).decode()
```



