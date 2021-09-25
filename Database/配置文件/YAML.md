# YAML

YAML 是 "YAML Ain't a Markup Language"（YAML 不是一种标记语言）的递归缩写。在开发的这种语言时，YAML 的意思其实是："Yet Another Markup Language"（仍是一种标记语言）。

YAML 的语法和其他高级语言类似，并且可以简单表达清单、散列表，标量等数据形态。它使用空白符号缩进和大量依赖外观的特色，特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲（例如：许多电子邮件标题格式和YAML非常接近）。

YAML 的配置文件后缀为 **.yml**，如：**runoob.yml** 。

## 基本语法

- 大小写敏感
- 使用缩进表示层级关系
- 缩进不允许使用tab，只允许空格
- 缩进的空格数不重要，只要相同层级的元素左对齐即可
- '#'表示注释

## 数据类型

YAML 支持以下几种数据类型：
```
- 对象：键值对的集合，又称为映射（mapping）/ 哈希（hashes） / 字典（dictionary）
- 数组：一组按次序排列的值，又称为序列（sequence） / 列表（list）
- 纯量（scalars）：单个的、不可再分的值
```

### 对象

对象键值对使用冒号结构表示 **key: value**，冒号后面要加一个空格。

也可以使用 **key:{key1: value1, key2: value2, ...}**。

还可以使用缩进表示层级关系；

```
key: 
    child-key: value
    child-key2: value2
```

较为复杂的对象格式，可以使用问号加一个空格代表一个复杂的 key，配合一个冒号加一个空格代表一个 value

```
?  
    - complexkey1
    - complexkey2
:
    - complexvalue1
    - complexvalue2
```

意思即对象的属性是一个数组 [complexkey1,complexkey2]，对应的值也是一个数组 [complexvalue1,complexvalue2]

### 数组

以 **-** 开头的行表示构成一个数组

```
- A
- B
- C
```

YAML 支持多维数组，可以使用行内表示：

```
key: [value1, value2, ...]
```

数据结构的子成员是一个数组，则可以在该项下面缩进一个空格。

```
-
 - A
 - B
 - C
```

一个相对复杂的例子：

```
companies:
    -
        id: 1
        name: company1
        price: 200W
    -
        id: 2
        name: company2
        price: 500W
```

意思是 companies 属性是一个数组，每一个数组元素又是由 id、name、price 三个属性构成。

数组也可以使用流式(flow)的方式表示：

```
companies: [{id: 1,name: company1,price: 200W},{id: 2,name: company2,price: 500W}]
```

### 复合结构

数组和对象可以构成复合结构，例：

```
languages:
  - Ruby
  - Perl
  - Python 
websites:
  YAML: yaml.org 
  Ruby: ruby-lang.org 
  Python: python.org 
  Perl: use.perl.org
```

转换为 json 为：

```
{ 
  languages: [ 'Ruby', 'Perl', 'Python'],
  websites: {
    YAML: 'yaml.org',
    Ruby: 'ruby-lang.org',
    Python: 'python.org',
    Perl: 'use.perl.org' 
  } 
}
```

### 纯量

纯量是最基本的，不可再分的值，包括：
```
- 字符串
- 布尔值
- 整数
- 浮点数
- Null
- 时间
- 日期
```
使用一个例子来快速了解纯量的基本使用：

```
boolean: 
    - TRUE  #true,True都可以
    - FALSE  #false，False都可以
float:
    - 3.14
    - 6.8523015e+5  #可以使用科学计数法
int:
    - 123
    - 0b1010_0111_0100_1010_1110    #二进制表示
null:
    nodeName: 'node'
    parent: ~  #使用~表示null
string:
    - 哈哈
    - 'Hello world'  #可以使用双引号或者单引号包裹特殊字符
    - newline
      newline2    #字符串可以拆成多行，每一行会被转化成一个空格
date:
    - 2018-02-17    #日期必须使用ISO 8601格式，即yyyy-MM-dd
datetime: 
    -  2018-02-17T15:02:31+08:00    #时间使用ISO 8601格式，时间和日期之间使用T连接，最后使用+代表时区
```

### 引用

**&** 用来建立锚点（defaults），**<<** 表示合并到当前数据，***** 用来引用锚点。

```yaml
defaults: &defaults
  adapter:  postgres
  host:     localhost

development:
  database: myapp_development
  <<: *defaults

test:
  database: myapp_test
  <<: *defaults
```

相当于:

```
defaults:
  adapter:  postgres
  host:     localhost

development:
  database: myapp_development
  adapter:  postgres
  host:     localhost

test:
  database: myapp_test
  adapter:  postgres
  host:     localhost
```

下面是另一个例子:

```yaml
- &showell Steve 
- Clark 
- Brian 
- Oren 
- *showell 
```

转为 JavaScript 代码如下:

```
[ 'Steve', 'Clark', 'Brian', 'Oren', 'Steve' ]
```

## python交互

[参考](https://blog.csdn.net/swinfans/article/details/88770119)

安装

```
pip install pyyaml
```

使用

```python
import yaml

# yaml文件--->python对象
yaml.load()
# python对象--->yaml
yaml.dump()
```

- 加载YAML

```python
yaml.load(stream, Loader=None)
yaml.load_all(stream, Loader=None)
yaml.unsafe_load(stream)
yaml.unsafe_load_all(stream)  
yaml.safe_load(stream)
yaml.safe_load_all(stream)

# yaml.load从不信任的源(例如互联网)接收一个YAML文档并由此构建一个任意的Python对象可能存在一定的风险
# yaml.safe_load方法能够将这个行为限制为仅构造简单的Python对象，如整数或者列表。

# 函数可以接受一个表示YAML文档的字节字符串、Unicode字符串、打开的二进制文件对象或者打开的文本文件对象作为参数。若参数为字节字符串或文件，那么它们必须使用 utf-8 、utf-16 或者 utf-16-le 编码。yaml.load 会检查字节字符串或者文件对象的BOM(byte order mark)并依此来确定它们的编码格式。如果没有发现 BOM ，那么会假定他们使用 utf-8 格式的编码。返回值为一个Python对象。
# 如果字符串或者文件中包含多个YAML文档，那么可以使用 yaml.load_all 函数将它们全部反序列化，得到的是一个包含所有反序列化后的YAML文档的生成器对象。
```

示例

```shell
# 字节字符串、Unicode字符串、打开的二进制文件对象或者打开的文本文件对象
>>> yaml.load("""
... - Hesperiidae
... - Papilionidae
... - Apatelodidae
... - Epiplemidae
... """)
['Hesperiidae', 'Papilionidae', 'Apatelodidae', 'Epiplemidae']
>>> yaml.load("'hello': ''")
{'hello': '\uf8ff'}
>>> with open('document.yaml', 'w') as f:
...     f.writelines('- Python\n- Ruby\n- Java')
... 
>>> stream = open('document.yaml')
>>> yaml.load(stream)
['Python', 'Ruby', 'Java']

# 多个YAML文档
>>> documents = """
... name: bob
... age: 18
... ---
... name: alex
... age: 20
... ---
... name: jason
... age: 16
... """
>>> datas = yaml.load_all(documents)
>>> datas
<generator object load_all at 0x105682228>
>>> for data in datas:
...     print(data)
... 
{'name': 'bob', 'age': 18}
{'name': 'alex', 'age': 20}
{'name': 'jason', 'age': 16}

# 任何类型的Python对象
>>> document = """
... none: [~, null]
... bool: [true, false, on, off]
... int: 55
... float: 3.1415926
... list: [Red, Blue, Green, Black]
... dict: {name: bob, age: 18}
... """
>>> yaml.load(document)
{'none': [None, None], 'bool': [True, False, True, False], 'int': 55, 'float': 3.1415926, 'list': ['Red', 'Blue', 'Green', 'Black'], 'dict': {'name': 'bob', 'age': 18}}

# Python 类的实例
>>> class Person:
...     def __init__(self, name, age, gender):
...         self.name = name
...         self.age = age
...         self.gender = gender
...     def __repr__(self):
...         return f"{self.__class__.__name__}(name={self.name!r}, age={self.age!r}, gender={self.gender!r})"
... 
>>> yaml.load("""
... !!python/object:__main__.Person
... name: Bob
... age: 18
... gender: Male
... """)
Person(name='Bob', age=18, gender='Male')
```

- 转存YAML

```python
yaml.dump(data, stream=None, Dumper=Dumper, **kwds)
yaml.dump_all(documents, stream=None, Dumper=Dumper,
        default_style=None, default_flow_style=False,
        canonical=None, indent=None, width=None,
        allow_unicode=None, line_break=None,
        encoding=None, explicit_start=None, explicit_end=None,
        version=None, tags=None, sort_keys=True) 
yaml.safe_dump(data, stream=None, **kwds)
yaml.safe_dump_all(documents, stream=None, **kwds)

# 参数一：Python对象；参数二；可选参数，用于写入生成的YAML文本，如果不提供这个可选参数，则直接返回生成的YAML文档
# 如果要将多个Python对象序列化到一个YAML流中，可以使用 yaml.dump_all 函数。该函数接受一个Python的列表或者生成器对象作为第一个参数，表示要序列化的多个Python对象。
# 参数含义
stream：指定由于输出YAML流的打开的文件对象。默认值为 None，表示作为函数的返回值返回。
default_flow_style：是否默认以流样式显示序列和映射。默认值为 None，表示对于不包含嵌套集合的YAML流使用流样式。设置为 True 时，序列和映射使用块样式。
default_style：默认值为 None。表示标量不使用引号包裹。设置为 '"' 时，表示所有标量均以双引号包裹。设置为 "'" 时，表示所有标量以单引号包裹。
canonical：是否以规范形式显示YAML文档。默认值为 None，表示以其他关键字参数设置的值进行格式化，而不使用规范形式。设置为 True 时，将以规范形式显示YAML文档中的内容。
indent：表示缩进级别。默认值为 None， 表示使用默认的缩进级别（两个空格），可以设置为其他整数。
width：表示每行的最大宽度。默认值为 None，表示使用默认的宽度80。
allow_unicode：是否允许YAML流中出现unicode字符。默认值为 False，会对unicode字符进行转义。设置为 True 时，YAML文档中将正常显示unicode字符，不会进行转义。
line_break：设置换行符。默认值为 None，表示换行符为 ''，即空。可以设置为 \n、\r 或 \r\n。
encoding：使用指定的编码对YAML流进行编码，输出为字节字符串。默认值为 None，表示不进行编码，输出为一般字符串。
explicit_start：每个YAML文档是否包含显式的指令结束标记。默认值为 None，表示流中只有一个YAML文档时不包含显式的指令结束标记。设置为 True 时，YAML流中的所有YAML文档都包含一个显式的指令结束标记。
explicit_end：每个YAML文档是否包含显式的文档结束标记。默认值为 None，表示流中的YAML文档不包含显式的文档结束标记。设置为 True 时，YAML流中的所有YAML文档都包含一个显式的文档结束标记。
version：用于在YAML文档中指定YAML的版本号，默认值为 None，表示不在YAML中当中指定版本号。可以设置为一个包含两个元素的元组或者列表，但是第一个元素必须为1，否则会引发异常。当前可用的YAML的版本号为1.0、1.1 和1.2。
tags：用于指定YAML文档中要包含的标签。默认值为 None，表示不指定标签指令。可以设置为一个包含标签的字典，字典中的键值对对应各个不同的标签名和值。
```

示例

```python
# 一个Python对象
>>> import yaml
>>> emp_info = { 'name': 'Lex',
... 'department': 'SQA',
... 'salary': 8000,
... 'annual leave entitlement': [5, 10]
... }
>>> print(yaml.dump(emp_info))
annual leave entitlement: [5, 10]
department: SQA
name: Lex
salary: 8000
# 参数的值可以是打开的文本或者二进制文件对象
>>> with open('document.yaml', 'w') as f:
...     yaml.dump(emp_info, f)
... 
>>> import os
>>> os.system('cat document.yaml')
annual leave entitlement: [5, 10]
department: SQA
name: Lex
salary: 8000
0
# 多个Python对象
>>> obj = [{'name': 'bob', 'age': 19}, {'name': 20, 'age': 23}, {'name': 'leo', 'age': 25}]
>>> print(yaml.dump_all(obj))
{age: 19, name: bob}
--- {age: 23, name: 20}
--- {age: 25, name: leo}
# 一个Python类的实例
>>> class Person:
...     def __init__(self, name, age, gender):
...         self.name = name
...         self.age = age
...         self.gender = gender
...     def __repr__(self):
...         return f"{self.__class__.__name__}(name={self.name!r}, age={self.age!r}, gender={self.gender!r})"
... 
>>> print(yaml.dump(Person('Lucy', 26, 'Female')))
!!python/object:__main__.Person {age: 26, gender: Female, name: Lucy}
```

