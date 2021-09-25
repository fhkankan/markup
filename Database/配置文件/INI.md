# INI

在早期的windows桌面系统中主要是用INI文件作为系统的配置文件，从win95以后开始转向使用注册表，但是还有很多系统配置是使用INI文件的。其实INI文件就是简单的text文件，只不过这种txt文件要遵循一定的INI文件格式。现在的WINCE系统上也常常用INI文件作为配置文件，这次研究INI文件的目的就是为了我的GPS定位系统客户端写个系统配置文件。“.INI ”就是英文 “initialization”的头三个字母的缩写；当然INI file的后缀名也不一定是".ini"也可以是".cfg"，".conf ”或者是".txt"。

## 要素

最基本的三个要素是：parameters，sections和comments。

- parameters

INI所包含的最基本的“元素”就是parameter；每一个parameter都有一个name和一个value，name和value是由等号“=”隔开。name在等号的左边。

```
name = value
```

- sections

所有的parameters都是以sections为单位结合在一起的。所有的section名称都是独占一行，并且sections名字都被方括号包围着（[ and ])。在section声明后的所有parameters都是属于该section。对于一个section没有明显的结束标志符，一个section的开始就是上一个section的结束，或者是end of the file。Sections一般情况下不能被nested，当然特殊情况下也可以实现sections的嵌套。

```
[section]
```

- comments

在INI文件中注释语句是以分号“；”开始的。所有的所有的注释语句不管多长都是独占一行直到结束的。在分号和行结束符之间的所有内容都是被忽略的。

```
；comments text
```

## 实例

```
;first section
[Section1 Name] 
KeyName1=value1 
KeyName2=value2 
...

[Section2 Name] 
KeyName1=value1 
KeyName2=value2
```

## python

配置

```
[xiong]
name = xiong
age = 23
gender = male

[ying]
name = ying
age = 24
gender = female

[cai]
host_ip = 127.0.0.1
host_name = test1
user1 = root
user1_passwd = 123456
user2 = hdfs
user2_passwd = abcdef
[host]
IP=127.0.0.1
PORT=8090

[auth]
user='Alex'
passwd='123456'
```

实例

```python
from configparser import ConfigParser  
from pathlib import Path 
BASE_DIR = Path(__file__).absolute().parent
ini_path = Path(BASE_DIR).joinpath("my.ini")
cf = ConfigParser(allow_no_value=True)
cf.read(ini_path， encoding="utf-8")

# 读取
print(cf.sections())  # 获取配置文件中的sections列表
print("sections-0",cf.sections()[0])
for section in cf.sections(): 
    print(section)  
    print(cf.items(section))
print(cf.items('xiong'))  # 返回xiong这个section中的子项，元组组成的列表
print(cf.options('xiong')) # 返回xiong这个section中的变量key组成的列表
print("0:",cf.items('cai')[0])

print(cf.has_section('xiong'))  #是否有此section，返回布尔值
print(cf.has_option('xiong','age')) #判断此section中是否有变量key，返回布尔值

print(cf.get('xiong','age'))  #获取这个section中变量key的值
print(cf.get('host','ip'))


# 设置
cf.remove_section('xiong')   # 移除xiong这个section
cf.add_section("cai-3")  # 添加一个叫cai的section
cf.set('cai-3','host','192.168.0.2') # 设置section下的变量key的值，不存在则创建，存在则修改
cf.write(open(ini_path,'w'))  #保存文件
```

