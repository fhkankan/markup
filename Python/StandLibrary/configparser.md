# configparser

读取配置文件

```
import configparser
```

## 方法

```
read(filename) 
#读取配置文件，直接读取ini文件内容

sections() 
#获取ini文件内所有的section，以列表形式返回

options(sections) 
#获取指定sections下所有options ，以列表形式返回

items(sections) 
#获取指定section下所有的键值对

get(section, option) 
#获取section中option的值，返回为string类型

getint(section,option) 		#返回int类型
getfloat(section, option)  	#返回float类型
getboolean(section,option) 	#返回boolen类型
```

## demo

配置文件test.ini

```
[logging]
level = 20
path =
server =

[mysql]
host=127.0.0.1
port=3306
user=root
password=123456
```

读取与显示

```
import configparser
from until.file_system import get_init_path

conf = configparser.ConfigParser()
file_path = get_init_path()
print('file_path :',file_path)
conf.read(file_path)

sections = conf.sections()
print('获取配置文件所有的section', sections)

options = conf.options('mysql')
print('获取指定section下所有option', options)


items = conf.items('mysql')
print('获取指定section下所有的键值对', items)


value = conf.get('mysql', 'host')
print('获取指定的section下的option', type(value), value)
```

