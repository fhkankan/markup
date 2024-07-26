[TOC]

# CSV

CSV(Comma-Separated Values)也就是逗号分隔值。有时分隔符也可以不是逗号。文件以纯文本的形式存储表格数据(数字和文本)。纯文本意味着文件是一个字符序列，不含二进制数字那样被解读的数据

csv是一种通用的、相对简单的文件格式，在程序之间转移表格数据。

##  读取

> 读取每一行

```
import csv

with open('env.csv) as f:
	# 使用csv模块读取文件，分隔符是，
	readCSV = csv.reader(f, delimitr=',')
	# 读取每一行数据
	for row in readCSV:
		print(row)
```

> 读取为字典组成的列表

```
import csv

with open('env.csv) as f:
	readCSV = csv.DictReader(f)
	return [row for row in readCSv]
```

> 通过第一行数类型的关键字来读取

```
import csv
form collections import namedtuple

with open('env.csv') as f:
	f_csv = csv.reader(f)
	headings = next(f_csv)
	Row = namedtuple('Row', headings)
	for r in f_csv:
		row = Row(*r)
		# 首行为env_type，env_value
		print(row.env_type)
		print(row.env_value)
```

> 读取每一列

```
import csv

with open('env.csv) as f:
	readCSV = csv.reader(f, delimitr=',')
	list_type = []
	list_value = []
	for row in readCSV:
		str_type = row[0]
		str_value = roe[1]
		list_type.append(str_type)
		list_value.append(str_value)
	
	print(lsit_type)
	print(list_value)
```

## 写入

> 从列表写入

```
import csv

headers = ['name', 'age', 'weight']
rows = [('a', 20, 64),('b', 34, 68)]
with open('demo.csv', 'w',  encoding="utf-8", newline="") as f:
	f_csv = csv.writer(f)
	# 写入单行
	f_csv.writerow(headers)
	# 写入多行
	f_csv.writerows(rows)
```

> 从字典写入

```
import csv

headers = ['name', 'age', 'weight']
rows = [{'name':'a','age':20,'weight':64},
	    {'name':'b','age':34,'weight':68}]
with open('demo.csv', 'w',  encoding="utf-8", newline="") as f:
	f_csv = csv.DictWriter(f, headers)
	f_csv.writeheader()
	f_csv.writerows(rows)
```

