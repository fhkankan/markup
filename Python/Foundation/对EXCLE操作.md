[TOC]

# Excel

```shell
# 方法一：
pip install xlrd
pip install xlwt
# 方法二
pip install xlutils
# 方法三
pip install pyExcelerator
# 方法四
pip install XlsxWriter
```

## 读取

```python
#coding=utf-8
import xlrd
'''
文件路径比较重要，要以这种方式去写文件路径不用
'''
# 读取的文件路径
file_path = r'd:/功率因数.xlsx'
# 文件中的中文转码
file_path = file_path.decode('utf-8')
# 获取数据
book = xlrd.open_workbook(file_path)

# 获取sheet
# 通过sheet索引
sheet1 = book.sheet_by_index(0)
# 获得指定索引的sheet名
sheet1_name = book.sheet_names()[0]
print(sheet1_name)
# 通过sheet名字获得sheet对象
sheet1 = book.sheet_by_name(sheet1_name)
# 获取所有工作表
sheets = book.sheets()

# 获取总行数
nrows = sheet1.nrows
# 获取总列数
ncols = sheet1.ncols
# 获取一行的数值
sheet1.row_values(i)
# 获取一列的数值
sheet1.col_values(i)

#获取一个单元格的数值
cell_value = sheet1.cell(a,b).value.encode('utf-8')
cell_value = sheet1.cell_value(a,b).encode('utf-8')
cell_value = sheet1.row(a)[0].value.encode('utf-8')

# 日期格式
 date_time = xlrd.xldate_as_datetime(time_str, 0).strftime("%Y-%m-%d %H:%M:%S")
```

## 写入

```python
# conding:utf-8
import xlwt
# 创建一个Wordbook对象，相当于创建了一个Excel文件
book = xlwt.Workbook(encoding = "utf-8", style_compression = 0)
# 创建一个sheet对象，一个sheet对象对应Excel文件中的一张表格
sheet = book.add_sheet("sheet1", cell_overwrite_ok=True)
# 向表sheet1中添加数据
# 参数(行数，列数， 内容)
sheet.write(0, 0, "EnglishName")
sheet.write(1, 0, "MaYi")
sheet.write(0, 1, "中文名字")
sheet.write(1, 1, "蚂蚁")
# 将以上操作保存到指定的Excel文件中
book.save("name.xls")
```

## 读写

方法一：

```python
#coding=utf-8
#######################################################
#filename:test_xlutils.py
#author:defias
#date:xxxx-xx-xx
#function：向excel文件中写入数据
#######################################################
import xlrd
import xlutils.copy
#打开一个workbook
rb = xlrd.open_workbook('E:\\Code\\Python\\test1.xls') 
wb = xlutils.copy.copy(rb)
#获取sheet对象，通过sheet_by_index()获取的sheet对象没有write()方法
ws = wb.get_sheet(0)
#写入数据
ws.write(1, 1, 'changed!')
#添加sheet页
wb.add_sheet('sheetnnn2',cell_overwrite_ok=True)
#利用保存时同名覆盖达到修改excel文件的目的,注意未被修改的内容保持不变
wb.save('E:\\Code\\Python\\test1.xls')
```

方法二：

```python
#coding=utf-8
#######################################################
#filename:test_pyExcelerator_read.py
#author:defias
#date:xxxx-xx-xx
#function：读excel文件中的数据
#######################################################
import pyExcelerator
#parse_xls返回一个列表，每项都是一个sheet页的数据。
#每项是一个二元组(表名,单元格数据)。其中单元格数据为一个字典，键值就是单元格的索引(i,j)。如果某个单元格无数据，那么就不存在这个值
sheets = pyExcelerator.parse_xls('E:\\Code\\Python\\testdata.xls')
print sheets
```

写

```python
#coding=utf-8
#######################################################
#filename:test_pyExcelerator.py
#author:defias
#date:xxxx-xx-xx
#function：新建excel文件并写入数据
#######################################################
import pyExcelerator
#创建workbook和sheet对象
wb = pyExcelerator.Workbook()
ws = wb.add_sheet(u'第一页')
#设置样式
myfont = pyExcelerator.Font()
myfont.name = u'Times New Roman'
myfont.bold = True
mystyle = pyExcelerator.XFStyle()
mystyle.font = myfont
#写入数据，使用样式
ws.write(0,0,u'ni hao 帕索！',mystyle)
#保存该excel文件,有同名文件时直接覆盖
wb.save('E:\\Code\\Python\\mini.xls')
print '创建excel文件完成！'
```

## 合并单元格

处理excel表格的时候经常遇到合并单元格的情况，使用xlrd中的merged_cells的方法可以获取当前文档中的所有合并单元格的位置信息。

```python
import xlrd

xls = xlrd.open_workbook('test.xls')
# 读取excel并读取第一页的内容
sh = xls.sheet_by_index(0)
# merged_cells返回的是一个列表，每一个元素是合并单元格的位置信息的数组，数组包含四个元素（起始行，结束行，起始列，结束列）
or crange in sh.merged_cells:
    rs, re, cs, ce = crange
# 计算合并的单元格
(rs, cs) : (re - rs, ce - cs) 
```



