# XlsxWriter

## 概述

XlsxWriter是一个python模块，用于以Excel2007+的XLSX文件格式编写文件。

它可用于将文本，数字和公式写入多个工作表，它支持格式化，图像，图表，页面设置，自动过滤，条件格式等功能。

相对于[其他模块](https://xlsxwriter.readthedocs.io/alternatives.html#alternatives) ，具有以下优缺点：

```python
# 优点
1. 它支持比任何替代模块更多的Excel功能。
2. 它与Excel生成的文件具有高度的保真度。在大多数情况下，生成的文件与Excel生成的文件完全等效。
3. 它有大量的文档，示例文件和测试。
4. 它速度很快，可配置为即使对于非常大的输出文件也只使用很少的内存。

# 缺点
不支持读取和更改现存的Excel XLSX文件
```

示例

```python
import xlsxwriter

workbook = xlsxwriter.Workbook('hello.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Hello world')
workbook.close()
```

## 创建文件

- 示例

```python
 from datetime import datetime
 import xlsxwriter

 # Create a workbook and add a worksheet.
 workbook = xlsxwriter.Workbook('Expenses03.xlsx')
 worksheet = workbook.add_worksheet()

 # Add a bold format to use to highlight cells.
 bold = workbook.add_format({'bold': 1})

 # Add a number format for cells with money.
 money_format = workbook.add_format({'num_format': '$#,##0'})

 # Add an Excel date format.
 date_format = workbook.add_format({'num_format': 'mmmm d yyyy'})

 # Adjust the column width.
 worksheet.set_column(1, 1, 15)

 # Write some data headers.
 worksheet.write('A1', 'Item', bold)
 worksheet.write('B1', 'Date', bold)
 worksheet.write('C1', 'Cost', bold)

 # Some data we want to write to the worksheet.
 expenses = (
     ['Rent', '2013-01-13', 1000],
     ['Gas',  '2013-01-14',  100],
     ['Food', '2013-01-16',  300],
     ['Gym',  '2013-01-20',   50],
 )

 # Start from the first cell below the headers.
 row = 1
 col = 0

 for item, date_str, cost in (expenses):
     # Convert the date string into a datetime object.
     date = datetime.strptime(date_str, "%Y-%m-%d")

     worksheet.write_string  (row, col,     item              )
     worksheet.write_datetime(row, col + 1, date, date_format )
     worksheet.write_number  (row, col + 2, cost, money_format)
     row += 1

 # Write a total using a formula.
 worksheet.write(row, 0, 'Total', bold)
 worksheet.write(row, 2, '=SUM(C2:C5)', money_format)

 workbook.close()
```

- 常用函数

创建文件和表

```python
workbook = xlsxwriter.Workbook('Expenses01.xlsx')  # 创建Excel文件
worksheet = workbook.add_worksheet()  # 创建文件表
```

设定数据样式

```python
bold = workbook.add_format({'bold': 1})
money_format = workbook.add_format({'num_format': '$#,##0'})
date_format = workbook.add_format({'num_format': 'mmmm d yyyy'})
```

设定宽度

```
worksheet.set_column(1, 1, 15)
worksheet.set_column('B:B', 15)
worksheet.set_row(1, 30, format)
```

写入数据

```python
# 直接写入数据
worksheet.write(row, col, some_data)
worksheet.write(row, col, some_data, [format])
worksheet.write('A1', some_data, bold)
# 各种类型数据写入
worksheet.write_string()
worksheet.write_datetime()
worksheet.write_number()
worksheet.write_blank()
worksheet.write_formula()
worksheet.write_boolean()
worksheet.write_url()
```

关闭数据文件

```python
workbook.close()
```

## API

### Workbook

```python
Workbook(filename[,options])

# 参数
filename(string)  # Excel文件名
options(dict)  # workbook参数
```

options

| name                  | desc                                                         | demo                                                         |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `constant_memory`     | 减少存储在内存中的数据量，以便有效地写入大文件。一旦此模式处于活动状态，数据应按顺序行顺序写入。因此，`add_table()`和`merge_range()`工作表方法在此模式下不起作用 | `workbook = xlsxwriter.Workbook(filename, {'constant_memory': True})` |
| `tmpdir`              | 在汇编最终的XLSX文件之前，将工作簿数据存储在临时文件中。临时文件在系统的临时目录中创建。如果应用程序无法访问默认临时目录，或者包含的空间不足，则可以使用tmpdir选项指定备用位置 | `workbook = xlsxwriter.Workbook(filename, {'tmpdir': '/home/user/tmp'})` |
| `in_memory`           | 为了避免在最终XLSX文件的程序集中使用临时文件，例如在不允许临时文件的服务器上，设置为`True`。这项会覆盖`constant_memory` | `workbook = xlsxwriter.Workbook(filename, {'in_memory': True})` |
| `strings_to_numbers`  |                                                              |                                                              |
| `strings_to_formulas` |                                                              |                                                              |
| `strings_to_urls`     |                                                              |                                                              |
| `nan_inf_to_errors`   |                                                              |                                                              |
| `default_date_format` |                                                              |                                                              |
| `remove_timezone`     |                                                              |                                                              |
| `date_1904`           |                                                              |                                                              |

