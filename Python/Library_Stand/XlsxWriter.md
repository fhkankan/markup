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

## [API](https://xlsxwriter.readthedocs.io/contents.html)

| Class Name                                                   | Desc                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Workbook](https://xlsxwriter.readthedocs.io/workbook.html)  | Workbook类是XlsxWriter模块公开的主类，它是您需要直接实例化的唯一类。Workbook类表示在Excel中看到的整个电子表格，在内部表示Excel文件写在磁盘上。 |
| [Worksheet](https://xlsxwriter.readthedocs.io/worksheet.html) | 工作表类表示Excel工作表。它处理诸如将数据写入单元格或格式化工作表布局等操作.工作表对象不直接实例化。而是通过从Workbook()对象调用add_worksheet()方法来创建新工作表 |
| [Foramt](https://xlsxwriter.readthedocs.io/format.html)      | 可用于在Excel中格式化单元格的方法和属性。可以格式化的单元格的属性包括：字体，颜色，图案，边框，对齐和数字格式。 |
| [Chart](https://xlsxwriter.readthedocs.io/chart.html)        | Chart模块是在XlsxWriter中实现图表的模块的基类。本节中的信息适用于所有可用的图表子类，例如Area，Bar，Column，Donut，Line，Pie，Scatter，Stock和Radar。通过Workbook add_chart()方法创建图表对象，其中指定了图表类型 |
| [Chartsheet](https://xlsxwriter.readthedocs.io/chartsheet.html) | 在Excel中，图表是一个仅包含图表的工作表.Chartsheet类具有数据工作表的一些功能，例如选项卡选择，页眉，页脚，边距和打印属性，但其主要目的是显示单个图表。这使得它与可以具有一个或多个嵌入图表的普通数据工作表不同.与数据工作表一样，图表表对象不直接实例化。而是通过从Workbook对象调用add_chartsheet()方法来创建新的图表. |
| [Exception](https://xlsxwriter.readthedocs.io/exceptions.html) | Exception类包含XlsxWriter可以引发的各种异常。通常，XlsxWriter仅针对不可恢复的错误或导致文件损坏的错误（例如创建具有相同名称的两个工作表）引发异常。 |

### Workbook

#### Constructor

```python
Workbook(filename[,options])  

# 创建一个新的XlsxWriter Workbook对象
# 返回一个workbook对象
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
| `strings_to_numbers`  | 启用`workheet.write()`方法，尽可能使用`float()`将字符串转换为数字，以避免出现有关“存储为文本的数字”的Excel警告。默认值为False。 | `workbook = xlsxwriter.Workbook(filename, {'strings_to_numbers': True})` |
| `strings_to_formulas` | 启用`worksheet.write()`方法将字符串转换为公式。默认值为True  | `workbook = xlsxwriter.Workbook(filename, {'strings_to_formulas': False})` |
| `strings_to_urls`     | 启用`worksheet.write()`方法将字符串转换为URL。默认值为True。 | `workbook = xlsxwriter.Workbook(filename, {'strings_to_urls': False})` |
| `nan_inf_to_errors`   | 启用`worksheet.write()`和`write_number()`方法将`nan，inf和-inf`转换为Excel错误。Excel不会将`NAN / INF`作为数字处理，因此解决方法将它们映射到产生错误代码`#NUM！和＃DIV/0！`的公式。默认值为False。 | `workbook = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})` |
| `default_date_format` | 此选项用于指定默认日期格式字符串，以便在未给出显式格式时与`worksheet.write_datetime()`方法一起使用。 | `xlsxwriter.Workbook(filename, {'default_date_format': 'dd/mm/yy'})` |
| `remove_timezone`     | Excel不支持日期时间/时间的时区，用户应该根据他们的要求以某种有意义的方式转换和删除时区。或者，remove_timezone选项可用于从datetime值中删除时区。默认值为False。 | `workbook = xlsxwriter.Workbook(filename, {'remove_timezone': True})` |
| `date_1904`           | Excel for Windows使用默认纪元1900而Excel for Mac使用1904纪元。但是，任一平台上的Excel都将在一个系统和另一个系统之间自动转换。默认情况下，XlsxWriter以1900格式存储日期。如果要更改此设置，可以使用date_1904工作簿选项。此选项主要用于增强与Excel的兼容性，通常不需要经常使用 | `workbook = xlsxwriter.Workbook(filename, {'date_1904': True})` |

示例

```python
# with
with xlsxwriter.Workbook('hello_world.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'Hello world')
    
# 使用BytesIO将文件写入内存中的字符串
from io import BytesIO

output = BytesIO()
workbook = xlsxwriter.Workbook(output)
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Hello')
workbook.close()

xlsx_data = output.getvalue()
```

#### add_worksheet

```python
add_worksheet([name])  

# 将新工作表添加到工作簿
# 返回一个worksheet对象
# 参数
name(string)  # 可选工作表名，缺省则默认为Sheet1等
```

示例

```python
worksheet1 = workbook.add_worksheet()           # Sheet1
worksheet2 = workbook.add_worksheet('Foglio2')  # Foglio2
worksheet3 = workbook.add_worksheet('Data')     # Data
worksheet4 = workbook.add_worksheet()           # Sheet4
```

#### add_format

```python
add_format([properties])  

# 创建一个新的Format对象以格式化工作表中的单元格
# 返回一个format对象
# 参数
properties(dict)  # 可选的格式属性字典
```

示例

```python
format1 = workbook.add_format(props)  # Set properties at creation.
format2 = workbook.add_format()       # Set properties later.
```

#### add_chart

```python
add_chart(options)

# 创建可以添加到工作表的图表对象
# 返回一个图表对象
# 参数
options(dict)  # 图表类型选项的字典
```

options

| name    | desc                          | value                                                        |
| ------- | ----------------------------- | ------------------------------------------------------------ |
| type    | 必选                          | area,bar,column,doughnut,line,pie,radar,scatter,stock        |
| subtype | 可选,用于定义可用的图表子类型 | area: stacked, percent_stacked  <br>bar: stacked,percent_stacked<br>column:stacked,percent_stacked  <br>scatter:straight_with_markers,straight ,smooth_with_markers,smooth  <br>radar: with_markers,filled |

示例

```python
chart = workbook.add_chart({'type': 'column'})
workbook.add_chart({'type': 'bar', 'subtype': 'stacked'})
```

#### add_chartsheet

```python
add_chartsheet([sheetname])

# 将新的add_chartsheet添加到工作簿
# 返回一个chartsheet对象
# 参数
sheetname(string)  # 可选的图表表名称，默认为Chart1等
```

#### close

```python
close()

# 关闭Workbook对象并编写XLSX文件
```

示例

```python
workbook.close()

# with中不必显式声明
With xlsxwriter.Workbook('hello_world.xlsx') as workbook:
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'Hello world')
```

#### set_size

```python
set_size(width, height)

# 设定workbook窗口的尺寸
# 参数
width(int)  # 窗口宽度的像素值
height(int)  # 窗口高度的像素值
```

示例

```
workbook.set_size(1200, 800)
```

#### set_tab_ratio

```python
set_tab_ration(tab_ratio)

# 设置工作表选项卡和水平滑块之间的比率
# 参数
tab_ratio(float)  # 比率值在0～100之间,默认60
```

示例

```
workbook.set_tab_ratio(75)  
```

