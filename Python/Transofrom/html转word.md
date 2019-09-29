# html转word

将web/html内容导出为world文档，再java中有很多解决方案，比如使用Jacob、Apache POI、Java2Word、iText等各种方式，以及使用freemarker这样的模板引擎这样的方式。php中也有一些相应的方法，但在python中将web/html内容生成world文档的方法是很少的。其中最不好解决的就是如何将使用js代码异步获取填充的数据，图片导出到word文档中。

## unoconv

- 优缺点

功能

```
1.支持将本地html文档转换为docx格式的文档，所以需要先将网页中的html文件保存到本地，再调用unoconv进行转换。转换效果也不错，使用方法非常简单。
```

缺点

```
1.只能对静态html进行转换，对于页面中有使用ajax异步获取数据的地方也不能转换（主要是要保证从web页面保存下来的html文件中有数据）。
2.只能对html进行转换，如果页面中有使用echarts,highcharts等js代码生成的图片，是无法将这些图片转换到word文档中；
3.生成的word文档内容格式不容易控制。
```

- 使用

```shell
# 安装
sudo apt-get install unoconv
# 使用
unoconv -f pdf *.odt
unoconv -f doc *.odt
unoconv -f html *.odt
```

## python-docx

- 优缺点

功能

> 1.python-docx是一个可以读写word文档的python库。

缺点

> 1.功能非常弱。有很多限制比如不支持模板等，只能生成简单格式的word文档。

- 使用方法

获取网页中的数据，使用python手动排版添加到word文档中。

```python
from docx import Document
from docx.shared import Inches
document = Document()
document.add_heading('Document Title', 0)
p = document.add_paragraph('A plain paragraph having some ')
p.add_run('bold').bold = True
p.add_run(' and some ')
p.add_run('italic.').italic = True
document.add_heading('Heading, level 1', level=1)
document.add_paragraph('Intense quote', style='IntenseQuote')
document.add_paragraph(
    'first item in unordered list', style='ListBullet'
)
document.add_paragraph(
    'first item in ordered list', style='ListNumber'
)
document.add_picture('monty-truth.png', width=Inches(1.25))
table = document.add_table(rows=1, cols=3)
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Qty'
hdr_cells[1].text = 'Id'
hdr_cells[2].text = 'Desc'
for item in recordset:
    row_cells = table.add_row().cells
    row_cells[0].text = str(item.qty)
    row_cells[1].text = str(item.id)
    row_cells[2].text = item.desc
document.add_page_break()
document.save('demo.docx')
```



```python
from docx import Document
from docx.shared import Inches
document = Document()
for row in range(9):
    t = document.add_table(rows=1,cols=1,style = 'Table Grid')
    t.autofit = False #很重要！
    w = float(row) / 2.0
    t.columns[0].width = Inches(w)
document.save('table-step.docx')
```

