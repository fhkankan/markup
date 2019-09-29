# python生成pdf

## pdfkit

- 优缺点

功能：

```
1.wkhtmltopdf主要用于HTML生成PDF。
2.pdfkit是基于wkhtmltopdf的python封装，支持URL，本地文件，文本内容到PDF的转换，其最终还是调用wkhtmltopdf命令。是目前接触到的python生成pdf效果较好的。
```

优点：

```
1.wkhtmltopdf：利用webkit内核将HTML转为PDF

webkit是一个高效、开源的浏览器内核，包括Chrome和Safari在内的浏览器都使用了这个内核。Chrome打印当前网页的功能，其中有一个选项就是直接“保存为 PDF”。

2.wkhtmltopdf使用webkit内核的PDF渲染引擎来将HTML页面转换为PDF。高保真，转换质量很好，且使用非常简单。
```

缺点：

```
1.对使用echarts，highcharts这样的js代码生成的图标无法转换为pdf（因为它的功能主要是将html转换为pdf,而不是将js转换为pdf）。对于纯静态页面的转换效果还是不错的。
```

- 使用方法

安装依赖[wkhtmltopdf](https://wkhtmltopdf.org/downloads.html)

```
brew cask install wkhtmltopdf
```

安装pdfkit

```
pip install pdfkit
```

使用

```python
import pdfkit

# url页面转化为pdf
pdfkit.from_url('http://google.com', 'out.pdf')
# 文件转化为pdf
pdfkit.from_file('test.html', 'out.pdf')
# 打开的文件转化为pdf
with open('file.html') as f:
  	pdfkit.from_file(f, 'out.pdf')
# 文本内容转化为pdf
pdfkit.from_string('Hello!', 'out.pdf')
```

## weasyprint

## reportlab

本文实例演示了Python生成pdf文件的方法，是比较实用的功能，主要包含2个文件。具体实现方法如下：

pdf.py文件如下：

```python
#!/usr/bin/python
from reportlab.pdfgen import canvas
def hello():
    c = canvas.Canvas("helloworld.pdf")
    c.drawString(100,100,"Hello,World")
    c.showPage()
    c.save()
hello()
```

diskreport.py文件如下：

```python
#!/usr/bin/env python
import subprocess
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
def disk_report():
    p = subprocess.Popen("df -h", shell=True, stdout=subprocess.PIPE)
#   print p.stdout.readlines()
    return p.stdout.readlines()
def create_pdf(input, output="disk_report.pdf"):
    now = datetime.datetime.today()
    date = now.strftime("%h %d %Y %H:%M:%S")
    c = canvas.Canvas(output)
    textobject = c.beginText()
    textobject.setTextOrigin(inch, 11*inch)
    textobject.textLines('''Disk Capcity Report: %s''' %date)
    for line in input:
        textobject.textLine(line.strip())
    c.drawText(textobject)
    c.showPage()
    c.save()
report = disk_report()
create_pdf(report)
```

## PyPDF2

