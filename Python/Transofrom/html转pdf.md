# html转pdf

## pdfkit

### 优缺点

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

### 使用方法

安装依赖[wkhtmltopdf](https://wkhtmltopdf.org/downloads.html)

```
brew cask install wkhtmltopdf
```

安装包

```
pip install pdfkit
pip install wkhtmltopdf
```

简单使用

```python
import pdfkit

# url页面转化为pdf
pdfkit.from_url('http://google.com', 'out.pdf')
# 文件转化为pdf
pdfkit.from_file('test.html', 'out.pdf')
# 文本内容转化为pdf
pdfkit.from_string('Hello!', 'out.pdf')
```

操作对象

```python
# 传入列表
pdfkit.from_url(['google.com', 'yandex.ru', 'engadget.com'], 'out.pdf')
pdfkit.from_file(['file1.html', 'file2.html'], 'out.pdf')

# 传入打开的文件对象
with open('file.html') as f:
  	pdfkit.from_file(f, 'out.pdf')
    
# 继续操作pdf，将其读取成一个string变量
# Use False instead of output path to save pdf to a variable
pdf = pdfkit.from_url('http://google.com', False)
```

### 指定pdf格式

我们可以指定各种选项，就是上面三个方法中的options。
具体的设置可以参考https://wkhtmltopdf.org/usage/wkhtmltopdf.txt 里面的内容。

```python
options = {
    'page-size': 'Letter',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': "UTF-8",
    'custom-header' : [
        ('Accept-Encoding', 'gzip')
    ]
    'cookie': [
        ('cookie-name1', 'cookie-value1'),
        ('cookie-name2', 'cookie-value2'),
    ],
    'no-outline': None
}

pdfkit.from_url('http://google.com', 'out.pdf', options=options)
```

默认的，pdfkit会show出所有的output，如果你不想使用，可以设置为quite

```python
options = {
    'quiet': ''
    }

pdfkit.from_url('google.com', 'out.pdf', options=options)

```

传入任何html标签

```python
body = """
    <html>
      <head>
        <meta name="pdfkit-page-size" content="Legal"/>
        <meta name="pdfkit-orientation" content="Landscape"/>
      </head>
      Hello World!
      </html>
    """

pdfkit.from_string(body, 'out.pdf') #with --page-size=Legal and --orientation=Landscape

```





### API

`from_url()`

```python
def from_url(url, output_path, options=None, toc=None, cover=None,
             configuration=None, cover_first=False):
    """
    Convert file of files from URLs to PDF document

    :param url: url可以是某一个url也可以是url的列表，
    :param output_path: 输出pdf的路径，如果设置为False意味着返回一个string

    Returns: True on success
    """

    r = PDFKit(url, 'url', options=options, toc=toc, cover=cover,
               configuration=configuration, cover_first=cover_first)

    return r.to_pdf(output_path)

```

`from_file()`

```python
def from_file(input, output_path, options=None, toc=None, cover=None, css=None,
              configuration=None, cover_first=False):
    """
    Convert HTML file or files to PDF document

    :param input: 输入的内容可以是一个html文件，或者一个路径的list，或者一个类文件对象
    :param output_path: 输出pdf的路径，如果设置为False意味着返回一个string

    Returns: True on success
    """

    r = PDFKit(input, 'file', options=options, toc=toc, cover=cover, css=css,
               configuration=configuration, cover_first=cover_first)

    return r.to_pdf(output_path)
```

`from_string()`

```python
def from_string(input, output_path, options=None, toc=None, cover=None, css=None,
                configuration=None, cover_first=False):
    #类似的，这里就不介绍了
    r = PDFKit(input, 'string', options=options, toc=toc, cover=cover, css=css,
               configuration=configuration, cover_first=cover_first)
    return r.to_pdf(output_path)
```

