# 使用Django输出PDF
这篇文档阐述了如何通过使用Django视图动态输出PDF。 这可以通过一个出色的、开源的Python PDF库ReportLab来实现。

动态生成PDF文件的优点是，你可以为不同目的创建自定义的PDF -- 这就是说，为不同的用户或者不同的内容。

例如，Django在kusports.com上用来为那些参加March Madness比赛的人，生成自定义的，便于打印的 NCAA 锦标赛晋级表作为PDF文件。

## 安装ReportLab 
ReportLab库在PyPI上提供。 也可以下载到用户指南 （PDF文件，不是巧合）。 你可以使用pip来安装ReportLab：
```
$ pip install reportlab
```
通过在Python交互解释器中导入它来测试你的安装：
```
>>> import reportlab
```
若没有抛出任何错误，则已安装成功。

## 编写你的视图
使用Django动态生成PDF的关键是，ReportLab API作用于类似于文件的对象，并且Django的 HttpResponse对象就是类似于文件的对象。

这里是一个 “Hello World”的例子：
```python
from reportlab.pdfgen import canvas
from django.http import HttpResponse

def some_view(request):
    # Create the HttpResponse object with the appropriate PDF headers.
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="somefilename.pdf"'

    # Create the PDF object, using the response object as its "file."
    p = canvas.Canvas(response)

    # Draw things on the PDF. Here's where the PDF generation happens.
    # See the ReportLab documentation for the full list of functionality.
    p.drawString(100, 100, "Hello world.")

    # Close the PDF object cleanly, and we're done.
    p.showPage()
    p.save()
    return response
```
代码和注释是不用多说的，但是一些事情需要提醒一下：

1. 响应对象获得了一个特殊的MIME类型， application/pdf。 这会告诉浏览器，文档是个PDF文件而不是HTML文件。 如果你把它去掉，浏览器可能会把输出解释为HTML，会在浏览器窗口中显示一篇丑陋的、可怕的官样文章。

2. 响应对象获取了附加的Content-Disposition协议头，它含有PDF文件的名称。 文件名可以是任意的；你想把它叫做什么都可以。 浏览器会在“另存为...”对话框等中使用。

3. 在此示例中，Content-Disposition头以'附件 '开头。 这样就强制让浏览器弹出对话框来提示或者确认，如果机器上设置了默认值要如何处理文档。 如果你去掉了'attachment;'，无论什么程序或控件被设置为用于处理PDF，浏览器都会使用它。 代码就像这样：
```
response['Content-Disposition'] = 'filename="somefilename.pdf"'
```
4. Hooking into the ReportLab API is easy: Just pass response as the first argument to canvas.Canvas. Canvas函数接受一个类似于文件的对象，而 HttpResponse对象正好合适。

5. 注意所有随后的PDF生成方法都在PDF对象（这个例子是p）上调用，而不是response对象上。

6. 最后，在PDF文件上调用showPage() 和 save()非常重要。

> 注
ReportLab并不是线程安全的。 一些用户报告了一些奇怪的问题，在构建生成PDF的Django视图时出现，这些视图在同一时间被很多人访问。

## 复杂的PDF

如果你使用ReportLab创建复杂的PDF文档，考虑使用io库作为你PDF文件的临时保存地点。 这个库提供了一个类似于文件的对象接口，非常实用。 这个是上面的“Hello World”示例采用 io重写后的样子：
```python
from io import BytesIO
from reportlab.pdfgen import canvas
from django.http import HttpResponse

def some_view(request):
    # Create the HttpResponse object with the appropriate PDF headers.
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="somefilename.pdf"'

    buffer = BytesIO()

    # Create the PDF object, using the BytesIO object as its "file."
    p = canvas.Canvas(buffer)

    # Draw things on the PDF. Here's where the PDF generation happens.
    # See the ReportLab documentation for the full list of functionality.
    p.drawString(100, 100, "Hello world.")

    # Close the PDF object cleanly.
    p.showPage()
    p.save()

    # Get the value of the BytesIO buffer and write it to the response.
    pdf = buffer.getvalue()
    buffer.close()
    response.write(pdf)
    return response
```
## 其他格式
要注意在这些例子中并没有很多PDF特定的东西 -- 只是使用了reportlab。 你可以使用相似的技巧来生成任何格式，只要你可以找到对应的Python库。 关于用于生成基于文本的格式的其它例子和技巧，另见Outputting CSV with Django。

> 请参见
Django Packages提供了一个帮助生成Django的[PDF文件的软件包的比较](https://djangopackages.org/grids/g/pdf/) 。

