# 视图-文件上传
当Django在处理文件上传的时候，文件数据被保存在request. FILES
注意:接受来自不被信任用户的文件上传是有安全风险的
## 基本文件上传
考虑一个简单的表单，它含有一个FileField：
```python
# forms.py
from django import forms

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()
```
处理这个表单的视图会在request中接收到上传文件的数据。FILES是个字典，它包含每个FileField的键 （或者 ImageField，FileField的子类）。这样的话就可以用`request.FILES['file']`来存放表单中的这些数据了。

注意request.FILES 只有在请求方法为POST，并且发送请求的`<form>`拥有`enctype="multipart/form-data"` 属性时，才会包含数据。否则`request.FILES` 为空。

大多数时候，你将只是简单地从request向表单中传递数据，就像绑定上传文件到表单描述的那样。类似下面这样：
```python
# views.py
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm

# Imaginary function to handle an uploaded file.
from somewhere import handle_uploaded_file

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})
```
注意我们必须向表单的构造器中传递request.FILES。这是文件数据绑定到表单的方法。

这里是一个普遍的方法，可能你会采用它来处理上传文件：
```python
def handle_uploaded_file(f):
    with open('some/file/name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
```
遍历`UploadedFile.chunks()`，而不是使用`read()`，能确保大文件并不会占用系统过多的内存。
`UploadedFile `对象也拥有一些其他的方法和属性

- 使用模型处理上传文件

如果你在Model上使用`FileField`保存文件，使用`ModelForm`可以让这个操作更加容易。调用`form.save()`的时候，文件对象会保存在相应的`FileField`的`upload_to`参数指定的地方。
```python
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import ModelFormWithFileField

def upload_file(request):
    if request.method == 'POST':
        form = ModelFormWithFileField(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            return HttpResponseRedirect('/success/url/')
    else:
        form = ModelFormWithFileField()
    return render(request, 'upload.html', {'form': form})
```
如果你手动构造一个对象，你可以简单地把文件对象从request.FILE赋值给模型：
```python
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from .models import ModelWithFileField

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            instance = ModelWithFileField(file_field=request.FILES['file'])
            instance.save()
            return HttpResponseRedirect('/success/url/')
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})
```
- 上传多文件

如果要使用一个表单字段上载多个文件，请设置字段小部件的多个HTML属性

```python
# forms.py
from django import forms

class FileFieldForm(forms.Form):
    file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
```

然后覆盖`FormView`子类的post方法来处理多个文件上传

```python
# views.py
from django.views.generic.edit import FormView
from .forms import FileFieldForm

class FileFieldView(FormView):
    form_class = FileFieldForm
    template_name = 'upload.html'  # Replace with your template.
    success_url = '...'  # Replace with your URL or reverse().

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('file_field')
        if form.is_valid():
            for f in files:
                ...  # Do something with each file.
            return self.form_valid(form)
        else:
            return self.form_invalid(form)
```

## 上传处理器

当用户上传一个文件的时候，Django会把文件数据传递给上传处理器 – 一个小型的类，会在文件数据上传时处理它。上传处理器在FILE_UPLOAD_HANDLERS中定义，默认为：
```python
("django.core.files.uploadhandler.MemoryFileUploadHandler",
 "django.core.files.uploadhandler.TemporaryFileUploadHandler",)
```
MemoryFileUploadHandler 和TemporaryFileUploadHandler一起提供了Django的默认文件上传行为，将小文件读取到内存中，大文件放置在磁盘中。

你可以编写自定义的处理器，来定制Django如何处理文件。例如，你可以使用自定义处理器来限制用户级别的配额，在运行中压缩数据，渲染进度条，甚至是向另一个储存位置直接发送数据，而不把它存到本地。关于如何自定义或者完全替换处理器的行为，见自定义上传处理程序

- 上传数据存储位置

在你保存上传文件之前，数据需要储存在某个地方。

通常，如果上传文件小于2.5MB，Django会把整个内容存到内存。这意味着，文件的保存仅仅涉及到从内存读取和写到磁盘，所以非常快。

但是，如果上传的文件很大，Django会把它写入一个临时文件，储存在你系统的临时目录中。在类Unix的平台下，你可以认为Django生成了一个文件，名称类似于/tmp/tmpzfp6I6.upload。如果上传的文件足够大，你可以观察到文件大小的增长，由于Django向磁盘写入数据。

这些特定值 – 2.5 MB，/tmp，以及其它 -- 都仅仅是"合理的默认值"，它们可以自定义

- 更改上传处理器的行为

Django的文件删除处理器的行为由一些设置控制
```Python
DEFAULT_FILE_STORAGE  # 默认的Storage 类，用于没有指定文件系统的任何和文件相关的操作
FILE_CHARSET  # 文件解码字符编码，默认utf-8
FILE_UPLOAD_HANDLERS  # 文件上传控制器
FILE_UPLOAD_MAX_MEMORY_SIZE  # 文件内存保存最大值2621440(2.5M)
FILE_UPLOAD_PERMISSIONS  # 默认None
FILE_UPLOAD_TEMP_DIR  # 临时文件存储位置，默认None
MEDIA_ROOT  # 用户上传文件所在目录的文件系统绝对路径
MEDIA_URL  # 指向MEDIA_ROOT所指定的media文件，通过这个地址管理所存储的文件
```
- 在运行中更改上传处理器

有时候一些特定的视图需要不同的上传处理器。在这种情况下，你可以通过修改`request.upload_handlers`，为每个请求覆盖上传处理器。通常，这个列表会包含`FILE_UPLOAD_HANDLERS`提供的上传处理器，但是你可以把它修改为其它列表。

例如，假设你编写了`ProgressBarUploadHandler`，它会在上传过程中向某类AJAX控件提供反馈。你可以像这样将它添加到你的上传处理器中：
```Python
request.upload_handlers.insert(0, ProgressBarUploadHandler(request))
```
在这中情况下你可能想要使用list.insert()（而不是append()），因为进度条处理器需要在任何其他处理器之前执行。要记住，多个上传处理器是按顺序执行的。

如果你想要完全替换上传处理器，你可以赋值一个新的列表：
```python
request.upload_handlers = [ProgressBarUploadHandler(request)]
```
注意

你只可以在访问request.POST或者request.FILES之前修改上传处理器-- 在上传处理工作执行之后再修改上传处理就毫无意义了。如果你在读取request.FILES之后尝试修改request.upload_handlers，Django会抛出异常。

所以，你应该在你的视图中尽早修改上传处理器。

`CsrfViewMiddleware` 也会访问request.POST，它是默认开启的。这意味着你需要在你的视图中使用`csrf_exempt() `来允许你修改上传处理器。然后你需要在真正处理请求的函数上使用`csrf_protect()`。注意这意味着处理器可能会在`CSRF`验证完成之前开始接收上传文件。例如：

```python
from django.views.decorators.csrf import csrf_exempt, csrf_protect

@csrf_exempt
def upload_file_view(request):
    request.upload_handlers.insert(0, ProgressBarUploadHandler())
    return _upload_file_view(request)

@csrf_protect
def _upload_file_view(request):
    ... # Process request
```



## File对象

django.core.files模块及其子模块包含了一些用于基本文件处理的内建类

