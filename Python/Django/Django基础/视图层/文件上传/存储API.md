## 文件存储API

### 获取当前的存储类

Django提供了两个便捷的方法来获取当前的储存类

`class DefaultStorage`

```
DefaultStorage提供对当前的默认储存系统的延迟访问，类似于DEFAULT_FILE_STORAGE中定义的那样。DefaultStorage 内部使用了get_storage_class()。
```

`get_storage_class([import_path=None])`

```
# 返回一个类或者模块，这个类或者模块实现了存储API。
# 参数
当没有带着import_path参数调用的时候， get_storage_class 会返回当前默认的储存系统，类似于DEFAULT_FILE_STORAGE中定义的那样。
如果提供了import_path， get_storage_class会尝试从提供的路径导入类或者模块，并且如果成功的话返回它。如果导入不成功会抛出异常。
```

### FileSystemStorage

```python
class FileSystemStorage(location=None, base_url=None, file_permissions_mode=None, directory_permissions_mode=None)
# FileSystemStorage类在本地文件系统上实现了基本的文件存储功能。它继承自Storage ，并且提供父类的所有公共方法的实现。
# 参数
location  # 储存文件的目录的绝对路径。默认为MEDIA_ROOT设置的值。
base_url  # 在当前位置提供文件储存的URL。默认为MEDIA_URL设置的值。
file_permissions_mode  # 文件系统的许可，当文件保存时会接收到它。默认为FILE_UPLOAD_PERMISSIONS。
directory_permissions_mode  # 文件系统的许可，当目录保存时会接收到它。默认为FILE_UPLOAD_DIRECTORY_PERMISSIONS。
```

方法

```python
delete()
# 在提供的文件名称不存在的时候并不会抛出任何异常
get_created_time(name)
# 返回系统的ctime的日期时间，即os.path.getctime()。在某些系统（如Unix）上，这是最后一次元数据更改的时间，而在其他系统（如Windows）上，则是文件的创建时间。
```

### Storage

```python
class Storage
# Storage类为文件的存储提供了标准化的API，并带有一系列默认行为，所有其它的文件存储系统可以按需继承或者复写它们。
# 注意
对于返回原生datetime对象的方法，所使用的有效时区为os.environ['TZ']的当前值。要注意它总是可以通过Django的TIME_ZONE来设置。
```

方法

| name                                        | desc                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `delete(name)`                              | 删除`name`引用的文件。如果目标储存系统不支持删除操作，会抛出`NotImplementedError`异常。 |
| `exists(name)`                              | 如果提供的名称所引用的文件在文件系统中存在，则返回`True`，否则如果这个名称可用于新文件，返回`False`。 |
| `get_accessed_time(name)`                   | 返回包含文件的最后访问时间的原生`datetime`对象。对于不能够返回最后访问时间的储存系统，会抛出`NotImplementedError`异常。如果`USE_TZ`为True，则返回一个知晓的日期时间，否则在本地时区返回一个天真的日期时间。 |
| `get_available_name(name, max_length=None)` | 返回基于name参数的文件名称，它在目标储存系统中可用于写入新的内容。如果提供了max_length，文件名称长度不会超过它。 如果不能找到可用的、唯一的文件名称，会抛出SuspiciousFileOperation 异常。如果name命名的文件已存在，一个下划线加上随机7个数字或字母的字符串会添加到文件名称的末尾，扩展名之前。 |
| `get_created_time(name)`                    | 返回文件创建时间的datetime。 对于无法返回创建时间的存储系统，这将引发NotImplementedError。如果USE_TZ是True，返回一个意识的datetime，否则返回本地时区的datetime。 |
| `get_modified_time(name)`                   | 返回文件的上次修改时间的datetime。 对于无法返回上次修改时间的存储系统，这将引发NotImplementedError。如果USE_TZ是True，返回一个意识的datetime，否则返回本地时区的datetime。 |
| `get_valid_name(name)`                      | 返回基于name参数的文件名称，它适用于目标储存系统。           |
| `generate_filename(filename)`               | 通过调用get_valid_name()来验证filename，并返回要传递给save()方法的文件名。filename参数可能包含由FileField.upload_to返回的路径。 在这种情况下，该路径将不会传递给get_valid_name()，但将返回到结果名称。默认实现使用os.path操作。 如果这不适合您的存储空间，请覆盖此方法。 |
| `listdir(path)`                             | 列出指定路径的内容，返回一个2元列表；第一个项目是目录，第二个项目是文件。 对于不能够提供列表功能的储存系统，抛出NotImplementedError异常。 |
| `open(name, mode='rb')`                     | 通过提供的name.打开文件。 注意虽然返回的文件确保为File对象，但可能实际上是它的子类。 在远程文件储存的情况下，这意味着读写操作会非常慢，所以警告一下。 |
| `path(name)`                                | 本地文件系统的路径，文件可以用Python标准的open()在里面打开。 对于不能从本地文件系统访问的储存系统，抛出NotImplementedError异常。 |
| `save(name, content, max_length=None)`      | 使用储存系统来保存一个新文件，最好带有特定的名称。 如果名称为 name的文件已存在，储存系统会按需修改文件名称来获取一个唯一的名称。 返回被储存文件的实际名称。max_length参数会传递给get_available_name()。content参数必须是django.core.files.File或可以包裹在File中的类似文件的对象的实例。 |
| `size(name)`                                | 返回name所引用的文件的总大小，以字节为单位。 对于不能够返回文件大小的储存系统，抛出NotImplementedError异常。 |
| `url(name)`                                 | 返回URL，通过它可以访问到name所引用的文件。 对于不支持通过URL访问的储存系统，抛出NotImplementedError异常。 |

