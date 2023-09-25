# pathlib

pathlib是ptyhon内置库，提供表示文件系统路径的类，其语义适用于不同的操作系统。

模块各个类：`PurePath、Path、PurePosixPath、PosixPath、PureWindowsPath、WindowsPath`

## 基本使用方法

### 当前路径/文件名/目录名

```python
from pathlib import Path

# 获取当前路径
print(Path.cwd())

# 从路径中分解文件名
p1 = Path("E:\a\b.log")
p1.stem  # 文件名（不带扩展名）:b
p1.name  # 文件名全名:b.log
p1.suffix  # 文件名后缀:.log
p1.suffixes  # [.log]
p1.anchor  # E:\

# 用parents属性来分解路径目录
p1.parent  # E:\a   # 获取path的完整目录名，不含文件名
p1.parents  # [,]
p1.parents[0]  # E:\a
p1.parents[1]  # E:\

# 用parts来分解路径
p = PurePath('/usr/bin/python3')
p.parts
p = PureWindowsPath("c:\a\b")
p.parts
```

### 当前目录的子目录

```python
p = Path(".")
[x for x in p.iterdir() if x.is_dir()]
```

### 指定路径下文件

```shell
# glob()方法返回列表，元素也是Path类型
# rglob()方法可获取当前目录及所有子目录的相关内容
list(p.glob("**/*.py"))
```

### 路径拼接

```python
p = Path("/etc")
q = p/'init.d'/'reboot'  # 使用/
z = p.joinpath("1.conf")  # 使用joinpath
```

### 查询路径是目录还是文件

```python
q.exists()  # 路径是否存在
q.is_dir()  # 是否为目录
q.is_file()  # 是否为文件
```

### 绝对路径

```python
p = Path()
p.resolve()  # 获取绝对路径
```

### path字符串化

```python
p = PurePath("/etc")
str(p)
p = PureWindowsPath("c:/program Files")
str(p)
```

### 用路径打开文件写文件

```python
with q.open() as f:
		f.readline()
```

### 删除文件

```python
file_path = Path('/path/to/file.txt')
dir_path = Path('/path/to/directory/')

# 删除文件
file_path.unlink()

# 删除空目录
dir_path.rmdir()

# 删除非空目录
import shutil
shutil.rmtree(dir_path)
```

## 与os对照方法

Patth是 PurePath的子类, 这个类代表 concrete path 实体路径
除了继承PurePath的方法外，下面表格列出来 Path的常用方法，以及与os 相关方法的对应关系，

| os and os.path             | pathlib          |
| -------------------------- | ---------------- |
| `os.path.abspath()`        | `Path.resolve()` |
| `os.chmod()`               | `Path.chmod()`   |
| `os.mkdir()`               | `Path.mkdir()`   |
| `os.makedirs()`            | `Path.mkdir()`   |
| `os.rename()`              | `Path.rename()`  |
| `os.replace()`             | `Path.replace()` |
| `os.rmdir()`               | `Path.rmdir()`   |
| `os.remove(), os.unlink()` | `Path.unlink()`  |
| `os.getcwd()`              | `Path.cwd()`     |
| `os.path.exists()`         | `Path.exists()`  |
| `os.path.expanduser()`                         | `Path.expanduser() and Path.home()`               |
| `os.listdir()`                         | `Path.iterdir()`               |
| `os.path.isdir()`                         | `Path.is_dir()`               |
| `os.path.isfile()`                         | `Path.is_file()`               |
| `os.path.islink()`                         | `Path.is_symlink()`               |
| `os.link()`                         | `Path.hardlink_to()`               |
| `os.symlink()`                         | `Path.symlink_to()`               |
| `os.readlink()`                         | `Path.readlink()`               |
| `os.path.relpath()`                         | `Path.relative_to()`               |
| `os.stat()`                         | `Path.stat(), Path.owner(), Path.group()`               |
| `os.path.isabs()`                         | `PurePath.is_absolute()`               |
| `os.path.join()`                         | `PurePath.joinpath()`               |
| `os.path.basename()`                         | `PurePath.name`               |
| `os.path.dirname()`                         | `PurePath.parent`               |
| `os.path.samefile()`                         | `Path.samefile()`               |
| `os.path.splitext()`                         | `PurePath.suffix`             |


​	
​	
​	
​	
