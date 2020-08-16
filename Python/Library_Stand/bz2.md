# bz2

[参考](https://www.cnblogs.com/xiaozx/p/10709720.html)

[参考](http://www.cocoachina.com/cms/wap.php?action=article&id=79141)

bz2模块提供了使用bzip2算法压缩和解压缩数据一套完整的接口。

bz2模块包括：

- 用于读写压缩文件的`open()`函数和`BZ2File`类

- 用于一次性压缩和解压缩的`compress()` 和 `decompress()` 函数

- 用于增量压缩和解压的 `BZ2Compressor` 和 `BZ2Decompressor` 类

## 文件压缩和解压

```
bz2.open(filename, mode='r', compresslevel=9, encoding=None, errors=None, newline=None)
```

以二进制或文本模式打开 bzip2 压缩文件，返回一个文件对象。

```python
import bz2

file = bz2.open('xy.bz2', 'r')
for line in file:
    print(line)
```
```
class bz2.BZ2File(filename, mode='r', buffering=None, compresslevel=9)
```
用二进制模式打开 bzip2 压缩文件

### 一次性
```python
bz2.compress(data)  # 压缩文件
bz2.decompress(data)  # 解压缩文件
```
示例
```python
import bz2

def main():
    username = bz2.decompress(un)
    username = username.decode()
    print(username)
    username1 = bz2.compress(username.encode())
    print(username1)

if __name__ == '__main__':
    un = b'BZh91AY&SYA\xaf\x82\r\x00\x00\x01\x01\x80\x02\xc0\x02\x00 \x00!\x9ah3M\x07<]\xc9\x14\xe1BA\x06\xbe\x084'
    main()
```

示例

```python
with ZipFile('something.zip', 'w') as zf:
    content = bz2.compress(bytes(csv_string, 'UTF-8'))  # also with lzma
    zf.writestr(
        'something.csv' + '.bz2',
        content,
        compress_type=ZIP_DEFLATED
    )
```

### 增量式

单次触发和增量之间的区别在于,对于单次触发模式,您需要将整个数据存储在内存中;如果你正在压缩一个100千兆字节的文件,你应该有大量的内存.

使用增量编码器,您的代码可以一次为压缩器提供1兆字节或1千字节,并在可用时立即将任何数据结果写入文件.另一个好处是可以使用增量压缩器来传输数据 – 您可以在所有未压缩数据可用之前开始编写压缩数据！

```
classbz2.BZ2Compressor(compresslevel=9)
```

`compress(data)` 向压缩对象提供数据，提供完压缩数据后，使用`fiush()`方法以完成压缩方。

`flush()`结束压缩进程，返回内部缓冲中剩余的压缩完成的数据。

```
class bz2.BZ2Decompressor
```
创建一个新的解压缩器对象。该对象可用于递增地解压缩数据。

```
decompress(data, max_length=-1)
```
解压缩数据，将未压缩的数据作为字节返回

示例

```python
>>> c = bz2.BZ2Compressor()
>>> c.compress(b'a' * 1000)
b''
>>> c.flush()
b'BZh91AY&SYI\xdcOc\x00\x00\x01\x81\x01\xa0\x00\x00\x80\x00\x08 \x00 
\xaamA\x98\xba\x83\xc5\xdc\x91N\x14$\x12w\x13\xd8\xc0'
```

示例

```python
with ZipFile('something.zip', 'w') as zf:
    compressor = bz2.BZ2Compressor()
    content = compressor.compress(bytes(csv_string, 'UTF-8'))  # also with lzma
    zf.writestr(
        'something.csv' + '.bz2',
        content,
        compress_type=ZIP_DEFLATED
    )
    content += compressor.flush()
```

