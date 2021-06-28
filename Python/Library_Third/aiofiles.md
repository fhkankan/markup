# aiofiles

[官网](https://github.com/mosquito/aiofile)

## 安装

```
pip install aiofile
```

## High-level

### async_open

```python
import asyncio
from aiofile import async_open


async def main():
    async with async_open("/tmp/hello.txt", 'w+') as afp:
        await afp.write("Hello ")
        await afp.write("world")
        afp.seek(0)

        print(await afp.read())

        await afp.write("Hello from\nasync world")
        print(await afp.readline())
        print(await afp.readline())

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

支持的方法

```python
async def read(length = -1)  # 从文件中读取块，当长度为 -1 时将读取文件到最后
async def write(data)  # 向文件写入块
def seek(offset)  # 设置文件指针位置
def tell()  # 返回当前文件指针位置
async def readline(size=-1, newline="\n")  # 读取块直到换行或 EOF。由于不重用读取缓冲区，因此不适合小行。当您想按行读取文件时，请避免使用 sync_open 而使用 LineReader替代。
```

### Reader/Writer

当您想线性读取或写入文件

```python
import asyncio
from aiofile import AIOFile, Reader, Writer


async def main():
    async with AIOFile("/tmp/hello.txt", 'w+') as afp:
        writer = Writer(afp)
        reader = Reader(afp, chunk_size=8)

        await writer("Hello")
        await writer(" ")
        await writer("World")
        await afp.fsync()

        async for chunk in reader:
            print(chunk)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

### LineReader

LineReader 是一个非常有效的助手，当你想线性地逐行读取文件时，它非常有效。它包含一个缓冲区，将逐块读取文件的片段到缓冲区中，在那里它会尝试查找行。默认块大小为 4KB。

当您想按行读取文件时，请避免使用 async_open 而使用 LineReader替代。

```python
import asyncio
from aiofile import AIOFile, LineReader, Writer


async def main():
    async with AIOFile("/tmp/hello.txt", 'w+') as afp:
        writer = Writer(afp)

        await writer("Hello")
        await writer(" ")
        await writer("World")
        await writer("\n")
        await writer("\n")
        await writer("From async world")
        await afp.fsync()

        async for line in LineReader(afp):
            print(line)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

## Low-level

Write and Read

```python
import asyncio
from aiofile import AIOFile


async def main():
    async with AIOFile("/tmp/hello.txt", 'w+') as afp:
        await afp.write("Hello ")
        await afp.write("world", offset=7)
        await afp.fsync()

        print(await afp.read())


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

Read file line by line

```python
import asyncio
from aiofile import AIOFile, LineReader, Writer


async def main():
    async with AIOFile("/tmp/hello.txt", 'w') as afp:
        writer = Writer(afp)

        for i in range(10):
            await writer("%d Hello World\n" % i)

        await writer("Tail-less string")


    async with AIOFile("/tmp/hello.txt", 'r') as afp:
        async for line in LineReader(afp):
            print(line[:-1])


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```
