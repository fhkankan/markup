# aiojobs

aiojobs可以构建方便管理的异步任务

## 安装

```
pip install aiojobs
```

## 使用

```python
create_scheduler(close_time=0.1, limit=100, pending_limit=10000, exception_handler=None)

# 参数
- close_time
- pending_limit

- exception_handler  
	# 如果出现异常，会调用exception_handler传入的方法，如果没有传入任何方法，会抛出异常报错
- limit
	# 同时执行的任务数上限
```

实例

```python
import asyncio
import aiojobs

async def func():
    for i in range(100):
        await asyncio.sleep(1)
        print(i)
        if i == 4:
            raise Exception('warning')

def exception_handler(job, errinfo):
    print(job)
    print(errinfo)

async def main():
    jobs = await aiojobs.create_scheduler(
        exception_handler=exception_handler,
    )
    await jobs.spawn(func())
    await asyncio.sleep(6)
    await jobs.close()


if __name__ == '__main__':
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main())
```

