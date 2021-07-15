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
import time

# 一次执行任务
async def func():
    for i in range(100):
        await asyncio.sleep(1)
        print(i)
        if i == 4:
            raise Exception('warning')

def exception_handler(job, errinfo):
    print(job)
    print(errinfo)
    
    
# 固定间隔的任务
async def cron_job_by_interval(jobs, method, interval=60):
    p_stamp = 0
    while not jobs.closed:
        now = int(time.time())
        if now - p_stamp >= interval:
            try:
                await method()
            except Exception as ex:
                logger.exception(ex)
            p_stamp = now
        await asyncio.sleep(0.3)
        
# 定时任务
async def cron_job_by_time(jobs, method, delta_time=0):
    """
    每日定时脚本
    :param method: 需要执行的方法
    :param delta_time: 距离0点的时间单位s
    """
    while not jobs.closed:
        now = int(time.time())
        day_start = time.mktime(datetime.now().date().timetuple())
        if delta_time < now - day_start <= delta_time + 60:
            try:
                await method()
            except Exception as ex:
                logger.exception(ex)
            left_time = 86400 - (now - day_start) + delta_time + 30  # 30s的阈值
            await asyncio.sleep(left_time)
        await asyncio.sleep(1)

        
async def main():
    jobs = await aiojobs.create_scheduler(
        exception_handler=exception_handler,
    )
    # 一次执行的任务
    await jobs.spawn(func())
    await asyncio.sleep(6)
    await jobs.close()
    # 固定/定时执行的任务
    await jobs.spawn(cron_job_by_interval(jobs, func, 60 * 10))
    await jobs.spawn(cron_job_by_time(jobs, func))
    

if __name__ == '__main__':
	loop = asyncio.get_event_loop()
	loop.run_until_complete(main())
```

