# MQ

“消息队列”是在消息的传输过程中保存消息的容器。消息队列管理器在将消息从它的源中继到它的目标时充当中间人。队列的主要目的是提供路由并保证消息的传递；如果发送消息时接收者不可用，消息队列会保留消息，直到可以成功地传递它。

##  作用

相信对任何架构或应用来说，消息队列都是一个至关重要的组件，下面是十个理由:

```
1. 解耦**
在项目启动之初来预测将来项目会碰到什么需求，是极其困难的。消息队列在处理过程中间插入了一个隐含的、基于数据的接口层，两边的处理过程都要实现这一接口。这允许你独立的扩展或修改两边的处理过程，只要确保它们遵守同样的接口约束。

2. 冗余
有时在处理数据的时候处理过程会失败。除非数据被持久化，否则将永远丢失。消息队列把数据进行持久化直到它们已经被完全处理，通过这一方式规避了数据丢失风险。在被许多消息队列所采用的"插入-获取-删除"范式中，在把一个消息从队列中删除之前，需要你的处理过程明确的指出该消息已经被处理完毕，确保你的数据被安全的保存直到你使用完毕。

3. 扩展性
因为消息队列解耦了你的处理过程，所以增大消息入队和处理的频率是很容易的；只要另外增加处理过程即可。不需要改变代码、不需要调节参数。扩展就像调大电力按钮一样简单。

4. 灵活性 & 峰值处理能力
当你的应用上了Hacker News的首页，你将发现访问流量攀升到一个不同寻常的水平。在访问量剧增的情况下，你的应用仍然需要继续发挥作用，但是这样的突发流量并不常见；如果为 以能处理这类峰值访问为标准来投入资源随时待命无疑是巨大的浪费。使用消息队列能够使关键组件顶住增长的访问压力，而不是因为超出负荷的请求而完全崩溃。 请查看我们关于峰值处理能力的博客文章了解更多此方面的信息。

5. 可恢复性
当体系的一部分组件失效，不会影响到整个系统。消息队列降低了进程间的耦合度，所以即使一个处理消息的进程挂掉，加入队列中的消息仍然可以在系统恢复后被处理。而这种允许重试或者延后处理请求的能力通常是造就一个略感不便的用户和一个沮丧透顶的用户之间的区别。

6. 送达保证
消息队列提供的冗余机制保证了消息能被实际的处理，只要一个进程读取了该队列即可。在此基础上，IronMQ提供了一个"只送达一次"保证。无论有多少进程在从队列中领取数据，每一个消息只能被处理一次。这之所以成为可能，是因为获取一个消息只是"预定"了这个消息，暂时把它移出了队列。除非客户端明确的 表示已经处理完了这个消息，否则这个消息会被放回队列中去，在一段可配置的时间之后可再次被处理。

7.排序保证
在许多情况下，数据处理的顺序都很重要。消息队列本来就是排序的，并且能保证数据会按照特定的顺序来处理。IronMO保证消息浆糊通过FIFO（先进先出）的顺序来处理，因此消息在队列中的位置就是从队列中检索他们的位置。

8.缓冲
在任何重要的系统中，都会有需要不同的处理时间的元素。例如,加载一张图片比应用过滤器花费更少的时间。消息队列通过一个缓冲层来帮助任务最高效率的执行--写入队列的处理会尽可能的快速，而不受从队列读的预备处理的约束。该缓冲有助于控制和优化数据流经过系统的速度。

9. 理解数据流
在一个分布式系统里，要得到一个关于用户操作会用多长时间及其原因的总体印象，是个巨大的挑战。消息系列通过消息被处理的频率，来方便的辅助确定那些表现不佳的处理过程或领域，这些地方的数据流都不够优化。

10. 异步通信
很多时候，你不想也不需要立即处理消息。消息队列提供了异步处理机制，允许你把一个消息放入队列，但并不立即处理它。你想向队列中放入多少消息就放多少，然后在你乐意的时候再去处理它们。
```

## 实现

### 线程队列

```
#!/usr/bin/env python
 
import Queue
import threading
import time
 
queue = Queue.Queue()
 
class ThreadNum(threading.Thread):
 """没打印一个数字等待1秒，并发打印10个数字需要多少秒？"""
 def __init__(self, queue):
  threading.Thread.__init__(self)
  self.queue = queue
 
 def run(self):
  whileTrue:
   #消费者端，从队列中获取num
   num = self.queue.get()
   print"i'm num %s"%(num)
   time.sleep(1)
   #在完成这项工作之后，使用 queue.task_done() 函数向任务已经完成的队列发送一个信号
   self.queue.task_done()
 
start = time.time()
def main():
 #产生一个 threads pool, 并把消息传递给thread函数进行处理，这里开启10个并发
 for i in range(10):
  t = ThreadNum(queue)
  t.setDaemon(True)
  t.start()
  
 #往队列中填错数据 
 for num in range(10):
   queue.put(num)
 #wait on the queue until everything has been processed
 queue.join()
 
main()
print"Elapsed Time: %s" % (time.time() - start)
```

运行结果：

```
i'm num 0
i'm num 1
i'm num 2
i'm num 3
i'm num 4
i'm num 5
i'm num 6
i'm num 7
i'm num 8
i'm num 9
Elapsed Time: 1.01399993896
```

解读：
具体工作步骤描述如下：
1，创建一个 Queue.Queue() 的实例，然后使用数据对它进行填充。
2，将经过填充数据的实例传递给线程类，后者是通过继承 threading.Thread 的方式创建的。
3，生成守护线程池。
4，每次从队列中取出一个项目，并使用该线程中的数据和 run 方法以执行相应的工作。
5，在完成这项工作之后，使用 queue.task_done() 函数向任务已经完成的队列发送一个信号。
6，对队列执行 join 操作，实际上意味着等到队列为空，再退出主程序。
在使用这个模式时需要注意一点：通过将守护线程设置为 true，程序运行完自动退出。好处是在退出之前，可以对队列执行 join 操作、或者等到队列为空。

### 多个队列

所谓多个队列，一个队列的输出可以作为另一个队列的输入！

```
#!/usr/bin/env python
import Queue
import threading
import time
 
queue = Queue.Queue()
out_queue = Queue.Queue()
 
class ThreadNum(threading.Thread):
  """bkeep"""
  def __init__(self, queue, out_queue):
    threading.Thread.__init__(self)
    self.queue = queue
    self.out_queue = out_queue
 
  def run(self):
    whileTrue:
      #从队列中取消息
      num = self.queue.get()
      bkeep = num
      
      #将bkeep放入队列中
      self.out_queue.put(bkeep)
 
      #signals to queue job is done
      self.queue.task_done()
 
class PrintLove(threading.Thread):
  """Threaded Url Grab"""
  def __init__(self, out_queue):
    threading.Thread.__init__(self)
    self.out_queue = out_queue
 
  def run(self):
    whileTrue:
      #从队列中获取消息并赋值给bkeep
      bkeep = self.out_queue.get()  
      keke = "I love " + str(bkeep)
      print keke,
      print self.getName()
      time.sleep(1)
 
      #signals to queue job is done
      self.out_queue.task_done()
 
start = time.time()
def main():
  #populate queue with data
  for num in range(10):
    queue.put(num)
    
  #spawn a pool of threads, and pass them queue instance
  for i in range(5):
    t = ThreadNum(queue, out_queue)
    t.setDaemon(True)
    t.start()
 
 
  for i in range(5):
    pl = PrintLove(out_queue)
    pl.setDaemon(True)
    pl.start()
 
  #wait on the queue until everything has been processed
  queue.join()
  out_queue.join()
 
main()
print"Elapsed Time: %s" % (time.time() - start)
```

运行结果：

```
I love 0 Thread-6
I love 1 Thread-7
I love 2 Thread-8
I love 3 Thread-9
I love 4 Thread-10
I love 5 Thread-7
I love 6 Thread-6
I love 7 Thread-9
I love 8 Thread-8
I love 9 Thread-10
Elapsed Time: 2.00300002098
```

解读：
ThreadNum 类工作流程
定义队列--->继承threading---->初始化queue---->定义run函数--->get queue中的数据---->处理数据---->put数据到另外一个queue-->发信号告诉queue该条处理完毕
main函数工作流程：
--->往自定义queue中扔数据
--->for循环确定启动的线程数---->实例化ThreadNum类---->启动线程并设置守护
--->for循环确定启动的线程数---->实例化PrintLove类--->启动线程并设置为守护
--->等待queue中的消息处理完毕后执行join。即退出主程序。

### 异步回调

```python
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import Queue
import threading
import time

def func_a(a, b):
    return a + b

def func_b():
    pass

def func_c(a, b, c):
    return a, b, c

# 异步任务队列
_task_queue = Queue.Queue()

def async_call(function, callback, *args, **kwargs):
    _task_queue.put({
        'function': function,
        'callback': callback,
        'args': args,
        'kwargs': kwargs
    })

def _task_queue_consumer():
    """
    异步任务队列消费者
    """
    while True:
        try:
            task = _task_queue.get()
            function = task.get('function')
            callback = task.get('callback')
            args = task.get('args')
            kwargs = task.get('kwargs')
            try:
                if callback:
                    callback(function(*args, **kwargs))
            except Exception as ex:
                if callback:
                    callback(ex)
            finally:
                _task_queue.task_done()
        except Exception as ex:
            logging.warning(ex)

def handle_result(result):
    time.sleep(5)
    print(type(result), result)

if __name__ == '__main__':
    t = threading.Thread(target=_task_queue_consumer)
    t.daemon = True
    t.start()

    async_call(func_a, handle_result, 1, 2)
    async_call(func_b, handle_result)
    async_call(func_c, handle_result, 1, 2, 3)
    async_call(func_c, handle_result, 1, 2, 3, 4)

    _task_queue.join()
```

