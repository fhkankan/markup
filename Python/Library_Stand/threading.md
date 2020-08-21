# threading


线程的状态
```
# Init,初始化:创建线程，操作新系统在内部会将其标识为初始化状态，只在系统内核中使用
# Ready,就绪：准备好被执行
# Deferred ready,延迟就绪：表示线程已经被选择在指定的处理器上运行，但还没有被调度
# Standby,备用：表示线程已经被选择下一个在指定的处理器上运行，当该处理器上运行的线程因等待资源等原因被挂起时，调度器将备用线程切换到处理器上运行，只有一个线程可以是备用状态
# Running,运行：表示调度器将线程切换到处理器上运行，可以运行一个线程周期，然后将处理器让给其他线程
# Waiting,等待：线程因为等待一个同步执行的对象或等待资源等原因切换到等待状态
# transition,过渡：表示线程以及该准备后被执行，但它的内核堆已经从内存中移除。一旦其内核堆被加载到内存中，线程就会变成运行状态
# Terminated,终止：当线程被执行完成后，其状态会变成终止，系统会释放线程中的数据结构和资源
```

## 创建线程
```
# 线程对象 = threading.Thread(target = 线程函数名, args=参数元组，kwargs=参数字典，[name=线程名，group=线程组])
```
## 模块方法
```
# threading.Thread()            --->实例化一个线程对象
# threading.current_thread()    --->返回当前的线程对象
# threading.currentThread()     --->返回当前的线程变量
# threading.activeCount()       --->返回当前进程里面线程的个数
# threading.enumerate()         --->返回当前运行中的Thread对象列表
```
## 类方法
```
# threading对象.setDaemon()     --->参数设置为True,会将线程声明为守护线程，且必须在start()方法之前设置，不设置为守护线程，程序将会被无限挂起
# threading对象.run()           --->用于表示线程活动的方法
# threading对象.start()         --->启动运行程序
# threading对象.join([timeout]) --->可以阻塞进程直到线程执行完毕，timeout设定超时时间
# threading对象.isAlive()       --->返回线程是否是活动的
# threading对象.getName()       --->返回线程名
# threading对象.setName()       --->设置线程名
```
## 多线程与多进程处理
```
# 互斥锁       --->避免多个线程同时修改同一块数据
# lock = threading.Lock()       --->创建互斥锁
# lock.acquire()                --->加锁
# lock.release()                --->解锁
 
 
# 递归锁       --->避免锁中有锁
# lock = threading.RLock()      --->创建递归锁
# lock.acquire()                --->加锁
# lock.release()                --->解锁


# 信号量       --->用于控制线程数量
# semaphore = threading.BoundedSemaphore([value])       --->创建信号量
# semaphore.acquire()           --->加锁
# semaphore.release()           --->解锁


# with Lock/RLock/BoundedSemaphore对象
# 可做with语句上下文管理器，进入该代码块时将调用acquire()方法，退出时调用release()方法


# 事件对象      --->线程间通信机制，每个事件对应一个内部标志
# event = threading.Event()     --->创建时间对象
# event.set()                   --->设置标记为True
# event.clear()                 --->设置标记为False
# event.wait()                  --->在事件调用wait()方法时，线程将阻塞直到内部标志被设置为True
# event.isSet()                 --->判断标志是否被设置


# threading.local() --->为多个线程提供不共享空间


# Timer
# 一个线程，用于在一个指定的时间间隔之后执行一个函数
# timer = threading.Timer(时间,函数名) 
# timer.start()
```





