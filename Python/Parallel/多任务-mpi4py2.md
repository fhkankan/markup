#mpi4py

## 概览

示例

```python
# hello.py
from mpi4py import MPI

comm = MPI.COMM_WORLD  # 通信器，定义了自己的一套可互相通信的进程
rank = comm.Get_rank() # 返回调用它的那个进程的等级
print("hello world from process ", rank)
```

执行

```
mpiexec -n 5 python hello.py
```

说明

```
在MPI中，并行程序执行过程中所涉及的进程可以由一个非负的整数序列进行标识，这些证书叫做等级(rank)。若一个程序有p个进程在运行，那么进程的等级就会从0到p-1.

标准输出上的输出并非总是有序的，因为多个进程会同时向屏幕写内容，操作系统则会随意选择顺序。因此，得到一个基本的结论：MPI执行过程中所涉及的每个进程都会运行相同的编译好的二进制代码，这样每个进程都会收到同样的待执行指令
```

## 点对点通信

点对点通信指的是可以在两个进程间传递数据：一个进程接收者，一个进程发送者

python模块的mpi4py通过以下函数实现

```python
comm.send(data, process_destination)
# 将数据发送给目标进程，目标进程是由其在通信器组中的等级来标识的
comm.recv(process_source)
# 从源进程接收数据，源进程也是由其在通信器组中的等级来标识的
comm = MPI.COMM_WORLD
# 通信器，定义了进程组，可以通过消息传递来通信
```

注意

```
comm.send()与comm.recv()函数都是阻塞函数，它们会阻塞调用者，直到缓冲数据被安全地址使用为止。在MPI中，有两种发送与接收消息的方法：缓冲模式、同步模式。
在缓冲模式中，当待发送的数据被复制到缓冲区后，流程控制就会返回到程序中。这并不意味着消息已经被发送或是接收了。
在同步模式中，只有在响应的接收函数开始接收消息时，函数才会终止。
```

- 在不同进程间交换信息

```python
from mpi4py import MPI

# 通信组comm
comm = MPI.COMM_WORLD
# 标识出组中的任务与进程
rank = comm.rank
print('my rank is: ', rank)

# rank值为0的进程会向rank值为4的接收者进程发送一个数字数据
if rank==0:
    data = 10000000
    destination_process = 4
    comm.send(data, dest=destination_process)
    print("sending data %s" % data + "to process %d" % destination_process)
    
if rank==1:
    data = "hello"
    destination_process = 8
    comm.send(data, dest=destination_process)
    print("sending data %s" % data + "to process %d" % destination_process)

# rank值为4的接收者进程
if rank==4:
    # recv必须包含一个从那还素，该参数指定了发送者进程的rank值
    data = comm.recv(source=0)
    print("data received is = %s" % data)

if rank==8:
    data1 = comm.recv(source=1)
    pritn("data received is = %s" data1)
```

## 避免死锁

死锁是两个或多个进程彼此阻塞，一个进程等待另一个进程执行某个动作来满足自己的需要，反之亦然。mpi4py模块并未提供任何具体的功能来解决这一问题，它只是提供了一些举措，开发者需要遵循这些举措来避免死锁问题

示例

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
print('my rank is: ', rank)

if rank == 1:
    data_send = "a"
    destination_process = 5
    source_process = 5
    data_received = comm.recv(source=source_process)
    comm.send(data_send, dest=destination_process)
    print("sending data %s" % data_send + "to process %d" % destination_process)
    print("data received is = %s" % data_received)

if rank == 5:
    data_send = "b"
    destination_process = 1
    source_process = 1
    data_received = comm.recv(source=source_process)
    comm.send(data_send, dest=destination_process)
    print("sending data %s" % data_send + "to process %d" % destination_process)
    print("data received is = %s" % data_received)
```

运行

```shell
mpiexec -n 9 python deadLockProblems.py
```

两个进程都想从过年对方那里接收消息，但都卡住了。首先想到的解决方案

```python
if rank == 1:
    data_send = "a"
    destination_process = 5
    source_process = 5
    comm.send(data_send, dest=destination_process)
    data_received = comm.recv(source=source_process)

if rank == 5:
    data_send = "b"
    destination_process = 1
    source_process = 1
    data_received = comm.recv(source=source_process)
    comm.send(data_send, dest=destination_process)
```

不过，这个解决方案虽然从逻辑上ok，但无法总能避免死锁问题。由于通信是经由缓冲区进行的，而缓冲区是comm.send()MPI复制待发送数据的地方，因此程序能够平滑运行的前提是该缓冲区可以承载所有数据，否则就会导致死锁；发送者无法发送完整数据，因为缓冲区已满，而接收者无法接收数据，因为它被comm.send()MPI阻塞了，无法完成。这时，避免死锁的一种解决方案就是交换发送与接收函数的位置，使它们变成非对称的

```python
if rank == 1:
    data_send = "a"
    destination_process = 5
    source_process = 5
    comm.send(data_send, dest=destination_process)
    data_received = comm.recv(source=source_process)

if rank == 5:
    data_send = "b"
    destination_process = 1
    source_process = 1
    comm.send(data_send, dest=destination_process)
    data_received = comm.recv(source=source_process)
```

上面并非唯一解决方案。比如，有这样一个特殊函数，它统一了向给定进程发送消息的调用与接收来自另外一个线程的消息的调用，该函数叫做Sendrecv

```python
Sendrecv(self, sendbuf, int dest=0, int sendtag=0, recvbuf=None, int source=0, int recvtag=0, Status status=None)
```

如此，所需参数与comm.send()MPI和comm.recv()MPI相同。此外，在该示例中，函数会阻塞，不过相比于之前看到的两个函数来说，它的优势在于让通信子系统负责检查发送与接收的依赖关系，从而避免了死锁。

```python
if rank == 1:
    data_send = "a"
    destination_process = 5
    source_process = 5
    comm.send(data_send, dest=destination_process)
    data_received = comm.recv(source=source_process)

if rank == 5:
    data_send = "b"
    destination_process = 1
    source_process = 1
    comm.send(data_send, dest=destination_process)
    data_received = comm.recv(source=source_process)
```

## 聚合通信

借助于聚合通信，可以实现一个组内的多个进程间同时进行数据传递。mpi4py只提供了聚合通信的阻塞版本(会阻塞调用者方法，直至缓冲区数据可以安全使用为止)

常见的聚合操作有

```
- 跨越组内进程的屏障同步
- 通信功能
	- 从一个进程向组内其他所有进程广播数据
	- 将所有进程的数据收集到一个进程
	- 从一个进程将数据分发到其他进程
- 汇聚操作
```

### 广播

在并行代码的时候，常遇到如下情况：运行期需要在多个进程间共享某个变量的值，或是共享由每个进程所提供的对变量的操作(操作不同的值)

为了解决这个问题，我们使用通信树(如，进程0向进程1与2发送数据，进程1与2则分别将数据发送给进程3,4,5等)

涉及属于某个通信器的所有进程的通信方法叫做聚合通信。因此，聚合通信一般来说会涉及两个以上的进程。不过，我们会将聚合通信叫做广播，其中一个进程向其他进程发送相同的数据。广播中的mpi4py功能由如下函数提供，该函数只是将消息进程根中的信息发送给属于comm通信器的其他进程；不过，每个进程都必须要通过相同的root与comm值来调用它。

```python
buf = comm.bcast(data_to_share, rank_of_root_process)

# 参数
data_to_share			待共享的数据
rank_of_root_process	根进程或主发送进程
```

示例

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    variable_to_share = 100
else:
    variable_to_share = None

variable_to_share = comm.bcast(variable_to_share, root=0)
print("process = %d" % rank + "variable shared = %d" % variable_to_share)
```

运行

```
mpiexec -n 10 python broadcast.py  # 开启10个进程
```

### scatter

scatter的功能类似于scatter广播，但是虽然comm.bcast会向所有舰艇进程发送同样的数据，但comm.scatter却能以数组形式将数据块发送给不同进程。

`comm.scatter`函数会接收数组元素，并根据进程的等级值将它们发送给相应的进程。地1个元素会被发送给进程0，第2个元素会被发送给进程1，以此类推。

mpi4py函数实现

```python
recvbuf = comm.scatter(sendbuf, rank_of_root_process)
```

还提供了另外两个函数用于散播数据

```python
comm.scatter(sendbuf, recvbuf, root=0)
# 会将相同通信器中的一个进程的数据发送给所有其他进程
comm.scatterv(sendbuf, recvbuf, root=0)
# 会将一个组中一个进程的数据发送给所有其他进程。在发送端，可以发送不同数量的数据，偏移量也可以不同

sendbuf与recvbuf参数必须以列表的形式给出:buf = [data, data_size, data_type]
data必须是一个类似于缓冲的对象，其大小为data_size，类型为data_type
```

示例

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    array_to_share = [1,2,3,4,5,6,7,8,9,10]
else:
    variable_to_share = None

# 通过comm.scatter将要发送给第i个进程的第i个变量的值
recvbuf = comm.scatter(array_to_share, root=0)
print("process = %d" % rank + "recvbuf= %d" % array_to_share)
```

运行

```
mpiexec -n 10 python scatter.py
```

### gather

gather函数会执行与scatter相反的功能。在该示例中，所有进程会向根进程发送数据，根进程则会收集所接收到的数据。

mpi4py实现

```
recvbuf = comm.gather(sendbuf, rank_of_root_process)

# 参数
sendbuf					发送的数据
rank_of_root_process	所有数据的进程接收者
```

收集数据函数

```python
# 收集到一个任务
comm.Gather/comm.Gatherv/comm.gather

# 收集到所有任务
comm.Allgather/comm.Allgatherv/comm.allgather
```

示例

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
data = (rank + 1) ** 2  # 有n个进程来发送数据
data = comm.gather(data, root=0) # 实现收集数据

if rank == 0:  # 如果进程的rank值为0，那么数据就会被收集到数组中
    print("rank = %s " % rank + "...receiving data to other process")
    for i in range(1, size):
        data[i] = (i + 1) ** 2
        value = data[i]
        print("process = %s receiving %s from process %s" % (rank, value, i))
```

运行

```
mpiexec -n 5 python gather.py
```

### Alltotall

Alltoall聚合通信整合了scatter与gather的功能。mpi4py中有3类Alltoall聚合通信

```python
comm.Alltoall(sendbuf, recvbuf)
# all-to-all scatter/gather从组中的all-to-all进程发送数据
comm.Alltoallv(sendbuf, recvbuf)
# all-to-all scatter/gather从组中的all-to-all进程发送数据，提供了不同的数据量与偏移量
comm.Alltoallw(sendbuf, recvbuf)
# 广义的all-to-all通信，支持每个进程使用不同数量、不同偏移量与不同数据类型的数据
```

All-to-all个性化通信又叫做全交换。该操作可用在各种并行算法中，如傅里叶变换、矩阵转置、抽样排序以及一些并行的数据库操作连接操作等。

示例

```python
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

a_size = 1
senddata = (rank + 1) * numpy.arrange(size, dtype=int)
recvdata = numpy.empty(size*a_size, dtype=int)
comm.Alltoall(senddata, recvdata)  # 从任务j的sendbuf中接收到第i个对象，然后将其肤质懂啊任务i的recvbuf的第j个对象处

print("process %s sending %s receiving %s" % (rank, senddata, recvdata))
```

运行

```
mpiexec -n 5 python alltoall.py
```

注意

```
[
	0 1 2 3 4
	0 2 4 6 8
	0 3 6 9 12
	0 4 8 12 16
	0 5 10 15 20
]
经过Alltoall
[
    0 0 0 0 0 
    1 2 3 4 5
    2 4 6 8 10
    3 6 9 12 15
    4 8 12 16 20
]
```

### 汇聚操作

类似于`comm.gather`,`comm.reduce`也会在每个进程中接收一个输入元素的数组，并将一个输出元素的数组返回给根进程。输出元素包含了汇聚后的结果

mpi4py中实现

```python
comm.Reduce(sendbuf, recvbuf, rank_of_root_process, op=type_of_reduction_operation)
```

`comm.gather`语句位于op参数中了，它指的是希望对数据所应用的操作，mpi4py模块包含了一组可以使用的汇聚操作。由MPI所定义的一些汇聚操作有如下几项

```
MPI.MAX		返回最大的元素
MPI.MIN		返回最小的元素
MPI.SUM		对元素求和
MPI.PROD	对所有元素相乘
MPI.LAND	对元素执行一个逻辑运算
MPI.MAXLOC	返回最大值以及拥有该最大值的进程的rank值
MPI.MINLOC	返回最小值以及拥有该最小值的进程的rank值
```

示例

```python
from mpi4py import MPI
import numpy


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

array_size = 3
recvdata = numpy.zeros(array_size, dtype=numpy.int)
senddata = (rank + 1) * numpy.arrange(array_size, dtype=numpy.int)
print("process %s sending %s " % (rank, senddata))
comm.Reduce(senddata, recvdata, root=0, op=MPI.SUM)  # 执行汇聚求和
print('on task', rank, 'after Reduce: data =  ', recvdata)
```

运行

```
mpiexec -n 3 python reduction2.py
```

## 如何优化通信

MPI所提供的一个有趣的特性是关于虚拟拓扑的。如前所属，所有的通信功能都指的是一组进程。我们总是在使用MPI_COMM_WORLD组，它包含了所有进程。它会为属于大小为n的通信器的每一个进程分配一个从0到n-1的rank值。不过，我们可以通过MPI为通信器分配一个虚拟拓扑。它为不同的进程定义了特殊的标签。这种机制可以提升执行性能。实际上，如果构建了虚拟拓扑，那么每个节点都只会于其虚拟邻居通信，这优化了性能

比如，如果rank值是随机分配的，那么消息在到达目的地前就会经过很多其他节点。除了性能问题以外，虚拟拓扑还可以确保代码更清晰，可读性更好。MPI提供了来个闹钟功能构建拓扑。第一种会创建笛卡儿拓扑，第二种会创建其他类型的拓扑。特别的，对于第二种情况，必须要为想要构建的图提供邻接矩阵。这里只讨论笛卡儿拓扑，通过它可以构建出几种广泛使用的结构：如网状、环形与螺旋状等。

创建笛卡儿拓扑的函数

```python
comm.Create_cart(number_of_rows, number_of_columns)

# 参数
number_of_rows		将要创建的网格的行数
number_of_columns	将要创建的网格的列数
```

示例

```python
from mpi4py import MPI
import numpy as np


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
neighbour_processes = [0, 0, 0, 0]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    # 生成的拓扑是一个2*2的网状结构，其大小等于输入的进程数
    grid_rows = int(np.floor(np.sqrt(comm.size)))
    grid_column = comm.size // grid_rows

    if grid_rows*grid_column > size:
        grid_column -= 1
    if grid_rows*grid_column > size:
        grid_rows -= 1

    if rank == 0:
        print("Building a %d x %d grid topology:" % (grid_rows, grid_column))

    # 构建笛卡儿拓扑
    cartesian_communicator = comm.Create_cart((grid_rows, grid_column), periods=(True, True), reorder=True)  # periods的值False，True待定
    # 为了找出第i个进程的位置，使用Get_coords()
    my_mpi_row, my_mpi_col = cartesian_communicator.Get_coords(cartesian_communicator.rank)
    # 获取拓扑
    neighbour_processes[UP], neighbour_processes[DOWN] = cartesian_communicator.Shift(0, 1)
    neighbour_processes[LEFT], neighbour_processes[RIGHT] = cartesian_communicator.Shift(1, 1)

    print("Process = %s row = %s column = %s ---> \
         neighbour_process[UP] = %s neighbour_process[DOWN] = %s\
         neighbour_process[LEFT] = %s neighbour_process[RIGHT] = %s"\
         (rank, my_mpi_row, my_mpi_col, \
          neighbour_processes[UP], neighbour_process[DOWN], neighbour_process[LEFT], neighbour_process[RIGHT]))
```

运行

```
mpiexec -n 4 python virtualTopology.py
```

