# MPI

[参考](https://www.jianshu.com/p/ee595dd0354a)

## 概述

- MPI

MPI 的全称是 Message Passing Interface，即消息传递接口。它是一种用于编写并行程序的标准，包括协议和和语义说明，他们指明其如何在各种实现中发挥其特性，有 MPICH、OpenMPI 等一些具体的实现，提供 Fortran、C、C++ 的相应编程接口。MPI 的目标是高性能，大规模性，和可移植性。MPI 在今天仍为高性能计算的主要模型。

MPI 的工作方式很好理解，我们可以同时启动一组进程，在同一个通信域中不同的进程都有不同的编号，程序员可以利用 MPI 提供的接口来给不同编号的进程分配不同的任务和帮助进程相互交流最终完成同一个任务。就好比包工头给工人们编上了工号然后指定一个方案来给不同编号的工人分配任务并让工人相互沟通完成任务。

MPI 的具体实现并没有提供 Python 的编程接口，这就使得我们没法直接地使用 Python 调用 MPI 实现高性能的计算，不过幸运的是，我们有 mpi4py。mpi4py 是一个构建在 MPI 之上的 Python 库，主要使用 Cython 编写，它以一种面向对象的方式提供了在 Python 环境下调用 MPI 标准的编程接口，这些接口是构建在 MPI-2 C++ 编程接口的基础之上的，因此和 C++ 的 MPI 编程接口非常类似，了解和有 C、C++ MPI 编程经验的人很容易地上手和使用 mpi4py 编写基于 MPI 的高性能并行计算程序。

- mpi4py

mpi4py 是一个构建在 MPI 之上的 Python 库，它使得 Python 的数据结构可以方便的在多进程中传递。

mpi4py 是一个很强大的库，它实现了很多 MPI 标准中的接口，包括点对点通信，集合通信、阻塞／非阻塞通信、组间通信等，基本上能用到的 MPI 接口都有相应的实现。不仅是任何可以被 pickle 的 Python 对象，mpi4py 对具有单段缓冲区接口的 Python 对象如 numpy 数组及内置的 bytes/string/array 等也有很好的支持并且传递效率很高。同时它还提供了 SWIG 和 F2PY 的接口能够将 C/C++ 或者 Fortran 程序在封装成 Python 后仍然能够使用 mpi4py 的对象和接口来进行并行处理。

- 信息传递

mpi4py 可以在不同的进程间传递任何可以被 pickle 系列化的内置和用户自定义 Python 对象，这些对象一般在发送阶段被 pickle 系列化为 ASCII 或二进制格式，然后在接收阶段恢复成对应的 Python 对象。

这种数据传递方式虽然简单通用，却并不高效，特别是在传递大量的数据时。对类似于数组这样的数据，准确来说是具有单段缓冲区接口（single-segment buffer interface）的 Python 对象，如 numpy 数组及内置的 bytes/string/array 等，可以用一种更为高效的方式直接进行传递，而不需要经过 pickle 系列化和恢复。

按照 mpi4py 的惯例，传递可以被 pickle 系列化的通用 Python 对象，可以使用通信子（Comm 类，后面会介绍）对象的以小写字母开头的方法，如 send()，recv()，bcast()，scatter()，gather() 等。但是如果要以更高效的方式传递具有单段缓冲区接口的 Python 对象，如 numpy 数组，则只能使用通信子对象的以大写字母开头的方法，如 Send()，Recv()，Bcast()，Scatter()，Gather() 等。

- MPI环境管理

mpi4py 提供了相应的接口 MPI.Init()，MPI.Init_thread() 和 MPI.Finalize() 来初始化和结束 MPI 环境。但是 mpi4py 通过在 **init**.py 中写入了初始化的操作，因此在我们 from mpi4py import MPI 的时候就已经自动初始化了 MPI 环境。

MPI_Finalize() 被注册到了 Python 的 C 接口 Py_AtExit()，这样在 Python 进程结束时候就会自动调用 MPI_Finalize()， 因此不再需要我们显式的去调用。

- 通信子(Communicator)

mpi4py 提供了相应的通信子的 Python 类，其中 MPI.Comm 是通信子的基类，在它下面继承了 MPI.Intracomm 和 MPI.Intercomm 两个子类，这跟 MPI 的 C++ 实现中是相同的。下图是通信子类的继承关系。

```
comm ---> intercomm
	 ---> intracomm ---> cartomm
	 				---> Distgraphcomm
	 				---> Graphcomm
```

同时它也提供了两个预定义的通信子对象：

```
- 包含所有进程的 MPI.COMM_WORLD；
- 只包含调用进程本身的 MPI.COMM_SELF。
```

可以由它们创建其它新的通信子。

可以通过通信子所定义的一些方法获取当前进程号、获取通信域内的进程数、获取进程组、对进程组进行集合运算、分割合并等等。

## 安装

- windows

