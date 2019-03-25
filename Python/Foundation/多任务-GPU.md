# GPU编程

图形处理器(graphics processing unit， GPU)是一个专门用来处理数据以使用多边图形基本体(polygonal primitives)渲染图线的电子线路班。尽管是被设计用来渲染图像的，但是GPU一直在不断演变，变得越来越复杂，也越来越高效，能满足实时渲染和离线渲染需求，可以高效地执行科学计算。GPU的特色是具备高度并行化的结构，因此它能以高效的方式处理大规模的数据集。这一特性与图形硬件性能的快速提升、高可编程性结合在一起，使得科学界开始关注GPU，考虑其用于渲染图像之外的其他用途。传统GPU使功能固定的设备，整个渲染管道都构建在硬件中。这限制了图形程序员，迫使他们使用不同的、高效及高质量的渲染算法。因此，后来开发出了一种新型GPU，使用数百万轻量级的并行核心构建而成，可使用*着色器*(shader)对其编程，实现图形渲染。这是计算机图形领域和游戏业取得的最大进展之一。由于出现了大量可编程的核心，GPU生产商开始研发用于并行编程的模型。每个GPU都由多个被称为*流式多处理器*(Streaming Multiprocessor,简称SM)的处理单元组成，这些处理单元代表了并行的第一个逻辑层，而且每个SM之间是相互独立且同时运转的。

每个SM包含一组流式处理器，每个流式处理器都具备一个真正的执行核心，可以线性地运行一个线程。SP是执行逻辑中的最小单元，代表了更细粒度并行的层次。SM和SP的区分在本质上是结构化的，不过可以进一步勾勒出GPU中SP的逻辑组织，这些SP聚集在逻辑区中，这些逻辑区都共享一种特定的执行模式。组成一组SP的所有核心同时执行相同的指令。这就是**单指令多数据**(Single instruction multiple data，SIMD)模式。

每个SM都有一些寄存器(register)，寄存器是可以快速访问的内存区域，这块区域是临时的，只能本地访问(不同核心之间无法共享)，而且大小有限制。它可以用于存储单个核心中经常使用的值。**图形处理器通用计算**(general-purpose computing on graphics processing units，GP-GPU)这个领域致力于研究利用GPU计算能力快速执行计算所需的技术，这也得益于GPU内部高度的并行化。如前多数，GPU的结构与传统处理器有很大区别；因此，他们存在的问题也完全不同，需要使用特定的编程技巧。图形处理器最突出的特性就是拥有大量可用的处理核心，可以运行许多相互竞争的执行线程，这些线程将部分同步式地执行相同的操作。如果希望将工作分成许多小部分，针对不同的数据执行相同的操作，该特性将十分有用且高效。相反，如果操作非常注重线性化和逻辑顺序，就很难利用这一架构。GPU计算的编程范式被称为流式处理(Stream Processing)，因为数据剋被看作一个全部由数值组成的流，在这个流上同步执行了相同的操作。

CPU由专为串行处理而优化的几个核心组成。GPU由数量众多的更小、更高效的核心组成，这些核心专为同时处理多个任务而设计，虽然每个核心的功能并不像CPU那么强大，但是用来并行处理一些小任务，还是有很大优势

实现GPU运算首先需要显卡(如NVIDIA显卡)的支持，并且根据需要安装相应的Python扩展库，如pycuda,pyopencl,theano,scikit-learn,NumbaPro,TensorFlow.不过，GPU加速并不适用于所有场景，如需要在CPU和GPU之间频繁传输数据，反而会影响效率，不如直接使用CPU的速度快

## PyCUDA

### 概述

PyCUDA是NVIDIA开发的GPU编程软件库CUDA(Compute Unified Device Architecture)在python中的封装。

- 混合编程模型

CUDA的混合编程模型是通过多个特定的C语言标准库扩展实现的，PyCUDA亦然。只要有可能，就会创建这些扩展，语法上类似C标准库中的函数调用。这样，我们就能以相对简单的方式使用包含主机(host)和设备(device)代码的混合编程模型。NVCC编译器对这两个逻辑部件进行管理。以下是对该编译器工作原理的简要描述：

```
- 将设备代码与主机代码分离
- 调用默认编译器(如GCC)编译主机代码
- 以二进制(Cubin对象)或汇编代码(PTX代码)的形式构建设备代码
- 生成一个主机键“global”，其中也包括PTX代码
```

编译后的CUDA代码将在运行时由驱动器转换为特定于设备的二进制文件。上面提到的所有步骤都是PyCUDA在运行时完成的，因此它也算事一个即时(just-in-time，JIT)编译器。这一方式的缺点是应用的加载时间会增加，因为这是唯一保持向前兼容的方法，即可以在实际编译时不存在的设备上执行运算。所以，JIT编译可以让应用与构建于更改计算能力架构之上的未来设备兼容，因此它还不能生成任何二进制代码。PyCUDA的执行模式如下

![PyCUDA执行模块](images/PyCUDA执行模块.png)



- 内核与线程层级

CUDA程序的一个重要元素就是其内核(kernel)。它代表可以并行执行的代码，其基础规格说明将在稍后举例说明。每个内核的执行均由叫做线程(thread)的计算单元完成。与CPU中的线程不同，GPU线程更加轻量，上下文的切换不会影响代码性能，因为切换可以说是瞬时完成的。为了确定运行一个内核所需的线程数及其逻辑组织形式，CUDA定义了一个二层结构。在最高一层中，定义了所谓的区块网格(grif of blocks)。这个网格代表了线程区块的二维结构，而这些线程区块是三维的。示意图如下所示

![PyCUDA二层结构中(三维)线程的分布png](images/PyCUDA二层结构中(三维)线程的分布png.png)

基于这一结构，内核函数被调用时，必须提供额外的参数，来自定网格和区块大小。

### 安装

不同操作系统下安装PyCUDA可以[参考](http;//wiki.tiker.net/PyCuda/Installation)

验证PyCUDA是否正常安装

```python
import pycuda.driver as drv

# 初始化pycuda.driver，其暴漏了CUDA的驱动器层编程接口，这比CUDA C语言中的“运行时层”编程接口更加灵活，而且拥有运行时所不具备的一些特性
drv.init()
print "%d device(s) found." % drv.Device.count()  # 显卡数
# 发现每块GPU显卡，就打印显卡信息
for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print "Device #%d: %s" % (ordinal, dev.name())  # 名称
    print "Compute Capability: %d.%d" % dev.compute_capability()  # 计算能力
    print "Total Memory: %s KB" % (dev.total_memory()//(1024))  # 总内存
```

### 构建PyCUDA应用

PyCUDA编程模式时为了能同时在CPU和GPU上执行程序而设计的，其中线性部分在CPU上执行，更耗资源的数值计算部分则在GPU上执行。以线性模式执行的阶段是在CPU(主机)上实现和执行的，而以并行模式执行的步骤则是在GPU(设备)上实现并执行的。在设备上并行执行的函数被称为内核。在设别上执行通用函数内核的步骤如下

```
- 在设备上分配内存
- 将数据从主机内存转移到设备上分配的内存中
- 运行设备：运行配置程序、调用内核函数
- 将结果从设备内存转移到主机内存中
- 释放设备上分配的内存
```

![PyCUDA编程模式](images/PyCUDA编程模式.png)

- 具体实现

构建一个5*5的随机数组，按照以下步骤

```
- 在CPU上创建一个5*5的数组
- 将数组转移到GPU
- 在GPU中对数组执行一个操作(将数组中的所有项翻一倍)
- 将数组从GPU转移到CPU
- 打印结果
```









如下案例：使用pycuda在GPU上并行判断素数，测试结果显示在640核GPU上的运行速度是4核CPU的8倍左右

```python
import time
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

# 编译C代码进入显卡，并判断素数
mod = SourceModule('''
__global__void isPrime(int*dest, int*a, int*b)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j;
    for(j=2;j<b[i];j++)
    {
        if(a[i]%j==0){
            break
        }
    }
    if(j >= b[i]){
        dest[i] = a[i]
    }
}
''')

# 定义待测数值范围，以及每次处理的数字数量
end = 10000000
size = 1000
# 获取函数
isPrime = mod.get_function("isPrime")
result = 0
start = time.time()
# 分段处理，每次处理1000个数字
for i in range(end//start):
    startN = i*size
    a = np.array(range(startN, startN+size)).astype(np.int64)
    dest = np.zeros_like(a)
    isPrime(drv.Out(dest), drv.In(a), drv.In(b),
            block=(size, 1, 1), grid=(2,1))
    result += len(set(filter(None, dest)))
print(time.time()-start)
# 上面的代码中把1也算上了，这里减去
print(result -1)
```

## pyopencl

pyopencl使得在python中调用OpenCL并行计算API。OpenCL是跨平台的并行编程标准，可以运行在个人计算机、服务器、移动终端及嵌入式系统等多个平台，既可以在CPU上，也可以运行在GPU上，大幅度提高了各类应用中数据处理速度，包括游戏、娱乐、医学软件及科学计算等。

下面的案例使用pyopencl在GPU上并行判断素数，测试结果显示在640核GPU上的运行速度是4核CPU的12倍左右

```python
import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

# 判断素数的C语言版本GPU代码
isPrime = ElementwiseKernel(
    ctx,
    'long*a_g, long*b_g, long*res_g',
    '''
        int j;
        for(j=2; j<b_g[i]; j++){
            if(a_g[i]%j == 0){
                break;
            }
        }
        if(j >= b_g[i]){
            res_g[i] = a_g[i];
        }
    ''',
    'isPrime'
)
# 定义待测数值范围，以及每次处理的数字数量
end = 100000000
start_end - range(2, end)
size = 1000
result = 0
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
# 对指定范围内的数字进行分批处理
for i in range(end//size + 1):
    startN = i*size
    # 本次需要处理的数字范围
    a_np = np.array(start_end[startN:startN + size]).astype(np.int64)
    # b_np里的数字是a_np中数字的平方根取整后加1
    b_np = np.array(list(map(lambda x: int(x**0.5)+1, a_np))).astype(np.int64)
    # 把数据写入GPU
    a_g = cl.array.to_device(queue, a_np)
    b_g = cl.array.to_device(queue, b_np)
    res_g = cl.array.zeros_like(a_g)
    # 批量判断
    isPrime(a_g, b_g, res_g)
    t = set(filter(None, res_g.get()))
    # 记录本批数字中素数的个数
    result += len(t)
print(result)
```

## tensorflow

tensorflow是一个用于人工智能的开源神器，是一个采用数据流图(data flow graphs)用于数值计算的开源软件库。数据流图使用节点(nodes)和边线(edges)的有向图来描述数学计算，图中节点表示数学操作，也可以表示数据输入的起点或者数据输出的终点，而边线白哦是在节点之间的输入输出关系，用来运输大小可动态调整的多维数据数组，也就是张量(temsor)。tensorflow可以在普通计算机、服务器和移动设备的CPU和GPU上展开计算，具有很强的可移植性，支持C++、python等多种语言

使用tensorflow中的梯度下降算法求解变量最优值

```python
import tensorflow as tf
import numpy as np
import time

# 使用Numpy生成随机数据，共2行100列个点
x_data = np.float32(np.random.rand(2, 200))
# 矩阵乘法
# 这里的W=[0.100, 0.200]和b=0.300是理论数据，通过后面的训练来验证
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型，训练求解W和b
# 初始值b=[0.0]
b = tf.Variable(tf.zeros([1]))
# 初始值W为1*2的矩阵，元素介于[-1.0, 1.0]区间
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
# 构建训练模型，matmul为矩阵乘法运算
y = tf.matmul(W, x_data) + b

# 最小均方差
loss = tf.reduce_mean(tf.square(y-y_data))
# 使用梯度下降算法进行优化求解
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.device('/gpu:0'):
    with tf.Session() as sess:
        # 初始化
        sess.run(init)
        # 拟合平面， 训练次数越多约精确，但是也没有必要训练过多
        for step in range(0, 201):
            sess.run(train)
            # 显示训练过程，这里掩饰了两种查看变量值的方法
            print(step, sess.run(W), b.eval())
```

