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

实现

```python
import pycuda.driver as cuda
#  自动根据GPU可用性和数量选择要使用的GPU，将创建一个后续代码运行中所需的GPU上下文。若需要，选中的设备和创建的上下文均可从pycuda.autoinit中访问
import pycuda.autoinit  
from pycuda.compiler import SourceModule  # SourceModule组件时一个必须编写GPU所需类C代码的对象

# 生成一个5*5的输入矩阵
import numpy
a = numpy.random.randn(5, 5)
# 矩阵中的项被转换为单精度模式，许多NVIDIA显卡只支持单精度
a = a.astype(numpy.float32)
# 从主机内存(CPU)中将输入数组加载至设备中(GPU)
# 设备上的内存分配通过cuda.mem_alloc安成。在执行函数内核时，设备和主机内存不能进行通信。者意味着，要在设备上并行运行一个函数，相关的数据必须已经存在于设备内存中。在将数据从主机内存复制到设备内存之前，必须分配设备上所需的内存
a_gpu = cuda.mem_alloc(a.nbytes)
# 将矩阵从主机内存复制到设备内存中
# 注意：a_gpu是一维的，在设备上需哟啊将其视为一维处理。所有这些操作都不要求调用内核，由主处理器直接完成
cuda.memcpy_htod(a_gpu, a)
# SourceModule用于定义内核函数(类C函数)doubleMatrix，该函数将每个数组乘以2
# __global__限定符表示函数将在设备上执行。只有CUDA中的NVCC编译器将执行该任务
# idx参数是矩阵多饮，用线程坐标表示
mod = SourceModule("""
        __global__ void doubleMatrix(float *a)
        {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] *= 2;
        }
        """)
# 内核函数会被CUDA编译器NVCC自动编译。若没有出错，将会创建指向被编译函数的指针。
# get_function返回是指向函数func的标识符
func = mod.get_function("doubleMatrix")
# 要在设备上执行函数，先正确配置执行过程，确定用于辨别、区分属于不同区块的线程的坐标。
# 线性化地输入矩阵a_gpu和一个线程区块执行内核函数，线程区块在x方向有5个线程，y方向有5个线程，z方向有1个线程
# 配置好内核调用参数后，就可以调用内核参数，并在设别上并行执行指令了，每个线程执行相同的代码内核
func(a_gpu, block=(5, 5, 1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print("ORIGINAL MATRIX")
print a
print("DOUBLED MATRIX AFTER PyCUDA EXECUTION")
print a_doubled
```

拓展

```
一个warp一次执行一个通用指令。因此，要最大化地提高这种结构的效率，必须使用同一个线程的执行流程。如果指定某个多处理器运行多个线程区块，会将线程区块氛围多个warp，并由warp调度器来实施调度
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

### 内存模型

为了最大限度地利用可用资源，PyCUDA程序应该遵守SM结构及其内部组织形式所要求的规则：SM对于线程的性能会有限制。在支持CUDA的GPU显卡中，有4类内存，如下

```
- 寄存器(registers):每个线程将被分配一个寄存器。每个线程智能访问自身的寄存器，而无法反问其他线程的寄存器，即使与对方同属于一个线程区块
- 共享存储器(shared memory):在共享存储器中，每个线程区块都有一个其内部线程共享的内存，这部分内存速度极快
- 常数存储器(constant memory):一个网格中的所有线程一直都可以访问这部分内存，但只能在读取时访问。常数存储器中的数据在应用持续期间一直存在
- 全局存储器(global memory):所有网格中的线程(包括所有内核)都可以访问全局存储器，其中的常数存储器在应用持续期间一直存在
```

如果要让PyCUDA程序达到令人满意的性能，一个关键点就是要理解并非所有的内存都是一样的，必须充分利用每种类型的内存。基本的思路是痛殴使用共享存储器，尽量减少对全局存储器的访问。这种技巧通常用于区分问题的域/陪域(domain/codomain)，好让线程区块在一个闭合的数据子集中进行运算。这样，属于该区块的线程将共同加载共享的全局存储器，用于在内存中进行运算，然后再充分利用这块内存区的高速特性

每个线程要执行的基本步骤如下

```
1.从全局存储器中将数据加载至共享存储器
2.同步该区块中的全部线程，让每个线程可以读取共享存储器的安全位置，共享存储器中包含所有其他线程
3.处理共享存储器中的数据
4.若有必要，再次进行同步，确保共享存储器中已经获得最新的结果
5.在全局存储器中写入结果
```

求两个矩阵的乘积。

以标准方式进行的矩阵乘法，以及相应的线性代码，用于计算应该从矩阵数据的哪一行哪一列加载元素

```python
void SequentialMatrixMultiplication(float *M, float *N, float *P, int width)
{
    for (int i=0; i<width; ++i){
        for(int j=0; j<width; ++j){
            float sum = 0;
            for(int k=0; k<width; ++k){
                float a = M[I*width+k];
                float b = M[I*width+j];
                sum += a*b
                }
            P[I*width+j] = sum;
            }
        }
}
```

让每个线程负责计算矩阵中的一个元素，那么内存访问将占用该算法的大部分执行时间。可用一个线程区块来计算输出的一个子矩阵，这样就可以重复使用从全局存储器加载的数据，并实现各个线程之间的协作，最小化每个线程的内存访问

```python
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# 初始化设备
import  pycuda.autoinit

kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float Pvalue = 0;
    for(int k=0; k<%(MATRIX_SIZE)s; ++k){
        float Aelement = a[ty * %(MATRIX_SIZE)s + k];
        float Belement = b[k * %(MATRIX_SIZE)s + tx];
        Pvalue += Aelement * Belement;
    }
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""
# 准备好输入矩阵以及用于保存结果的输出矩阵
MATRIX_SIZE = 5
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
c_cpu = np.dot(a_cpu, b_cpu)
# 将矩阵转移到GPU设备中
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

kernel_code = kernel_code_template % {'MATRIX_SIZE': MATRIX_SIZE}
mod = compiler.SourceModule(kernel_code)
matrixmul = mod.get_function("MatrixMulKernel")
matrixmul(
        a_gpu, b_gpu, c_gpu,
        block = (MATRIX_SIZE, MATRIX_SIZE, 1),
        )

# 打印结果
print '-'*80
print 'Matrix A (GPU)'
print a_gpu.get()

print '-'*80
print 'Matrix B (GPU)'
print b_gpu.get()

print '-'*80
print 'Matrix C (GPU)'
print c_gpu.get()

print '-'*80
print 'CPU-GPU difference:'
print c_cpu - c_gpu.get()

np.allclose(c_cpu, c_gpu.get())
```

### GPUArray调用内核

上面的例子，使用如下的类调用一个内核函数

```python
pycuda.compiler.SourceModule(kernel_source, nvcc="nvcc", options=None, other_options)
```

这行代码从CUDA源代码kernel_source中创建一个模块，然后使用给定的选项调用NVIDIA的NVCC编译器编译代码

PyCUDA引入`pycuda.gpuarray.GPUArray`类，提供了使用CUDA执行计算的高层接口

```python
class pycuda.gpuarray.GPUArray(shape, dtype, *, allocator=None, order="C)
```

其工作方式类似`numpy.ndarray`，都是将数据保存至计算设备中并在该设备中进行计算。shape和dtype参数与它们在Numpy中的使用方式一模一样

GPUArray类中的所有数值计算方法都支持广播标量数据类型。gpuarray的创建很简单。一种方法是创建一个Numpy数组然后转换，如下所示

```python
import pycuda.gpuarray as gpuarray
from numpy.random import randn
from numpy import float32, int32, array
x = randn(5).astype(float32)
x_gpu = gpuarray.to_gpu(x)
```

示例：GPU计算的一个常见使用场景，作为其他计算的一个辅助步骤

```python
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
# 输入矩阵a_gpu中包含随机生成的项。若要在GPU中执行计算(将矩阵的所有项*2)
a_doubled = (2*a_gpu).get()
print a_doubled
print a_gpu
```

拓展

```
pycuda.gpuarray.GPUArray类支持所有算术运算符，以及许多方法和函数，所有这些都是参照NumPy中的对应功能实现的。此外，pycuda.cumath中还提供了许多特殊函数。可以使用pycuda.curandom中的功能，生成近似于均匀分布随机数组成的数组
```

### 对逐元素表达式求值

`pycuda.elementwise.ElementwiseKernel`函数支持在复杂表达式上执行内核，这些复杂表达式由一个或多个操作数组成，共同构成一个计算步骤，具体如下

```python
ElementwiseKernel(arguments, operation, name, optional_parameters)

# 参数
arguments:这是一个C参数列表，包含内核执行涉及的所有参数
operation:这是要在给定参数上执行的操作。如果参数是向量，那么每一项都将执行所有操作
name:内核的名称
optional_parameters:这是以下示例中没有用到的编译指令
```

示例

假设有由50个元素组成的两个向量`input_vector_a`和`input_vector_b`，这两个向量都是随机构建的。任务是要计算出它们的线性组合

```python
import pycuda.autoinit
import numpy
# 初始化两个向量
from pycuda.curandom import rand as curand
from pycuda.elementwise import ElementwiseKernel
import numpy.linalog as la

# 定义随机变量
input_vector_a = curand((50,))
input_vector_b = curand((50,))
# 定义两个乘法的系数
mult_coefficient_a = 2
mult_coefficient_b = 5

# 构造函数
linear_combination = ElementwiseKernel(
        "float a, float *x, float b, float *y, float *c",  # 参数列表
        "c[i] = a*x[i] + b*y[i]",  # 定义了如何操作参数列表
        "linear_combination"  # 将ElementwiseKernel命名为linear_combination
        )
# 定义最终保存结果的向量，是一个和输入向量维度相同的空向量
linear_combination_result = gpuarray.empty_like(input_vector_a)
# 对内核求值
linear_combination(
        mult_coefficient_a, input_vector_a,
        mult_coefficient_b, input_vector_b,
        linear_combination_result
        )

print("INPUT VECTOR A=")
print(input_vector_a)
print("INPUT VECTOR B=")
print(input_vector_b)
print("RESULTING VECTOR C=")
print(linear_combination_result)

print("CHECKING THE RESULT EVALUATING THE DIFFERENCE VECTOR BETWEEN THE LINEAR COMBINATION OF A AND B")
print("C - (%sA + %sB) =" % (mult_coefficient_a, mult_coefficient_b))
print(linear_combination_result - (mult_coefficient_a*input_vector_a + mult_coefficient_b*input_vector_b))
# 检查结果
assert la.norm((linear_combination_result - (mult_coefficient_a*input_vector_a + mult_coefficient_b*input_vector_b)).get()) < le-5
```



### 进行MapReduce操作

PyCUDA提供了在GPU上执行归约操作(reduction operation)的狗狗能耐。可通过`pycuda.reduction.ReductionKernel`方法实现

```python
ReductionKernel(dtype_out, arguments, map_expr, reduce_expr, name, optional_parameters)

# 参数
dtype_out:输出的数据类型，必须通过numpy.dtype数据类型指定
arguments:一个C参数列表，包含了归约操作涉及的所有参数
map_expr:一个表示映射操作的字符串，该表达式中的每个向量必须通过变量i引用
reduce_expr:一个表示归约操作的字符串，该表达式中国的操作数以小写字母表示，如a,b,c,...,z
name:与ReductionKernel相关联的名称，内核使用该名称进行编译
optional_parameters:这些参数在该实例中并不重要，因为他们是编译器的指令
```

此方法在向量参数(至少一个)执行一个内核，在向量参数的每个项上执行map_expr，然后在操作结果上执行reduce_expr

通过实例化ReductionKernel类求两个向量(500个元素)的点积(dot product)。点积，也被称为标量积(scalar product)，是一个代数运算，操作数是两个长度相等的数字序列(通常是坐标向量)，结果为两个数字序列中对应乘积之和。这是一个常见的MapReduce操作，map操作是按索引求乘积，reduction操作是求所有乘积之和

```python
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy
from pycuda.reduction import ReductionKernel

# 定义两个整数向量，值在0～399间
vector_length = 400
input_vector_a = gpuarray.arrange(vector_length, dtype=numpy.int)
input_vector_b = gpuarray.arrange(vector_length, dtype=numpy.int)
# 定义MapReduce操作
dot_product = ReductionKernel(
  numpy.int,   # 输出将是一个整数
  arguments="int *x, int *y",  # 以类C语言的标记定义了输入(整数数组)的数据类型
  map_expr="x[i]*y[i]",  # map操作，即求两个向量第i项元素的乘积 
  reduce_expr="a+b",  # reduction操作，即求所有乘积之和
  neutral="0"
)
# 调用ReductionKernel实例的最终结果是一个仍存在于GPU中的GPUArray标量。可调用get方法将其放到CPU中，或者也可以在GPU中使用这个值，如下调用内核函数
dot_product = dot_product(input_vector_a, input_vector_b).get()

print("INPUT VECTOR A")
print(input_vector_a)
print("INPUT VECTOR B")
print(input_vector_b)
print("RESULT DOT PRODUCT OF A * B")
print(dot_product)
```

## NumbaPro

NumbaPro是一个Python编译器，提供了基于CUDA的API编程接口，可以编写CUDA程序。它专门设计用来执行于数组相关的计算任务，和广泛使用的Numpy库类似。与数组相关的计算任务中的数据并行性非常适合由GPU等加速器完成。NumbaPro可接受Numpy数组类型，并用这些数组生成可在GPU或多核CPU上执行高效率编译代码。

该编译器支持对python函数指定类型签名(type signature)，可开启运行时编译特性(被称为JIT编译)。

其中最重要的装饰器如下所示

```
- numbapro.jit:支持开发者编写类CUDA的函数。遇到该装饰器时，编译器会将装饰器下的代码编译为伪汇编语言PTX，在GPU上执行。
- numbapro.autojit:注释一个函数为延迟编译(deferred compilation)，这意味着带有该签名的每个函数只会被编译一次
- numbapro.vectorize:创建一个所谓的ufunc对象(Numpy中的通用函数),接受一个函数，并以向量参数并行执行
- guvectorize:创建所谓的gufunc对象(Numpy中的泛通用函数)。这种对象可以操作整个子数组
```

这些装饰器都有一个叫目标(target)的编译器指令，用于选择代码生成目标。NumbaPro编译器支持并行和GPU等目标。并行目标可用于向量化操作，而GPU指令则将计算负载交给NVIDIA的CUDA GPU进行。

### 准备工作





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

