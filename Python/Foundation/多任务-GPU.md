# GPU编程

CPU由专为串行处理而优化的几个核心组成。GPU由数量众多的更小、更高效的核心组成，这些核心专为同时处理多个任务而设计，虽然每个核心的功能并不像CPU那么强大，但是用来并行处理一些小任务，还是有很大优势

实现GPU运算首先需要显卡(如NVIDIA显卡)的支持，并且根据需要安装相应的Python扩展库，如pycuda,pyopencl,theano,scikit-learn,NumbaPro,TensorFlow.不过，GPU加速并不适用于所有场景，如需要在CPU和GPU之间频繁传输数据，反而会影响效率，不如直接使用CPU的速度快

## pucuda

借助pycuda可以在python中访问NVIDIA显卡提供的CUDA并行计算API。windows上安装pycuda要求已经正确安装合适版本的CUDA和Visual Studio，然后`pip install pycuda`

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

