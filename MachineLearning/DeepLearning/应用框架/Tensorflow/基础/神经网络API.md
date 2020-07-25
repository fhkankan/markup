# 神经网络API

## Adam算法

````python
tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,epsilon=1e-08,name='Adam')

# 参数
α：学习率，需要尝试一系列的值，来寻找比较合适的
β1：常用的缺省值为 0.9
β2：Adam 算法的作者建议为 0.999
ϵ：Adam 算法的作者建议为epsilon的默认值1e-8
注：β1、β2、ϵ 通常不需要调试
````

## FC全连接

```python
tf.nn.softmax_cross_entropy_with_logits(labels=None, logits=None, name=None)
# 计算logits和labels之间的交叉熵损失。返回损失值列表
# 参数
- labels:标签值(真实值)
- logits:样本加权之后的值
 
tf.reduce_mean(input_tensor)
# 计算张量的尺寸的元素平均值
```

## 卷积网络

卷积 

```python
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None)
# 计算给定4-D输入和filter张量的2维卷积 
# 参数
- input:给定的输入张量，具有[batch,height,width,channel]，类型为float32/64
- filter:指定过滤器的权重数量，[filter_height, filter_width, in_channels, out_channels]
- strides:步长，[1,1,1,1]
- padding:零填充的方式："SAME","VALID"
    SAME:越过边缘取样，取样的面积和输入图像的像素宽度一致，公式：ceil(H/S)，H为输入的图片的高或宽，S为步长。无论过滤器的大小是多少，零填充的数量由API自动计算。
    VALID:不越过边缘取样，取样的面积小于输入的图像的像素宽度。不填充
```

> 在Tensorflow中，卷积API设置"SAME"之后，若步长为1，输出高宽和输入大小一样

激活函数

```python
tf.nn.relu(features, name=None)
# relu激活函数，返回结果
- features:卷积后加上偏置的结果
```

池化层

```python
tf.nn.max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
# 输入上执行最大池数
# 参数
- value:4-D的Tensor形状[batch, height, width, channels]，其中channel并不是原始图片的通道数，而是多少filter观察
- ksize:池化窗口大小，[1,1,1,1]
- strides:步长大小,[1,1,1,1]
- padding:使用填充算法的类型，"SAME","VALID"
```

