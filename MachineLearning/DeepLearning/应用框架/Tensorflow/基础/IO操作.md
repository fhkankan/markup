# IO操作

在TensorFlow中有三种读取数据的方式：

```
QueueRunner：基于队列的输入管道从TensorFlow图形开头的文件中读取数据
Feeding：运行每一步，Python代码提供数据
预加载数据：TensorFlow图中的张量包含所有数据(对于小数据集)
```

## QueueRunner

通用文件读取流程三个阶段：构造文件名队列、读取与解码、批处理

### 构造文件名队列

将需要读取的文件的文件名放入文件名队列

```python
tf.train.string_input_producer(string_tensor, num_epochs=None, shuffle=True)

# 参数
- string_tensor:含有文件名+路径的1阶张量
- num_epochs:过几遍数据，默认无限个数据
- return 文件队列
```

### 读取与解码

从队列中读取文件内容，并进行解码操作

- 读取文件内容

默认每次只读取一个样本：文本文件默认一次读取一行，图片文件默认一次读取一张图片，二进制文件一次读取指定字节数(最好是一个样本的字节数)，TFRecords默认一次读取一个example

```python
tf.TextLineReader
# 阅读文本文件逗号分隔值(CSV)格式，默认按行读取。返回读取器实例
tf.WholeFileReader
# 读取图片文件。返回读取器实例
tf.FixedLengthRecordReader(record_bytes)
# 读取每个记录是固定数量字节的二进制文件。record_bytes是整型，指定每次读取（一个样本）的字节数。返回读取器实例
tf.TFRecordReader
# 读取TFRecords文件。返回读取器实例

读取器实例.read(file_queue)
# 返回一个Tensors元组(key是文件名字，value是默认的内容即一个样本
```

由于默认只会读取一个样本，所以若要进行批处理，需要使用`tf.train.batch`或`tf.train.shuffle_batch`进行批处理，便于之后指定每批次多个样本的训练

- 内容解码

读取不同类型的文件，也应该对读取到的不同类型的内容进行相对应的解码操作，解码成统一的Tensor格式

```python
tf.decode_csv
# 解码文本文件内容
tf.image.decode_jpeg(contents)
# 将JPEG编码的图像解码为uint8张量。返回uint8张量，3-D形状[height,width,channels]
tf.decode_raw
# 解码二进制文件内容。与tf.FixedLengthRecordReader搭配使用，二进制读取为uint类型
```

解码阶段，默认所有内容均解码为`tf.uint8`类型，如果之后需要转换为指定类型则可使用`tf.cast()`进行相应转换

### 批处理

解码之后，可以直接获取默认的一个样本内容了，但如果想要获取多个样本，需要加入到新的队列进行批处理

```python
tf.train.batch(tensor, batch_size,num_threads=1, capacity=32, name=None)
# 读取指定大小(个数)的张量，返回tensors
# 参数
- tensor：可以是包含张量的列表，批处理的内容翻到列表当中
- batch_size：从队列中读取的批处理大小
- num_threads：进入队列的线程数
- capacity：整数，队列中元素的最大数量

tf.train.shuffle_batch(tensors, batch_size, capacity, min_after_dequeue, num_threads=1, name=None)
```

- 线程操作

以上用的队列都是`tf.train.QueueRunner`对象。

每个`QueueRunner`都负责一个阶段，`tf.train.start_queue_runners`函数会要求图中的每个`QueueRunner`启动它的运行队列操作的线程(这些操作需要再会话中开启)

```python
tf.train.start_queue_runners(sess=None, coord=None)
# 收集图中所有的队列线程，默认同时启动线程。返回所有线程
# 参数
- sess：所在的会话
- coord：线程协调器

tf.train.Coordinator()
# 线程协调员，对线程进行管理和协调。返回线程协调员实例
# 实例函数
- request_stop()： 请求停止
- should_stop()：询问是否结束
- join(threads=None, stop_grace_period_secs=120)：回收线程
```

## 数据

### 图片数据

- 图像基本知识

> 图片三要素

组成一张图片特征值是所有的像素值。有三个维度：图片长度、图片宽度、图片通道数。

灰度图：长、宽、1，每个像素点是0～255的数

彩色图：长、宽、3，每个像素点是0～255的数

> 张量形状

一张图片可以被表示为一个3D张量，即其形状为`[height,widht,channel]`

单个图片：`[height,width,channel]`

多个图片：`[batch,height,width,channel]`，`batch`表示一个批次的张量数量

- 图片特征值处理

将图片缩放到统一的大小：1.某些样本数据量过大，缩小后不影响识别；2.各个样本大小不一，不便于批量处理

```python
tf.image.resize_images(images, size)
# 缩小放大图片。返回4D格式或3D格式图片
# 参数
- images：4D形状[batch,height,width,channels]或3D形状张量[height,width,channels]的图片数据
- size：1Dint32张量：new_height,new_width,图像的新尺寸
```

- 数据格式

存储：uint8(节约空间)

矩阵计算：float32(提高精度)

- 读取案例

流程：构造图片文件名队列、读取图片数据并进行解码、处理图片形状放入批处理队列中、开启会话线程运行

```python
import tensorflow as tf
import os

def picture_read(file_list):
    # 狗图片读取
    # 1.构建文件名队列
    file_queue = tf.train.string_input_producer(file_list)
    # 2.读取与解码
    # 读取
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    # 解码
    image = tf.image.decode_jpeg(value)
    # 图像的形状类型修改
    image_resized = tf.image.resize_images(image, [200, 200])
    # 静态形状修改
    image_resized.set_shape(shape=[200, 200, 3])
    # 3.批处理
    tf.train.batch([image_resized], batch_size=100, num_threads=2, capacity=100)

    # 4.开启会话
    with tf.Session() as sess:
        # 开启线程
        # 创建线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        key_new, value_new = sess.run([key, value])
        # 回收线程
        coord.request_stop()
        coord.join(threads)
if __name__ == "__main__":
    # 构造路径+文件名的列表
    filename = os.listdir("./log")
    file_list = [os.path.join("./dog/", file) for file in fliename]
    picture_read()
```

### 二进制数据

流程：构造文件队列、读取二进制数据并进行解码、处理图片数据形状及类型放入批处理队列中、开启会话线程运行

```python
import tensorflow as tf
import os

class Cifar(object):

    def __init__(self):
        # 初始化操作
        self.height = 32
        self.width = 32
        self.channels = 3
        # 字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.label_bytes + self.image_bytes

    def read_and_decode(self, file_list):
        # 1.构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)
        # 2.读取与解码
        # 读取
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        key, value = reader.read(file_queue)
        # 解码
        decoded = tf.decode_raw(value, tf.uint8)
        # 将目标值和特征值分割
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        # 调整图片形状
        image_reshaped = tf.reshape(image, shape=[self.channels, self.height, self.width])
        # 转置，将图片的顺序转为height,width,channels
        image_transposed = tf.transpose(image_reshaped, [1,2,0])
        # 调整图像类型
        image_cast = tf.cast(image_transposed, tf.float32)
        # 3.批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=2, capacity=100)
        # 4.开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            key_n, value_n, decoded_n, label_n, image_n, image_reshaped_n, image_transposed_n = \
             sess.run([key, value, decoded, label, image, image_reshaped, image_transposed])
            label_value, image_value = sess.run([label_batch, image_batch])
            print('new-key-value:\n{}\n-{}'.format(key_n, value_n))
            print('new-decoded:\n', decoded_n)
            print('new-lable:\n', label_n)
            print('new-image:\n', image_n)
            print('new-image_reshaped:\n', image_reshaped_n)
            print('new-image_transposed:\n', image_transposed_n)
            print("label_value:\n", label_value)
            print("image_value:\n", image_value)
            # 回收线程
            coord.request_stop()
            coord.join(threads)
if __name__ == "__main__":
    # 构造路径+文件名的列表
    file_name = os.listdir("./cifar-10-batches-bin")
    file_list = [os.path.join("./cifar-10-batches-bin/", file) for file in file_name if file[-3:] == "bin"]
    print(file_list)
    cifar = Cifar()
    cifar.read_and_decode(file_list)
```

### TFRecords

TFRecords其实是一种二进制文件，虽然不如其他格式好理解，但是他能更好地利用内存，更方便复制和移动，并且不需要单独的标签文件。文件格式`*.tfrecords`

使用步骤

```
1. 获取数据
2. 将数据填入到Example协议内存块(protocol buffer)
3. 将协议内存块序列化为字符串，并且通过tf.python_io.TFRecordWriter写入到TFRecords文件
```

- Example

Example结构
> 这种结构很好地实现了数据和标签(训练的类别标签)或其他属性数据存储在同一个文件中

```
Example:
	features{
	  feature{
	    key: "image"
	    value{
	      bytes_list{
	        value: "\377..."
	      }
	    }
	  }
	  feature{
	    key: "label"
	    value{
	      int64_list{ 
	        value: 9
	      }
	    }
	  }
	}
```

写入相关函数

```python
# 实例化一个example对象
example = tf.train.Example(features=tf.train.Features(
    feature={
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
   }))

tf.train.Example(features=None)
# 写入tfrecords文件，返回example格式协议块
# 参数features:tf.train.Features类型的特征实例

tf.train.Features(feature=None)
# 构建每个样本的信息键值对，返回Features类型
# 参数feature：字典数据，key为要保存的名字，value为tf.train.Feature实例

tf.train.Feature(options)
# 参数options:支持存入的类型如下
- tf.train.Int64List(value=[Value])
- tf.train.BytesList(value=[Bytes])
- tf.train.FloatList(value=[Value])
```

读取相关函数

```python
# 解析example的一个步骤
feature = tf.parse_single_example(values, features={
    "image": tf.FixedLenFeature([], tf.string),
    "label": tf.FixedLenFeature([], tf.int64)
})

tf.parse_single_example(serialized, features=None, name=None)
# 解析一个单一的Example原型。返回一个键值对组成的字典，键为读取的名字
# 参数
- serialized:标量字符串Tensor，一个序列化的Example
- features:dict字典数据，键为读取的名字，值为FixedLenFeature
    
tf.FixedLenFeature(shape, dtype)
# 参数
- shape:输入数据的形状，一般不指定，为空列表
- dtype:输入数据类型，与存储进文件的类型要一致，只能是float32,int64,string
```


- 写入与读取实例

```python
import tensorflow as tf
import os

class Cifar(object):

    def __init__(self):
        # 设置图像大小
        self.height = 32
        self.width = 32
        self.channels = 3
        # 设置图像字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.label_bytes + self.image_bytes

    def read_binary(self):
        # 1.构造文件名队列
        file_name = os.listdir("./cifar-10-batches-bin")
        file_list = [os.path.join("./cifar-10-batches-bin/", file) for file in file_name if file[-3:] == "bin"]
        file_queue = tf.train.string_input_producer(file_list)
        # 2.读取与解码
        # 读取
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        key, value = reader.read(file_queue)  # key文件名，value样本
        # 解码
        decoded = tf.decode_raw(value, tf.uint8)
        # 将目标值和特征值分割
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        # 调整图片形状
        image_reshaped = tf.reshape(image, shape=[self.channels, self.height, self.width])
        # 转置，将图片的顺序转为height,width,channels
        image_transposed = tf.transpose(image_reshaped, [1,2,0])
        # 调整图像类型
        image_cast = tf.cast(image_transposed, tf.float32)
        # 3.批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size=100, num_threads=2, capacity=100)
        # 4.开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            label_value, image_value = sess.run([label_batch, image_batch])
            # 回收线程
            coord.request_stop()
            coord.join(threads)
        return image_value, label_value

    def write_to_tfrecords(self, image_batch, label_batch):
        # 将样本的特征值和目标值一起写入tfrecords文件
        with  tf.python_io.TFRecordWriter("cifar10.tfrecords") as writer:
            # 循环构造example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        }
                ))
                # 将序列化后的example写入文件
                writer.write(example.SerializeToString())

    def read_tfrecords(self):
        # 读取tfrecords文件
        # 1.构造文件名队列
        file_queue = tf.train.string_input_producer(['cifar10.tfrecords'])
        # 2.读取与解码
        # 读取
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)
        # 解析example
        feature = tf.parse_single_example(value, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64)
        })
        image = feature["image"]
        label = feature["label"]
        # 解码
        image_decoded = tf.decode_raw(image, tf.uint8)
        # 图像形状调整
        image_reshaped = tf.reshape(image_decoded, [self.height, self.width, self.channels])
        # 3.构造批处理队列
        image_batch, label_batch = tf.train.batch([image_reshaped, label], batch_size=100, num_threads=2, capacity=100)
        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            image_value, label_value = sess.run([image_batch, label_batch])
            print(image_value, label_value)
            # 回收资源
            coord.request_stop()
            coord.join(threads)
     

if __name__ == "__main__":
    cifar = Cifar()
    # 写入
    image_value, label_value = cifar.read_binary()
    cifar.write_to_tfrecords(image_value, label_value) 
    # 读取
    cifar.read_tfrecords()
```

