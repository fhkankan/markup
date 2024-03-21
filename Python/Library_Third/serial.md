# serial

串口通信是指外设和计算机间，通过数据信号线 、地线、控制线等，按位进行传输数据的一种通讯方式。这种通信方式使用的数据线少，在远距离通信中可以节约通信成本，但其传输速度比并行传输低。串口是计算机上一种非常通用的设备通信协议。pyserial模块封装了python对串口的访问，为多平台的使用提供了统一的接口。

## 串口通信
串口Uart操作是嵌入式最基础，最简单，也是使用最广范的一种通信协议
串口（serial），全称串行通信接口或串行通讯接口，是一种常用于电子设备间通讯的全双工扩展接口
串行通信：串口通讯的技术基础，指一位一位地按顺序传送数据。其特点是线路简单，只需一对传输线，即可实现双向通信，大大降低成本。适用于远距离通信，但速度较慢；

## 安装使用

- 安装

```
pip install serial
pip install pyserial
```

- 初始化函数

```python
ser = serial.Serial('com3', 115200, timeout=5) 
```

参数属性

| 属性       | 示例                | 含义         |
| ---------- | ------------------- | ------------ |
| `port`     | `port = ‘COM1’`     | 读或者写端口 |
| `baudrate` | `baudrate = 115200` | 波特率       |
| `bytesize` | `bytesize = 8`      | 字节大小     |
| `parity`   | `parity = ‘N’`      | 校验位       |
| `stopbit`  | `stopbits = 1`      | 停止位       |
| `timeout`  | `timeout = None`    | 超时设置     |
| `xonxoff`  | `xonxoff = False`   | 软件流控     |
| `rtscts`   | `rtscts = False`    | 硬件流控     |
| `dsrdtr`   | `dsrdtr = False`    | 硬件流控     |

- 发送接收数据

```python
# 发送数据
success_byres = ser.write(b'This is data for test\r\n')
# 接收数据， 该方法是阻塞的，在没设置超时时间下，不接受到size单位的字符旧一直等待接收，若设置了超时时间，时间未到则一直等待。
data = ser.read(11)
```





- 其他常用方法

| 方法                 | 说明                                     |
| -------------------- | ---------------------------------------- |
| `ser.isOpen()`       | 查看端口是否被打开                       |
| `ser.open()`         | 打开端口                                 |
| `ser.close()`        | 关闭端口                                 |
| `ser.read()`         | 从端口读字节数据。默认1个字节。          |
| `ser.read_all()`     | 从端口接收全部数据。                     |
| `ser.write(“hello”)` | 向端口写数据。                           |
| `ser.readline()`     | 读一行数据。                             |
| `ser.readlines()`    | 读多行数据。                             |
| `in_waiting()`       | 返回接收缓存中的字节数。                 |
| `flush()`            | 等待所有数据写出。                       |
| `flushInput()`       | 丢弃接收缓存中的所有数据。               |
| `flushOutput()`      | 终止当前写操作，并丢弃发送缓存中的数据。 |


​	
​	
​	
​	
​	
​	
​	
​	
​	
​	
​	
​			
​		
​		
​		
​		
​		



```python


# 串口读数据
ser.read()  # 从端口读字节数据，默认1个字节
ser.read_all()  # 从端口接收全部数据
ser.readline()  # 读一行数据
ser.readlines()  # 读多行数据

# 查看端口是否被打开
ser.isOpen()
# 打开端口
ser.open()
# 等待所有数据写出
flush()
# 丢弃接收缓存中的所有数据
flushInput()
# 种植当前写操作，并丢弃发送缓存中的数据
flushOutput()
```

- 属性

```
```

