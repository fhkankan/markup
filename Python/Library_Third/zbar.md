# zbar

用于解析二维码，可用于图片，视频，传感器，常用的还有zbarlight（支持Python2与python3），zxing(需要java插件)等

## 安装

[安装文件](https://pypi.python.org/pypi/zbar/0.10)

只支持python2.5~2.6

windows只支持32位,64位python最后会出现"**ImportError: DLL load failed: %1 不是有效的 Win32 应用程序。**"错误，暂时无解. 解决途径:[ZBarWin64](https://link.jianshu.com/?t=https://github.com/NaturalHistoryMuseum/ZBarWin64)

## 简单图片扫描

```python
# -*- coding:utf-8 -*-
import zbar
from PIL import Image
  
# 创建图片扫描对象
scanner = zbar.ImageScanner()
# 设置对象属性
scanner.parse_config('enable')  
# 打开含有二维码的图片
img = Image.open('<你的图片路径>').convert('L')
#获取图片的尺寸
width, height = img.size  
#建立zbar图片对象并扫描转换为字节信息
qrCode = zbar.Image(width, height, 'Y800', img.tobytes())
scanner.scan(qrCode)  
data = ''
for s in qrCode:
    data += s.data  
# 删除图片对象
del img  
# 输出解码结果
print data
```

代码二

```python
#!/usr/bin/env python  
# coding: u8  
  
import zbar  
import Image  
  
# create a reader  
scanner = zbar.ImageScanner()  
  
# configure the reader  
scanner.parse_config('enable')  
  
# obtain image data  
pil = Image.open('./55.jpg').convert('L')  
width, height = pil.size  
#pil.show()  
raw = pil.tostring()  
  
# wrap image data  
image = zbar.Image(width, height, 'Y800', raw)  
  
# scan the image for barcodes  
scanner.scan(image)  
  
# extract results  
for symbol in image:  
    # do something useful with results  
    print symbol.type, '图片内容为:\n%s' % symbol.data  
  
# clean up  
del(image)  
```

## 视频检测

```python
#!/usr/bin/python  
from sys import argv  
import zebra  
  
# create a Processor  
proc = zbar.Processor()  
  
# configure the Processor  
proc.parse_config('enable')  
  
# initialize the Processor  
device = '/dev/video0'  
if len(argv) > 1:  
    device = argv[1]  
proc.init(device)  
  
# setup a callback  
def my_handler(proc, image, closure):  
    # extract results  
    for symbol in image:  
        if not symbol.count:  
            # do something useful with results  
            print 'decoded', symbol.type, 'symbol', '"%s"' % symbol.data  
  
proc.set_data_handler(my_handler)  
  
# enable the preview window  
proc.visible = True  
  
# initiate scanning  
proc.active = True  
try:  
    proc.user_wait()  
except zbar.WindowClosed:  
    pass  
```

使用wxpython做成一个小软件，wxpython默认的是unicode编码，使用qrcode进行生成二维码。经常将中文解析为乱码。解码部分做了如下修改，基本上能解析所有二维码。

```
try:   
                utf8Data = symbol.data.decode("gbk")   
            except UnicodeDecodeError:   
                try:  
                    utf8Data = symbol.data.decode("utf-8").encode("gbk")  
                except:  
                    utf8Data=symbol.data.decode('utf-8').encode('sjis').decode('utf-8')   
```









