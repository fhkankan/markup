# qrcode模块

**by: GanZiQim**

------

功能描述：[qrcode](https://github.com/lincolnloop/python-qrcode)模块是Github上的一个开源项目，提供了生成二维码的接口。qrcode默认使用PIL库用于生成图像。

------

## 预装pillow

This module uses image libraries, Python Imaging Library (PIL) by default, to generate QR Codes.

It is recommended to use the [pillow](https://pypi.python.org/pypi/Pillow) fork rather than PIL itself.

```
pip install pillow
```

------

## 函数形式生成二维码

qrcode.make(data)

make函数返回一个qrcode.image.pil.PilImage对象。该对象不像PIL.Image.Image一样可以直接调用show函数直接显示，但可以通过调用get_image函数返回一个PIL.Image.Image对象，再在Image对象上面进行操作。也可以直接调用save函数将二维码保存到本地。

```
import qrcode
img = qrcode.make("hello world!")
img.get_image().show()
img.save('hello.png')1234
```

------

## 类实例形式生成二维码

QRCode(version=None, error_correction=constants.ERROR_CORRECT_M, box_size=10, border=4, image_factory=None)

QRCode是qrcode模块的一个内置类。QRCode的实例使用QRCode.add_data(data)函数添加数据。使用QRCode.make(fit=True)函数生成图片。使用QRCode.make_image(image_factory=None)函数得到Image对象。

1. version参数为一个取值范围1-40的整数（或字符串），用于控制二维码的尺寸。最小的尺寸1是一个21格*21格的矩阵。该值设置为None（默认），并且调用make函数时fit参数为True（默认）时，模块会自己决定生成二维码的尺寸。
2. error_correction参数用于控制二维码的错误纠正程度。可以取以下四个保存在模块中的常量：
   - ERROR_CORRECT_L：大约7%或者更少的错误会被更正。
   - ERROR_CORRECT_M：默认值，大约15%或者更少的错误会被更正。
   - ERROR_CORRECT_Q：大约25%或者更少的错误会被更正。
   - ERROR_CORRECT_H：大约30%或者更少的错误会被更正。
3. box_size参数控制二维码中每个格子的像素数，默认为10。
4. border参数控制边框（二维码四周留白）包含的格子数（默认为4，是标准规定的最小值）。
5. image_factory参数是一个继承于qrcode.image.base.BaseImage的类，用于控制make_image函数返回的图像实例。image_factory参数可以选择的类保存在模块根目录的image文件夹下。image文件夹里面有五个.py文件，其中一个为__init__.py，一个为base.py。还有pil.py提供了默认的qrcode.image.pil.PilImage类。pure.py提供了qrcode.image.pure.PymagingImage类。svg.py提供了SvgFragmentImage、SvgImage和SvgPathImage三个类。

**注：实际上make函数也是通过实例化一个QRCode对象来生成二维码的。调用make的时候也可以传入初始化参数。**

**QRCode.make_image函数可以通过改变fill_color和back_color参数改变生成图片的背景颜色和格子颜色。**

```
import qrcode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data('Some data')
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save('qrcode.png')123456789101112
```

------

## 生成SVG格式的二维码

qrcode可以生成三种不同的svg图像，一种是用路径表示的svg，一种是用矩形集合表示的完整svg文件，还有一种是用矩形集合表示的svg片段。第一种用路径表示的svg其实就是矢量图，可以在图像放大的时候保持图片质量，而另外两种可能会在格子之间出现空隙。*具体请用搜素引擎搜“矢量图与位图的区别。*

这三种分别对应了svg.py中的SvgPathImage、SvgImage和SvgFragmentImage类。在调用qrcode.make函数或者实例化QRCode时当作参数传入就可以了。

另外还有qrcode.image.svg.SvgFillImage和qrcode.img.svg.SvgPathFillImage。分别继承自SvgImage和SvgPathImage。这两个并没有其他改变，只不过是默认把背景颜色设置为白色而已。

```
import qrcode
import qrcode.image.svg

if method == 'basic':
    # Simple factory, just a set of rects.
    factory = qrcode.image.svg.SvgImage
elif method == 'fragment':
    # Fragment factory (also just a set of rects)
    factory = qrcode.image.svg.SvgFragmentImage
else:
    # Combined path factory, fixes white space that may occur when zooming
    factory = qrcode.image.svg.SvgPathImage

img = qrcode.make('Some data here', image_factory=factory)1234567891011121314
```

**注：Python2.6版本的xml.etree.ElementTree无法用于生成SVG图像，需要安装lxml才可以完成相应功能。**

------

## Pure Python PNG

通过`pip install git+git://github.com/ojii/pymaging.git#egg=pymaging`和`pip install git+git://github.com/ojii/pymaging-png.git#egg=pymaging-png`安装pymaging相关模块之后，就可以通过给image_factory参数传入qrcode.image.pure.PymagingImage来生成PNG图片了。

```
import qrcode
from qrcode.image.pure import PymagingImage
img = qrcode.make('Some data here', image_factory=PymagingImage)123
```

------

## 命令行模式

qrcode还提供了通过命令行生成二维码的程序。在命令行中使用`qr [--factory] [--error-correction]`命令即可。

```
qr "Some text" > test1.png

qr --factory=svg-path "Some text" > test2.svg
qr --factory=svg "Some text" > test3.svg
qr --factory=svg-fragment "Some text" > test4.svg

qr --factory=pymaging "Some text" > test5.png1234567
```

------

## 在二维码中放入LOGO

[思凡念真的博客](http://www.cnblogs.com/sfnz/p/5457862.html)上还介绍了一种给二维码加上logo的方法，这里我将代码摘下来并加上注释，供读者参考。

```
from PIL import Image
import qrcode

# 初步生成二维码图像
qr = qrcode.QRCode(version=5,error_correction=qrcode.constants.ERROR_CORRECT_H,box_size=8,border=4)
qr.add_data("http://www.cnblogs.com/sfnz/")
qr.make(fit=True)

# 获得Image实例并把颜色模式转换为RGBA
img = qr.make_image()
img = img.convert("RGBA")

# 打开logo文件
icon = Image.open("D:/favicon.jpg")

# 计算logo的尺寸
img_w,img_h = img.size
factor = 4
size_w = int(img_w / factor)
size_h = int(img_h / factor)

# 比较并重新设置logo文件的尺寸
icon_w,icon_h = icon.size
if icon_w >size_w:
    icon_w = size_w
if icon_h > size_h:
    icon_h = size_h
icon = icon.resize((icon_w,icon_h),Image.ANTIALIAS)

# 计算logo的位置，并复制到二维码图像中
w = int((img_w - icon_w)/2)
h = int((img_h - icon_h)/2)
icon = icon.convert("RGBA")
img.paste(icon,(w,h),icon)

# 保存二维码
img.save('D:/createlogo.jpg')12345678910111213141516171819202122232425262728293031323334353637
```

------

## 总结

qrcode模块提供了生成各种二维码图像的接口，还提供了通过命令行生成二维码的方法，使用起来非常方便。不过模块内部还有很多细节无法一一细讲，还是那句话，希望有能力有精力的读者可以去阅读一下源码。

------

**参考资料：**

以上内容主要参考[qrcode官方文档及源码](https://github.com/lincolnloop/python-qrcode) 
放入logo部分参考[思凡念真的博客](http://www.cnblogs.com/sfnz/p/5457862.html)



