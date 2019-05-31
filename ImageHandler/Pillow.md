

# Pillow

PIL：Python Imaging Library，已经是Python平台事实上的图像处理标准库了。PIL功能非常强大，但API却非常简单易用。

由于PIL仅支持到Python 2.7，加上年久失修，于是一群志愿者在PIL的基础上创建了兼容的版本，名字叫[Pillow](https://github.com/python-pillow/Pillow)，支持最新Python 3.x，又加入了许多新特性，因此，我们可以直接安装使用Pillow。

要详细了解PIL的强大功能，请请参考Pillow官方文档：

<https://pillow.readthedocs.org/>

## 安装Pillow

如果安装了Anaconda，Pillow就已经可用了。否则，需要在命令行下通过pip安装：

```
$ pip install pillow
```

如果遇到`Permission denied`安装失败，请加上`sudo`重试。

## 相关概念

### 颜色与RGBA值

计算机通常将图像表示为RGB值，或者再加上alpha值（通透度，透明度），称为RGBA值。在Pillow中，RGBA的值表示为由4个整数组成的元组，分别是R、G、B、A。整数的范围0~255。RGB全0就可以表示黑色，全255代表黑色。可以猜测(255, 0, 0, 255)代表红色，因为R分量最大，G、B分量为0，所以呈现出来是红色。但是当alpha值为0时，无论是什么颜色，该颜色都不可见，可以理解为透明。

```python
from PIL import ImageColor
print(ImageColor.getcolor('red', 'RGBA'))
# 也可以只以RBG的方式查看
print(ImageColor.getcolor('black', 'RGB'))
(255, 0, 0, 255)
(0, 0, 0)
```

### 图像的坐标表示

图像中**左上角**是坐标原点(0, 0)，这和平常数学里的坐标系不太一样。这样定义的坐标系意味着，X轴是从左到右增长的，而Y轴是从上到下增长。

在Pillow中如何使用上述定义的坐标系表示一块矩形区域？许多函数或方法要求提供一个矩形元组参数。元组参数包含四个值，分别代表矩形四条边的距离X轴或者Y轴的距离。顺序是`(左，顶，右，底)`。右和底坐标稍微特殊，表示直到但不包括。可以理解为`[左, 右)`和`[顶， 底)`这样左闭右开的区间。比如(3, 2, 8, 9)就表示了横坐标范围[3, 7]；纵坐标范围[2, 8]的矩形区域。

## 操作图像

[参考](https://www.cnblogs.com/sun-haiyu/p/7127582.html)

[参考](https://www.cnblogs.com/bigmonkey/p/7352094.html)

### 读取

很多图像处理库（如opencv）都以`imread()`读取图片。Pillow中使用`open`方法。

```python
from PIL import Image

im_path = r'F:\Jupyter Notebook\csv_time_datetime_PIL\rabbit.jpg'
im = Image.open(im_path)
width, height = im.size
# 宽高
print(im.size, width, height)
# 格式，以及格式的详细描述
print(im.format, im.format_description)

im.save(r'C:\Users\Administrator\Desktop\rabbit_copy.jpg')
im.show()
```

方法属性

```python
im.size  # 返回一个元组，分别是宽和高。
show()  # 会调用系统默认图像查看软件，打开并显示。
im.format	# 可查看图像的格式。
save()	# 可保存处理后的图片，如果未经处理，保存后的图像占用的空间(字节数)一般也与原图像不一样，可能经过了压缩。
```

### 色彩

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('girl0.png')
model = img.convert('L')
plt.figure("girl")
#the argument comp is Colormap
plt.imshow(model, cmap='pink')
plt.show()
```

 其中img.convert指定一种色彩模式：

```
- 1 (1-bit pixels, black and white, stored with one pixel per byte)
- L (8-bit pixels, black and white)
- P (8-bit pixels, mapped to any other mode using a colour palette)
- RGB (3x8-bit pixels, true colour)
- RGBA (4x8-bit pixels, true colour with transparency mask)
- CMYK (4x8-bit pixels, colour separation)
- YCbCr (3x8-bit pixels, colour video format)
- I (32-bit signed integer pixels)
- F (32-bit floating point pixels)
```

### 分离rgba

rgb指红绿蓝光色三原色，a指alpha通道，一般用作不透明度参数

需要注意的是，并非所有图片都有alpha通道，此时 img.split()仅能返回r,g,b

```python
img = Image.open('girl0.png')
# 分离rgba
r, g, b, a = img.split()  
plt.figure("girl0")
plt.imshow(r)
plt.show()
```

- 显示多个图片

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('girl0.png')
gray = img.convert('L')
# 分离rgba
r, g, b, a = img.split()  
plt.figure("girl")

def setPlot(num, title):
    #subplot(nrows, ncols, plot_number)
    #图表的整个绘图区域被等分为numRows行和numCols列，然后按照从左到右、从上到下的顺序对每个区域进行编号，左上区域的编号为1
    plt.subplot(2, 3, num)
    plt.title(title)
    plt.axis('off')
    
setPlot(1, 'origin')
plt.imshow(img)

setPlot(2, 'gray')
plt.imshow(gray, cmap='gray')

setPlot(3, 'rgba')
# 合并rgba
plt.imshow(Image.merge('RGBA', (r, g, b, a)))

setPlot(4, 'r')
plt.imshow(r)
  
setPlot(5, 'g')
plt.imshow(g)

setPlot(6, 'b')
plt.imshow(b)
```

### 二值化处理

图片由像素组成，每个像素对应着rgb值，整个图片可以看成一个矩阵。我们将大于128的像素点转换为1，其它转换为0

```python
from PIL import Image
import matplotlib.pyplot as plt

#二值化处理
img = Image.open('girl0.png')
gray = img.convert('L')

WHITE, BLACK = 1, 0
img_new = gray.point(lambda x: WHITE if x > 128 else BLACK)
plt.imshow(img_new, cmap='gray')
plt.show()
```

- 压缩

如果图片大小不一，不利于下一步工作，在此需要将图片压缩成统一大小，对于手写数字，可将其压缩为32*32

```python
#等比例压缩图片
#参考 http://fc-lamp.blog.163.com/blog/static/174566687201282424018946/
def resizeImg(**args):
    #dst_w,dst_h  目标图片大小,  save_q  图片质量
    args_key = {'ori_img':'', 'dst_img':'', 'dst_w':'', 'dst_h':'', 'save_q':75}
    arg = {}
    for key in args_key:
        if key in args:
            arg[key] = args[key]
        
    im = Image.open(arg['ori_img'])
    ori_w, ori_h = im.size
    widthRatio = heightRatio = None
    ratio = 1
    if (ori_w and ori_w > arg['dst_w']) or (ori_h and ori_h > arg['dst_h']):
        if arg['dst_w'] and ori_w > arg['dst_w']:
            widthRatio = float(arg['dst_w']) / ori_w
        if arg['dst_h'] and ori_h > arg['dst_h']:
            heightRatio = float(arg['dst_h']) / ori_h

        if widthRatio and heightRatio:
            if widthRatio < heightRatio:
                ratio = widthRatio
            else:
                ratio = heightRatio

        if widthRatio and not heightRatio:
            ratio = widthRatio
        if heightRatio and not widthRatio:
            ratio = heightRatio
            
        newWidth = int(ori_w * ratio)
        newHeight = int(ori_h * ratio)
    else:
        newWidth = ori_w
        newHeight = ori_h
    
    im.resize((newWidth, newHeight), Image.ANTIALIAS).save(arg['dst_img'], quality=arg['save_q'])
```

- 打印

```python
resizeImg(ori_img='7.jpg', dst_img='7_1.jpg', dst_w=32, dst_h=32, save_q=60)

#二值化处理
img = Image.open('7_1.jpg')
gray = img.convert('L')

WHITE, BLACK = 1, 0
img_new = gray.point(lambda x: WHITE if x > 128 else BLACK)
arr = nmp.array(img_new)

for i in range(arr.shape[0]):
    print(arr[i].flatten())
```

### 新建

Pillow也可以新建空白图像, 第一个参数是mode即颜色空间模式，第二个参数指定了图像的分辨率(宽x高)，第三个参数是颜色。

- 可以直接填入常用颜色的名称。如'red'
- 也可以填入十六进制表示的颜色，如`#FF0000`表示红色。
- 还能传入元组，比如(255, 0, 0, 255)或者(255， 0， 0)表示红色。

```python
# 通常使用RGB模式就可以了
newIm= Image.new('RGB', (100, 100), 'red')
newIm.save(r'C:\Users\Administrator\Desktop\1.png')

# 也可以用RGBA模式，还有其他模式查文档吧
blcakIm = Image.new('RGB',(200, 100), 'red')
blcakIm.save(r'C:\Users\Administrator\Desktop\2.png')
# 十六进制颜色
blcakIm = Image.new('RGBA',(200, 100), '#FF0000')
blcakIm.save(r'C:\Users\Administrator\Desktop\3.png')
# 传入元组形式的RGBA值或者RGB值
# 在RGB模式下，第四个参数失效，默认255，在RGBA模式下，也可只传入前三个值，A值默认255
blcakIm = Image.new('RGB',(200, 100), (255, 255, 0, 120))
blcakIm.save(r'C:\Users\Administrator\Desktop\4.png')
```

### 裁剪

`Image`有个`crop()`方法接收一个矩形区域元组(上面有提到)。返回一个新的Image对象，是裁剪后的图像，对原图没有影响。

```python
im = Image.open(im_path)
cropedIm = im.crop((700, 100, 1200, 1000))
cropedIm.save(r'C:\Users\Administrator\Desktop\cropped.png')
```

### 复制粘贴

`Image`的`copy`函数如其名会产生一个原图像的副本，在这个副本上的任何操作不会影响到原图像。`paste()`方法用于将一个图像粘贴（覆盖）在另一个图像上面。谁调用它，他就在该Image对象上直接作修改。

```python
im = Image.open(im_path)
cropedIm = im.crop((700, 100, 1200, 1000))
im.paste(cropedIm, (0, 0))
im.show()
im.save(r'C:\Users\Administrator\Desktop\paste.png')
```

`im.show()`显示图像发现这时im（即原图）已经被改变。

这如果之后还会用到原图的信息，由于信息被改变就很麻烦。所以paste前最好使用`copy()`复制一个副本，在此副本操作，不会影响到原图信息。虽然在程序里原图信息已改变，但由于保存文件时用的其他文件名，相当于改变没有生效，所以查看的时候原图还是没有改变的。

```python
im = Image.open(im_path)
cropedIm = im.crop((700, 100, 1200, 1000))
copyIm = im.copy()
copyIm.paste(cropedIm, (0, 0))
im.show()
copyIm.save(r'C:\Users\Administrator\Desktop\paste.png')
```

这回再看原图，没有改变了。这就保证了之后再次使用im时，里面的信息还是原汁原味。来看个有趣的例子。

```python
im = Image.open(im_path)
cropedIm = im.crop((700, 100, 1200, 1000))

crop_width, crop_height = cropedIm.size
width, height = im.size

copyIm = im.copy()
for left in range(0, width, crop_width):
    for top in range(0, height, crop_height):
        copyIm.paste(cropedIm, (left, top))

copyIm.save(r'C:\Users\Administrator\Desktop\dupli-rabbit.png')
```

以裁剪后的图像宽度和高度为间隔，在循环内不断粘贴在副本中，这有点像是在拍证件照。

### 调整大小

`resize`方法返回指定宽高度的新Image对象，接受一个含有宽高的元组作为参数。**宽高的值得是整数。**不是等比例缩放

```python
im = Image.open(im_path)
width, height = im.size
resizedIm = im.resize((width, height+(1920-1080)))
resizedIm.save(r'C:\Users\Administrator\Desktop\resize.png')
```

### 旋转翻转

`rotate()`返回旋转后的新Image对象, 保持原图像不变。逆时针旋转。

```python
im = Image.open(im_path)
im.rotate(90).save(r'C:\Users\Administrator\Desktop\rotate90.png')
im.rotate(270).save(r'C:\Users\Administrator\Desktop\rotate270.png')
im.rotate(180).save(r'C:\Users\Administrator\Desktop\rotate180.png')
im.rotate(20).save(r'C:\Users\Administrator\Desktop\rotate20.png')
im.rotate(20, expand=True).save(r'C:\Users\Administrator\Desktop\rotate20_expand.png')
```

由上到下，分别是旋转了90°，180°， 270°、普通的20°，加了参数`expand=True`旋转的20°。expand放大了图像尺寸（变成了2174x1672），使得边角的图像不被裁剪（四个角刚好贴着图像边缘）。再看旋转90°、270°时候图像被裁剪了，但是如下查看图像的宽高，确是和原图一样，搞不懂。

```python
im90 = Image.open(r'C:\Users\Administrator\Desktop\rotate90.png')
im270 = Image.open(r'C:\Users\Administrator\Desktop\rotate270.png')
# 宽高信息并没有改变
print(im90.size, im270.size)
(1920, 1080) (1920, 1080)
```

图像的镜面翻转。`transpose()`函数可以实现，必须传入`Image.FLIP_LEFT_RIGHT`或者`Image.FLIP_TOP_BOTTOM`，第一个是水平翻转，第二个是垂直翻转。

```python
im = Image.open(im_path)
im.transpose(Image.FLIP_LEFT_RIGHT).save(r'C:\Users\Administrator\Desktop\transepose_lr.png')
im.transpose(Image.FLIP_TOP_BOTTOM).save(r'C:\Users\Administrator\Desktop\transepose_tb.png')
```

### 缩放

来看看最常见的图像缩放操作

```python
from PIL import Image

# 打开一个jpg图像文件，注意是当前路径:
im = Image.open('test.jpg')
# 获得图像尺寸:
w, h = im.size
print('Original image size: %sx%s' % (w, h))
# 缩放到50%:
im.thumbnail((w//2, h//2))
print('Resize image to: %sx%s' % (w//2, h//2))
# 把缩放后的图像用jpeg格式保存:
im.save('thumbnail.jpg', 'jpeg')
```

### 过滤

Pillow使用ImageFilter可以简单做到图像的模糊、边缘增强、锐利、平滑等常见操作

```python
from PIL import Image, ImageFilter

im = Image.open(im_path)
# 高斯模糊
im.filter(ImageFilter.GaussianBlur).save(r'C:\Users\Administrator\Desktop\GaussianBlur.jpg')
# 普通模糊
im.filter(ImageFilter.BLUR).save(r'C:\Users\Administrator\Desktop\BLUR.jpg')
# 边缘增强
im.filter(ImageFilter.EDGE_ENHANCE).save(r'C:\Users\Administrator\Desktop\EDGE_ENHANCE.jpg')
# 找到边缘
im.filter(ImageFilter.FIND_EDGES).save(r'C:\Users\Administrator\Desktop\FIND_EDGES.jpg')
# 浮雕
im.filter(ImageFilter.EMBOSS).save(r'C:\Users\Administrator\Desktop\EMBOSS.jpg')
# 轮廓
im.filter(ImageFilter.CONTOUR).save(r'C:\Users\Administrator\Desktop\CONTOUR.jpg')
# 锐化
im.filter(ImageFilter.SHARPEN).save(r'C:\Users\Administrator\Desktop\SHARPEN.jpg')
# 平滑
im.filter(ImageFilter.SMOOTH).save(r'C:\Users\Administrator\Desktop\SMOOTH.jpg')
# 细节
im.filter(ImageFilter.DETAIL).save(r'C:\Users\Administrator\Desktop\DETAIL.jpg')
```

### 绘图

PIL的`ImageDraw`提供了一系列绘图方法，让我们可以直接绘图。比如要生成字母验证码图片：

```python
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random

# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))

# 随机颜色1:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

# 240 x 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象:
font = ImageFont.truetype('Arial.ttf', 36)
# 创建Draw对象:
draw = ImageDraw.Draw(image)
# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
# 模糊:
image = image.filter(ImageFilter.BLUR)
image.save('code.jpg', 'jpeg')
```

我们用随机颜色填充背景，再画上文字，最后对图像进行模糊，得到验证码图片如下：

![验证码](https://cdn.liaoxuefeng.com/cdn/files/attachments/0014076720724832de067ce843d41c58f2af067d1e0720f000)

如果运行的时候报错：

```
IOError: cannot open resource
```

这是因为PIL无法定位到字体文件的位置，可以根据操作系统提供绝对路径，比如：

```
'/Library/Fonts/Arial.ttf'
```

### 小结

PIL提供了操作图像的强大功能，可以通过简单的代码完成复杂的图像处理。