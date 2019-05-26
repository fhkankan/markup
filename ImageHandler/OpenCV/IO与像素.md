# IO与像素

## 安装使用

```shell
# 安装
pip install python-opencv
# 使用
import cv2 as cv
```

## I/O处理

### 图像

- 载入

```python
cv.imread(path, [desc])
# 将图片载入到内存中
# 返回ndarry类型的图像数据
# 参数
path	图片路径
desc	图片描述
			cv.IMREAD_UNCHANGED = -1
  		cv.IMREAD_GRAYSACLE = 0
    	cv.IMREAD_COLOR = 1
      cv.IMREAD_ANYDEPTH = 2
      cv.IMREAD_ANYCOLOR = 4
			cv.IMREAD_LOAD_GDAL = 8
```

- 显示

```python
cv.imshow(name, img)
# 在名字为name的窗口中显示img图像
```

- 保存

```python
cv.iwrite(path, img,)
# 将图像img写入到硬盘的path路径中
```
-  暂停
```python
cv.waitKey(0) 
# 暂停
# 返回键值
```

示例

```python
import cv2 as cv
import numpy as np

# 读取载入图片
img = cv.imread("./cat.jpg")
# 创建图片
emptyImage = np.zeros(img.shape, np.uint8)
# 复制图片
emptyImage2 = img.copy()
# 获得原图片灰度化后的图片 
emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#emptyImage3[...]=0  # 将其转成空白的黑色图像
 
cv.imshow("EmptyImage", emptyImage)  # 展示图像
cv.namedWindow("Image")  # 新建重命名的窗口
cv.imshow("Image", img)  # 展示图像
cv.imshow("EmptyImage2", emptyImage2)
cv.imshow("EmptyImage3", emptyImage3)
cv.imwrite("./cat1.jpg", img)  # 保存图像，参数1：路径，参数2：图像数据
cv.imwrite("./cat2.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 5])  # 参数3：图像质量0~100，默认5
cv.imwrite("./cat3.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
cv.imwrite("./cat.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  # 参数3：压缩级别0~9， 默认3
cv.imwrite("./cat2.png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
cv.waitKey (0)
cv.destroyAllWindows()  # 销毁所有窗口
```

### 视频

## 像素处理

### 图像表示

图像在opencv-python中以ndarry表示

```python
img = numpy.zeros([3, 3], dtype=numpy.unit8)
# 每个像素都由一个8位证书来表示，值范围0～255
img = arrray([[0, 0, 0],
							[0, 0, 0],
							[0, 0, 0]], dtype=uint8)
							
# 转换为BGR格式
img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
# 每个像素都由一个三元数组表示，每个整型向量分别表示一个B、G和R通道
img = arrray([[[0, 0, 0],
							 [0, 0, 0],
							 [0, 0, 0]], 
							[[0, 0, 0],
							 [0, 0, 0],
							 [0, 0, 0]],
              [[0, 0, 0],
							 [0, 0, 0],
							 [0, 0, 0]], dtype=uint8)
```

属性

```python
img.shape  # 图像高、宽、通道数组成的元组,若图像是单色或灰度的，则不包含通道
img.size  # 图像像素的大小
img.dtype  # 图像的数据类型(无符号整数类型的变量和该类型占的位数，如uint8)
```

###  像素操作
读取与设置
```python
# 方法一：
# 灰度图
img[j, i] = 255  # j,i分别表示图像的行和列

# BGR图像
img[j,i,0]= 255  # 0通道
img[j,i,1]= 255  # 1通道
img[j,i,2]= 255  # 2通道
# 或
img[j, i] = [255, 255, 255]

# 方法二
img.item(j, i, 0)  # 读取0通道
img.itemset((j, i, 0), 255)  # 设置
```
像素取反

```python
# 方法一：
def access_pixels(image):
    # height = image.shape[0]
    # width = image.shape[1]
    # channels = image.shape[2]
    height, width, channels = img.shape
    print("width : %s, height : %s channels : %s"%(width, height, channels))
    # 像素遍历取反，速度较慢
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("pixels_demo", image
              
# 方法二：
def inverse(image):
    """像素取反内部api，速度快"""
    dst = cv.bitwise_not(image)
    cv.imshow("inverse demo", dst
```

创建图像

```python
def create_image():
		# numpy操作  
    m1 = np.ones([3, 3], np.uint8)
    m1.fill(12222.388)  # 溢出截断
    print(m1)

    m2 = m1.reshape([1, 9])  # 调整数组维度
    print(m2)

    m3 = np.array([[2,3,4], [4,5,6],[7,8,9]], np.int32)  # 直接指定值
    #m3.fill(9)
    print(m3)

    # 创建图片
    img = np.zeros([400, 400, 3], np.uint8)
    #img[: , : , 0] = np.ones([400, 400])*255
    img[:, :, 2] = np.ones([400, 400]) * 255
    cv.imshow("new image", img)

    img = np.ones([400, 400, 1], np.uint8)
    img = img * 0
    cv.imshow("new image", img)
    cv.imwrite("D:/myImage.png", img)
```

人工添加椒盐

```python
import cv2
import numpy as np
 
def salt(img, n):
	for k in range(n):
        # np.random.random()随机数生成，比python中的random()拥有更多方法，但是非线程安全；若需要使用多线程，使用python自带的random()或构建一个本地的np.random.Random类的实例
		i = int(np.random.random() * img.shape[1]);
		j = int(np.random.random() * img.shape[0]);
		if img.ndim == 2:  # 灰度图像 
			img[j,i] = 255
		elif img.ndim == 3:  # BGR图像 
			img[j,i,0]= 255
			img[j,i,1]= 255
			img[j,i,2]= 255
	return img
 
if __name__ == '__main__':
	img = cv2.imread("图像路径")
	saltImage = salt(img, 500)
	cv2.imshow("Salt", saltImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
```
### 像素运算

- 算数运算

```python
# 相加
dst = cv.add(m1, m2) 
# 相减  
dst = cv.subtract(m1, m2)
# 相乘
dst = cv.multiply(m1, m2)
# 相除
dst = cv.divide(m1, m2)  
```

- 逻辑运算

```python
# 与
dst = cv.bitwise_and(m1, m2)
# 或
dst = cv.bitwise_or(m1, m2)  
# 非
dst = cv.bitwise_not(image)
```

增加亮度和对比度

```python
def contrast_brightness_demo(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)  # 增加对比度和亮度
    cv.imshow("con-bri-demo", dst)
    
contrast_brightness_demo(src, 1.5, 0)
```

对比度表征

```python
# 均值，表示了整体的亮度
# 方差，表征了对比度
M1, dev1 = cv.meanStdDev(m1)  
M2, dev2 = cv.meanStdDev(m2)
h, w = m1.shape[:2]
print(M1)
print(M2)
print(dev1)
print(dev2)

img = np.zeros([h, w], np.uint8)
m, dev = cv.meanStdDev(img)
print(m)
print(dev)
```



### 通道分离

可以使用OpenCV自带的split函数(推荐)，也可以直接操作numpy数组来分离通道
```python
import cv2
import numpy as np
 
img = cv2.imread("D:/cat.jpg")
# opencv
b, g, r = cv2.split(img)  # 三个通道同时分离
b = cv2.split(img)[0]  # b单通道
g = cv2.split(img)[1]  # g单通道
r = cv2.split(img)[2]  # r单通道
# numpy
b = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
g = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
r = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
b[:,:] = img[:,:,0]
g[:,:] = img[:,:,1]
r[:,:] = img[:,:,2]

cv2.imshow("Blue", r)
cv2.imshow("Red", g)
cv2.imshow("Green", b)
cv2.waitKey(0)
```
### 通道合并

通道合并也有两种方法，实际使用时请用OpenCV自带的merge函数！用NumPy组合的结果不能在OpenCV中其他函数使用，因为其组合方式与OpenCV自带的不一样
```python
import cv2
import numpy as np
 
img = cv2.imread("D:/cat.jpg")
 
b, g, r = cv2.split(img)  # 三个通道同时分离
# opencv
merged = cv2.merge([b,g,r])
print merged.strides  # 表示的是在每个维数上以字节计算的步长
# numpy
mergedByNp = np.dstack([b,g,r]) 
print mergedByNp.strides
 
cv2.imshow("Merged", merged)
cv2.imshow("MergedByNp", merged)
cv2.imshow("Blue", b)
cv2.imshow("Red", r)
cv2.imshow("Green", g)
cv2.waitKey(0)
```
