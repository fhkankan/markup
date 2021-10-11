# imghdr

推测图像类型

| 值       | 图像格式                      |
| :------- | :---------------------------- |
| `'rgb'`  | SGI 图像库文件                |
| `'gif'`  | GIF 87a 和 89a 文件           |
| `'pbm'`  | 便携式位图文件                |
| `'pgm'`  | 便携式灰度图文件              |
| `'ppm'`  | 便携式像素表文件              |
| `'tiff'` | TIFF 文件                     |
| `'rast'` | Sun 光栅文件                  |
| `'xbm'`  | X 位图文件                    |
| `'jpeg'` | JFIF 或 Exif 格式的 JPEG 数据 |
| `'bmp'`  | BMP 文件                      |
| `'png'`  | 便携式网络图像                |
| `'webp'` | WebP 文件                     |
| `'exr'`  | OpenEXR 文件                  |

- 使用

函数

```python
imghr.what(file, h=None)

# 测试包含在名为 file 的文件中的图像数据，并返回描述该图像类型的字符串。 如果提供了可选的 h，则 file 参数会被忽略并且 h 会被视为包含要测试的字节流。
```

样例

```python
import imghdr


# 判断路径下文件
res = imghdr.what('./demo.png')

# 判断文件流
res = imghdr.what('', img.body)   
```

