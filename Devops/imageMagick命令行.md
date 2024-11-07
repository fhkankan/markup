# ImageMagick

[官网](https://imagemagick.org)

命令行图片处理工具

## 安装

很多系统中可能已经自带

```
# mac
brew install ghostscript  # 依赖
brew install imagemagick
```

## 格式转换

```shell
# 批量转换当前文件夹下所有png文件为jpg，保留原文件
mogrify -format webp *.png
```

