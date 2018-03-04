# pypdf2

官方文档：https://pythonhosted.org/PyPDF2/

```
作为PDF工具包构建的纯Python库。它能够：
	提取文档信息（标题，作者...）
	逐页分割文档
	合并文件逐页
	裁剪页面
	将多个页面合并成一个页面
	加密和解密PDF文件
	和更多！
通过使用Pure-Python，它可以在任何Python平台上运行，而不依赖于外部库。它也可以完全在StringIO对象而不是文件流上工作，允许在内存中进行PDF操作。因此，它是管理或操纵PDF的网站的有用工具。
```

## 安装

```
pip install pypdf2
```

## 概述

### PdfFileReader

####构造方法：

```
PyPDF2.PdfFileReader(stream,strict = True,warndest = None,overwriteWarnings = True)

初始化一个 PdfFileReader 对象，此操作可能需要一些时间，因为 PDF 流的交叉引用表被读入内存。

参数：
- stream：*File 对象或支持与 File 对象类似的标准读取和查找方法的对象，也可以是表示 PDF 文件路径的字符串。*
- strict（bool）： 确定是否应该警告用户所用的问题，也导致一些可纠正的问题是致命的，默认是 True
- warndest : 记录警告的目标(默认是 sys.stderr)
- overwriteWarnings(bool)：确定是否 warnings.py 用自定义实现覆盖 Python 模块（默认为 True）

```

####属性和方法

```
decrypt(password)
当使用带有PDF标准加密处理程序的加密/安全PDF文件时，此功能将允许对文件进行解密。它会根据文档的用户密码和所有者密码检查给定的密码，然后在密码正确的情况下存储解密密钥。
密码是否匹配并不重要。这两个密码都提供了正确的解密密钥，可以使文档与该库一起使用。
参数：	password (str) - 要匹配的密码。
返回：	0：如果密码失败或者1：密码与用户密码匹配或者2：密码与所有者密码匹配。
返回类型：int
引发NotImplementedError：如果文档使用不受支持的加密方法。

documentInfo
访问该getDocumentInfo()函数的只读属性

getDestinationPageNumber(destination)
检索给定目标对象的页码
参数：	destination (Destination) - 获取页码的目的地。应该是一个实例 Destination
返回：	页码或者-1:如果找不到页面
返回类型：init

getDocumentInfo()
检索PDF文件的文档信息字典（如果存在）。请注意，某些PDF文件使用元数据流而不是docinfo字典，并且这些元数据流将不会被此功能访问。
返回：	该PDF文件的文件信息
返回类型：	DocumentInformation或者None如果不存在。

getFields（tree = None，retval = None，fileobj = None ）
如果此PDF包含交互式表单字段，则提取字段数据。tree和retval的参数是递归使用。
参数：	fileobj - 一个文件对象（通常是一个文本文件），用于在所有找到的交互式表单域上编写报告。
返回：	一个字典，其中每个键是一个字段名称，每个值都是一个Field对象。默认情况下，映射名称用于键。
返回类型：字典，或者None表单数据无法找到。

getFormTextFields（）
从文档中检索带有文本数据的表单域（输入，下拉列表）

getNamedDestinations（tree = None，retval = None ）
检索文档中存在的指定目标。
返回：	一个将名字映射到的字典 Destinations。
返回类型：	字典

getNumPages（）
计算此PDF文件中的页面数。
返回：	页数
返回类型：	INT
引发PdfReadError：	如果文件被加密并且限制阻止了此操作。

getOutlines（node=None, outlines=None）
检索文档中出现的文档大纲。
返回：	一个嵌套列表Destinations。

getPage（pageNumber）
通过此PDF文件中的编号检索页面。
参数：	pageNumber（int） - 要检索的页码（页面从零开始）
返回：	一个PageObject实例。
返回类型：	PageObject

getPageLayout（）
获取页面布局。请参阅有关setPageLayout() 有效布局的说明。
返回：	目前正在使用的页面布局。
返回类型：str，None如果没有指定

getPageMode（）
获取页面模式。请参阅有关setPageMode() 有效模式的说明。
返回：	目前正在使用的页面模式。
返回类型：	str，None如果没有指定

getPageNumber（page）
检索给定PageObject的页码
参数：	page（PageObject） - 获取页码的页面。应该是一个实例PageObject
返回：	页码或-1如果找不到页面
返回类型：	INT

getXmpMetadata（）
从PDF文档根目录中检索XMP（可扩展元数据平台）数据。
返回：	一个XmpInformation 可用于从文档访问XMP元数据的实例。
返回类型：XmpInformation或者 None如果在文档根目录中找不到元数据。

isEncrypted
显示此PDF文件是否加密的只读布尔属性。请注意，如果该属性为true，则即使在decrypt()调用该方法后该属性也将保持为真 。

namedDestinations
访问该getNamedDestinations()函数的只读属性 。

numPages
访问该getNumPages()函数的只读属性 。

outlines
访问的只读属性
getOutlines() 功能。

pageLayout
访问该getPageLayout()方法的只读属性 。

pageMode
访问该getPageMode()方法的只读属性 。

pages
基于getNumPages()和 getPage()方法模拟列表的只读属性 。

xmpMetadata
访问该getXmpMetadata()函数的只读属性 。
```

####PDF 读取操作

```
# encoding:utf-8
from PyPDF2 import PdfFileReader, PdfFileWriter

readFile = 'C:/Users/Administrator/Desktop/RxJava 完全解析.pdf'
# 获取 PdfFileReader 对象
pdfFileReader = PdfFileReader(readFile)  # 或者这个方式：pdfFileReader = PdfFileReader(open(readFile, 'rb'))
# 获取 PDF 文件的文档信息
documentInfo = pdfFileReader.getDocumentInfo()
print('documentInfo = %s' % documentInfo)
# 获取页面布局
pageLayout = pdfFileReader.getPageLayout()
print('pageLayout = %s ' % pageLayout)

# 获取页模式
pageMode = pdfFileReader.getPageMode()
print('pageMode = %s' % pageMode)

xmpMetadata = pdfFileReader.getXmpMetadata()
print('xmpMetadata  = %s ' % xmpMetadata)

# 获取 pdf 文件页数
pageCount = pdfFileReader.getNumPages()

print('pageCount = %s' % pageCount)
for index in range(0, pageCount):
    # 返回指定页编号的 pageObject
    pageObj = pdfFileReader.getPage(index)
    print('index = %d , pageObj = %s' % (index, type(pageObj)))  # <class 'PyPDF2.pdf.PageObject'>
    # 获取 pageObject 在 PDF 文档中处于的页码
    pageNumber = pdfFileReader.getPageNumber(pageObj)
    print('pageNumber = %s ' % pageNumber)
1234567891011121314151617181920212223242526272829303132
```

输出结果:

```
documentInfo = {'/Title': IndirectObject(157, 0), '/Producer': IndirectObject(158, 0), '/Creator': IndirectObject(159, 0), '/CreationDate': IndirectObject(160, 0), '/ModDate': IndirectObject(160, 0), '/Keywords': IndirectObject(161, 0), '/AAPL:Keywords': IndirectObject(162, 0)}
pageLayout = None 
pageMode = None
xmpMetadata  = None 
pageCount = 3
index = 0 , pageObj = <class 'PyPDF2.pdf.PageObject'>
pageNumber = 0 
index = 1 , pageObj = <class 'PyPDF2.pdf.PageObject'>
pageNumber = 1 
index = 2 , pageObj = <class 'PyPDF2.pdf.PageObject'>
pageNumber = 2 1234567891011
```

###PdfFileMerger

####构造方法

```
class PyPDF2.PdfFileMerger(strict=True)

初始化一个PdfFileMerger对象。PdfFileMerger将多个PDF合并为一个PDF。它可以连接，切片，插入或上述的任意组合。

查看功能merge()（或append()）和write()使用信息。
参数：strict (bool) - 确定是否应该警告用户所有问题，并且还会导致一些可纠正的问题。默认为True。
```

#### 属性和方法

````
addBookmark（title，pagenum，parent = None）
为此PDF文件添加书签。
参数：	
title（str） - 用于此书签的标题。
pagenum（int） - 此书签将指向的页码。
parent- 对创建嵌套书签的父书签的引用。

addMetadata（infos)
将自定义元数据添加到输出。
参数：	infos（dict） - 一个Python字典，其中每个键都是一个字段，每个值都是您的新元数据。例：{u'/Title': u'My title'}

addNamedDestination（title，pagenum ）
将输出目标添加到输出中。
参数：	
title (str)  - 要使用的标题
pagenum（int） - 此目的地指向的页码。

append（fileobj，bookmark=None，pages=None，import_bookmarks=True)
与merge()方法相同，但假定您想要将所有页面连接到文件的末尾而不是指定位置。
参数：	
fileobj - 文件对象或支持与文件对象类似的标准读取和查找方法的对象。也可以是表示PDF文件路径的字符串。
bookmark (str) - 您可以选择通过提供书签文本来指定要在包含文件的开头应用的书签。
pages - 可以是 Page Range或元组，以将源文档中指定范围的页面合并到输出文档中。如(start, stop[, step])
import_bookmarks（bool） - 可通过指定为False，来防止源文档的书签被导入。

close（）
关闭所有文件描述符（输入和输出）并清除所有内存使用情况。

merge（position，fileobj，bookmark=None，pages=None，import_bookmarks=True ）
将给定文件中的页面合并到指定页码的输出文件中。
参数：	
position（int） - 插入此文件的页码。文件将在给定的数字后插入。
fileobj - 文件对象或支持与文件对象类似的标准读取和查找方法的对象。也可以是表示PDF文件路径的字符串。
bookmark (str) - 您可以选择通过提供书签文本来指定要在包含文件的开头应用的书签。
pages - 可以是页面范围或元组，以将源文档中指定范围的页面合并到输出文档中。如(start, stop[, step])
import_bookmarks（bool） - 您可以通过指定为False，来防止源文档的书签被导入。

setPageLayout（布局）
设置页面布局
参数：	layout (str) - 要使用的页面布局
有效的布局是：
/NoLayout	显式没有指定布局
/SinglePage	一次显示一页
/OneColumn	一次显示一列
/TwoColumnLeft	以两列显示页面，左边显示奇数页面
/TwoColumnRight 以两列显示页面，右侧显示奇数页面
/TwoPageLeft	一次显示两页，左侧显示奇数页
/TwoPageRight	一次显示两页，右侧显示奇数页

setPageMode（模式）
设置页面模式。
参数：	模式（str） - 要使用的页面模式。
有效的模式是：
/UseNone	不要显示轮廓或缩略图面板
/UseOutlines	显示轮廓（又名书签）面板
/UseThumbs	显示页面缩略图面板
/FullScreen	全屏视图
/UseOC	显示可选内容组（OCG）面板
/UseAttachments	显示附件面板

write（fileobj ）
将已合并的所有数据写入给定的输出文件。
参数：	fileobj - 输出文件。可以是文件名或任何类型的文件类对象。
````

### PageObject

####构造方法

```
class PyPDF2.pdf.PageObject（pdf = None，indirectRef = None ）
此类表示 PDF 文件中的单个页面，通常这个对象是通过访问 PdfFileReader 对象的 getPage() 方法来得到的，也可以使用 createBlankPage() 静态方法创建一个空的页面。

参数：
- pdf : 页面所属的 PDF 文件。
- indirectRef：将源对象的原始间接引用存储在其源 PDF 中。
```

####属性和方法

```
addTransformation（ctm ）
将转换矩阵应用于页面。
参数：	ctm（tuple） - 包含变换矩阵的操作数的6元素元组。

artBox
A RectangleObject，以默认用户空间单位表示，用于定义页面创建者所期望的页面有意义内容的范围。

bleedBox
A RectangleObject，以默认的用户空间单位表示，定义在生产环境中输出时应将页面内容剪切到的区域。

compressContentStreams（）
通过连接所有内容流并应用FlateDecode过滤器来压缩此页面的大小。
但是，如果内容流压缩由于某种原因变为“自动”，则此功能可能不执行任何操作。

static createBlankPage（pdf=None，width=None，height=None ）
返回一个新的空白页面。如果width或者height是None，尝试从最后一页得到的页面大小的PDF。
参数：	
pdf - 页面所属的PDF文件
width (float) - 以默认用户空间单位表示的新页面的宽度。
height (float) - 以默认用户空间单位表示的新页面的高度。
返回：	新的空白页面：
返回类型：PageObject
引发PageSizeNotDefinedError：如果pdf是None或不包含页面

cropBox
A RectangleObject，以默认用户空间单位表示，定义默认用户空间的可见区域。当页面被显示或打印时，其内容将被裁剪（裁剪）到该矩形，然后以某种实现定义的方式强加在输出媒体上。默认值：类似mediaBox。

extractText（）
按照它们在内容流中提供的顺序查找所有文本绘图命令，并提取文本。这适用于某些PDF文件，但对其他文件很不好，具体取决于所使用的发生器。这将在未来完善。不要依赖这个函数产生的文本的顺序，因为如果这个函数变得更复杂，它将会改变。
返回：	一个unicode字符串对象。

getContents（）
访问页面内容。
返回：	该/Contents对象，或者None它不存在。 /Contents是可选的，如PDF参考7.7.3.3中所述

mediaBox
A RectangleObject，以默认用户空间单位表示，定义打算在其上显示或打印页面的物理介质的边界。

mergePage（page2）
将两个页面的内容流合并为一个。资源引用（即字体）由两个页面维护。此页面的mediabox / cropbox / etc不会更改。参数页面的内容流将被添加到此页面的内容流的末尾，这意味着它将在此页面之后或“在此页面的顶部”绘制。

参数：	page2（PageObject） - 要合并到此页的页面。应该是一个实例PageObject。

mergeRotatedPage（page2，rotation，expand = False ）
这与mergePage类似，但要合并的流通过应用转换矩阵进行旋转。
参数：	
page2（PageObject） - 要合并到这个页面中的页面。应该是一个实例PageObject。
rotation (float) - 旋转的角度，以度为单位
expand (bool) - 页面是否应该展开以适应要合并页面的尺寸。

mergeRotatedScaledPage（page2，rotation，scale，expand = False ）
这与mergePage类似，但要合并的流通过应用转换矩阵进行旋转和缩放。
参数：	
page2（PageObject） - 要合并到这个页面中的页面。应该是一个实例PageObject。
rotation (float)  - 旋转的角度，以度为单位
scale (float)  - 比例因子
expand (bool)  - 页面是否应该展开以适应要合并页面的尺寸。

mergeRotatedScaledTranslatedPage（page2，rotation，scale，tx，ty，expand = False ）
这与mergePage类似，但要合并的流通过应用转换矩阵进行转换，旋转和缩放。
参数：	
page2（PageObject） - 要合并到这个页面中的页面。应该是一个实例PageObject。
tx（float） - 在X轴上的平移
ty（float） - Y轴上的平移
rotation (float) - 旋转的角度，以度为单位
scale (float) - 比例因子
expand (bool)  - 页面是否应该展开以适应要合并页面的尺寸。

mergeRotatedTranslatedPage（page2，rotation，tx，ty，expand = False ）
这与mergePage类似，但要合并的流通过应用转换矩阵进行旋转和翻译。
参数：	
page2（PageObject） - 要合并到这个页面中的页面。应该是一个实例PageObject。
tx（float） - 在X轴上的平移
ty（float） - Y轴上的平移
rotation (float) - 旋转的角度，以度为单位
expand (bool) - 页面是否应该展开以适应要合并页面的尺寸。

mergeScaledPage（page2，scale，expand = False ）
这与mergePage类似，但要合并的流通过应用转换矩阵进行缩放。
参数：	
page2（PageObject） - 要合并到此页的页面。应该是一个实例PageObject。
scale (float)  - 比例因子
expand (bool)  - 页面是否应该展开以适应要合并页面的尺寸。

mergeScaledTranslatedPage（page2，scale，tx，ty，expand = False ）
这与mergePage类似，但要合并的流将通过应用转换矩阵进行转换和缩放。

参数：	
page2（PageObject） - 要合并到这个页面中的页面。应该是一个实例PageObject。
scale (float) - 比例因子
tx（float） - 在X轴上的平移
ty（float） - Y轴上的平移
expand (bool)  - 页面是否应该展开以适应要合并页面的尺寸。

mergeTransformedPage（page2，ctm，expand = False ）
这与mergePage类似，但转换矩阵应用于合并流。
参数：	
page2（PageObject） - 要合并到此页的页面。应该是一个实例PageObject。
ctm (tuple) - 一个包含变换矩阵操作数的6元素元组
expand (bool)  - 页面是否应该展开以适应要合并页面的尺寸。

mergeTranslatedPage（page2，tx，ty，expand = False ）
这与mergePage类似，但要合并的流通过应用转换矩阵进行转换。
参数：	
page2（PageObject） - 要合并到这个页面中的页面。应该是一个实例PageObject。
tx（float） - 在X轴上的平移
ty（float） - Y轴上的平移
expand (bool)  - 页面是否应该展开以适应要合并页面的尺寸。

rotateClockwise（angle）
顺时针以90度为增量旋转页面。
参数：	angle（int） - 旋转页面的角度。必须是90度的增量。

rotateCounterClockwise（angle）
逆时针以90度为增量旋转页面。
参数：	angle（int） - 旋转页面的角度。必须是90度的增量。

scale（sx，sy ）
通过向其内容应用转换矩阵并更新页面大小，按给定因子缩放页面。
参数：	
sx（float） - 水平轴上的缩放因子。
sy（float） - 垂直轴上的缩放因子。

scaleBy（factor）
通过向其内容应用转换矩阵并更新页面大小，按给定因子缩放页面。
参数：	factor（float） - 缩放因子（对于X和Y轴）。

scaleTo（width, height）
通过向其内容应用转换矩阵并更新页面大小，将页面缩放到指定的维度。
参数：	
width (float) - 新的宽度。
height (float) - 新的高度。

trimBox
A RectangleObject，用默认的用户空间单位表示，在修剪后定义完成页面的预期尺寸。
```

**粗略读取 PDF 文本内容**

```
def getPdfContent(filename):
    pdf = PdfFileReader(open(filename, "rb"))
    content = ""
    for i in range(0, pdf.getNumPages()):
        pageObj = pdf.getPage(i)

        extractedText = pageObj.extractText()
        content += extractedText + "\n"
        # return content.encode("ascii", "ignore")
    return content
```

### PdfFileWriter

#### 构造方法

```
class PyPDF2.PdfFileWriter()
这个类支持写PDF文件，给出其他类（通常PdfFileReader）生成的页面。
```

#### 属性和方法

````
addAttachment（fname，fdata ）
在PDF中嵌入文件。
参数：	
fname（str） - 要显示的文件名。
fdata（str） - 文件中的数据。
参考：https : //www.adobe.com/content/dam/Adobe/en/devnet/acrobat/pdfs/PDF32000_2008.pdf第7.11.3节

addBlankPage（width = None，height = None ）
追加一个空白页面到这个PDF文件并且返回它。如果未指定页面大小，请使用最后一页的大小。
参数：	
width (float) - 以默认用户空间单位表示的新页面的宽度。
height (float) - 以默认用户空间单位表示的新页面的高度。
返回：	新添加的页面
返回类型：PageObject
引发PageSizeNotDefinedError：如果宽度和高度未定义且前一页不存在。

addBookmark（title，pagenum，parent = None，color = None，bold = False，italic = False，fit ='/ Fit'，* args ）
为此PDF文件添加书签。
参数：	
title（str） - 用于此书签的标题。
pagenum（int） - 此书签将指向的页码。
parent - 对创建嵌套书签的父书签的引用。
color (tuple) - 书签的颜色为从0.0到1.0的红色，绿色和蓝色元组
bold (bool) - 书签是大胆的
italic (bool)  - 书签是斜体
fit（str） - 适合目标页面。详情请参阅 addLink()。

addJS（javascript ）
添加将在打开此PDF时启动的Javascript。
参数：	
javascript（str） 

addLink（pagenum，pagedest，rect，border = None，fit ='/ Fit'，* args ）
从矩形区域添加一个内部链接到指定页面。
参数：	
pagenum（int） - 放置链接的页面的索引。
pagedest（int） - 链接应该到的页面的索引。
rect - RectangleObject或四个整数数组，指定可点击的矩形区域[xLL, yLL, xUR, yUR] ，或表单中的字符串。"[ xLL yLL xUR yUR ]"
border - 如果提供了一个描述边框图属性的数组。有关详细信息，请参阅PDF规范。如果省略此参数，则不会绘制边框。
fit（str） - 页面拟合或“缩放”选项（请参阅下文）。可能需要提供其他参数。传递None将作为该坐标的空值读取。
有效缩放参数（有关详细信息，请参阅PDF 1.7参考的表8.2）：
/Fit	No additional arguments
/XYZ	[left] [top] [zoomFactor]
/FitH	[top]
/FitV	[left]
/FitR	[left] [bottom] [right] [top]
/FitB	No additional arguments
/FitBH	[top]
/FitBV	[left]

addMetadata（infos ）
将自定义元数据添加到输出。
参数：	
infos（dict） - 一个Python字典，其中每个键都是一个字段，每个值都是您的新元数据。

addPage（page）
添加页面到这个PDF文件。该页面通常从一个PdfFileReader实例中获取 。
参数：
page（PageObject） - 要添加到文档中的页面。应该是一个实例PageObject

appendPagesFromReader（reader，after_page_append =无）
从阅读器复制页面到写入器。包含一个可选的回调参数，在将页面追加到写入器后调用该参数。
参数：
reader - 一个PdfFileReader对象，从该对象将页面注释复制到此writer对象。作者的注释将会被更新：
callback after_page_append（function）：
后面调用的回调函数，每个页面都附加到作者。
Callback signature:
param writer_pageref（PDF页面参考）：参考附加到作者的页面。

cloneDocumentFromReader（reader，after_page_append =无）
从PDF文件阅读器创建文档的副本（克隆）
参数：	
reader - 应从其创建克隆的PDF文件阅读器实例
Callback after_page_append (function)： 	
在每个页面附加到编写器后调用的回调函数。签名包括对附加页面的引用（委托给appendPagesFromReader）。
Callback signature：
param writer_pageref（PDF页面参考）： 	引用刚刚添加到文档中的页面。

cloneReaderDocumentRoot（reader）
将阅读器文档根复制到作者。
参数：	reader - 应复制文档根目录中的PdfFileReader。
：callback after_page_append

encrypt（user_pwd，owner_pwd = None，use_128bit = True ）
使用PDF标准加密处理程序加密此PDF文件。
参数：	
user_pwd（str） - “用户密码”，允许打开和阅读提供的限制的PDF文件。
owner_pwd（str） - “所有者密码”，允许无任何限制地打开PDF文件。默认情况下，所有者密码与用户密码相同。
use_128bit（bool） - 标志是否使用128位加密。当错误时，将使用40位加密。默认情况下，此标志打开。

getNumPages（）
返回：	页数。
返回类型：INT

getPage（pageNumber ）
通过此PDF文件中的编号检索页面。
参数：	pageNumber（int） - 要检索的页码（页面从零开始）
返回：	由pageNumber给出的索引处的页面
返回类型：PageObject

getPageLayout（）
获取页面布局。请参阅有关setPageLayout()有效布局的说明。
返回：	目前正在使用的页面布局。
返回类型：	str，如果未指定，则为None

getPageMode（）
获取页面模式。请参阅有关setPageMode()有效模式的说明。
返回：	目前正在使用的页面模式。
返回类型：	str，如果未指定，则为None

insertBlankPage（width = None，height = None，index = 0 ）
将空白页面插入此PDF文件并返回。如果未指定页面大小，请使用最后一页的大小。
参数：	
width (float)  - 以默认用户空间单位表示的新页面的宽度。
height (float) - 以默认用户空间单位表示的新页面的高度。
index（int） - 添加页面的位置。
返回：新添加的页面
返回类型：PageObject
引发PageSizeNotDefinedError：如果宽度和高度未定义且前一页不存在。

insertPage（page，index = 0 ）
在这个PDF文件中插入一个页面。该页面通常从一个PdfFileReader实例中获取 。
参数：	
page（PageObject） - 要添加到文档中的页面。这个论点应该是一个实例PageObject。
index（int） - 页面将被插入的位置。

pageLayout
读取和写入访问getPageLayout() 和setPageLayout()方法的属性。

pageMode
读取和写入访问getPageMode() 和setPageMode()方法的属性。

removeImages（ignoreByteStringObject = False ）
从该输出中删除图像。
参数：	ignoreByteStringObject（bool） - 可选参数，用于忽略ByteString对象。

removeLinks（）
从此输出中删除链接和注释。

removeText（ignoreByteStringObject = False ）
从该输出中删除图像。
参数：	ignoreByteStringObject（bool） - 可选参数，用于忽略ByteString对象。

setPageLayout（布局）
设置页面布局
参数：	layout (str) - 要使用的页面布局
有效的布局是：
/NoLayout	显式没有指定布局
/SinglePage	一次显示一页
/OneColumn	一次显示一列
/TwoColumnLeft	以两列显示页面，左边显示奇数页面
/TwoColumnRight	以两列显示页面，右侧显示奇数页面
/TwoPageLeft	一次显示两页，左侧显示奇数页
/TwoPageRight	一次显示两页，右侧显示奇数页

setPageMode（模式）
设置页面模式。
参数：	mode (str)  - 要使用的页面模式。
有效的模式是：
/UseNone	不要显示轮廓或缩略图面板
/UseOutlines	显示轮廓（又名书签）面板
/UseThumbs	显示页面缩略图面板
/FullScreen	全屏视图
/UseOC	显示可选内容组（OCG）面板
/UseAttachments	显示附件面板

updatePageFormFieldValues（page, fields）
更新字段字典中给定页面的表单字段值。将字段文本和值从字段复制到页面。
参数：	
page - 来自PDF编写器的页面引用，其中注释和字段数据将被更新。
fields - 字段名称（/ T）和文本值（/ V）的Python字典

write（stream）
将添加到此对象的页面集合写为PDF文件。
参数：	stream - 将文件写入的对象。该对象必须支持写入方法和tell方法，类似于文件对象。

````

**PDF 写入操作：**

```
def addBlankpage():
    readFile = 'C:/Users/Administrator/Desktop/RxJava 完全解析.pdf'
    outFile = 'C:/Users/Administrator/Desktop/copy.pdf'
    pdfFileWriter = PdfFileWriter()

    # 获取 PdfFileReader 对象
    pdfFileReader = PdfFileReader(readFile)  # 或者这个方式：pdfFileReader = PdfFileReader(open(readFile, 'rb'))
    numPages = pdfFileReader.getNumPages()

    for index in range(0, numPages):
        pageObj = pdfFileReader.getPage(index)
        pdfFileWriter.addPage(pageObj)  # 根据每页返回的 PageObject,写入到文件
        pdfFileWriter.write(open(outFile, 'wb'))

    pdfFileWriter.addBlankPage()   # 在文件的最后一页写入一个空白页,保存至文件中
    pdfFileWriter.write(open(outFile,'wb'))
```

结果是：在写入的 copy.pdf 文档的最后最后一页写入了一个空白页。

## 使用

PyPDF2 包含了 PdfFileReader PdfFileMerger PageObject PdfFileWriter 四个常用的主要 Class。

### 简单读写

```
from PyPDF2 import PdfFileReader, PdfFileWriter 
readFile = 'read.pdf'
writeFile = 'write.pdf'
# 获取一个 PdfFileReader 对象 
pdfReader = PdfFileReader(open(readFile, 'rb')) 
# 获取 PDF 的页数 
pageCount = pdfReader.getNumPages() 
print(pageCount) 
# 返回一个 PageObject 
page = pdfReader.getPage(i) 
# 获取一个 PdfFileWriter 对象 
pdfWriter = PdfFileWriter() 
# 将一个 PageObject 加入到 PdfFileWriter 中 
pdfWriter.addPage(page) 
# 输出到文件中 
pdfWriter.write(open(writeFile, 'wb'))
```

### 合并分割

```
from PyPDF2 import PdfFileReader, PdfFileWriter 
def split_pdf(infn, outfn): 
    pdf_output = PdfFileWriter() 
    pdf_input = PdfFileReader(open(infn, 'rb')) 
    # 获取 pdf 共用多少页 
    page_count = pdf_input.getNumPages() 
    print(page_count) 
    # 将 pdf 第五页之后的页面，输出到一个新的文件 
    for i in range(5, page_count): 
        pdf_output.addPage(pdf_input.getPage(i)) 
    pdf_output.write(open(outfn, 'wb')) 
def merge_pdf(infnList, outfn): 
    pdf_output = PdfFileWriter() 
    for infn in infnList: 
        pdf_input = PdfFileReader(open(infn, 'rb')) 
        # 获取 pdf 共用多少页 
        page_count = pdf_input.getNumPages() 
        print(page_count) 
        for i in range(page_count): 
            pdf_output.addPage(pdf_input.getPage(i)) 
    pdf_output.write(open(outfn, 'wb')) 
if __name__ == '__main__': 
    infn = 'infn.pdf'
    outfn = 'outfn.pdf'
    split_pdf(infn, outfn)
```

###包含添加书签方法：

```
# -*- coding: utf-8 -*-

import os

from PyPDF2 import PdfFileWriter, PdfFileReader


class Pdf(object):
    def __init__(self, path):
        self.path = path
        reader = PdfFileReader(open(path, "rb"))
        self.writer = PdfFileWriter()
        self.writer.appendPagesFromReader(reader)
        self.writer.addMetadata(reader.getDocumentInfo())

    @property
    def new_path(self):
        name, ext = os.path.splitext(self.path)
        return name + '_new' + ext

    def add_bookmark(self, title, pagenum, parent=None):
        return self.writer.addBookmark(title, pagenum, parent=parent)

    def save_pdf(self):
        with open(self.new_path, 'wb') as out:
            self.writer.write(out)

```
###裁切(by pypdf)

```
from pyPdf import PdfFileWriter, PdfFileReader

pdf = PdfFileReader(file('original.pdf', 'rb'))
out = PdfFileWriter()

for page in pdf.pages:
page.mediaBox.upperRight = (580,800)
page.mediaBox.lowerLeft = (128,232)
out.addPage(page)

ous = file('target.pdf', 'wb')
out.write(ous)
ous.close()
```

###把三个pdf文件合成一个

```
import PyPDF2  
pdff1=open("index2.pdf","rb")  
pr=PyPDF2.PdfFileReader(pdff1)  
print pr.numPages  
  
pdff2=open("a.pdf","rb")  
pr2=PyPDF2.PdfFileReader(pdff2)  
  
pdf3=open("foot.pdf","rb")  
pr3=PyPDF2.PdfFileReader(pdf3)  
  
pdfw=PyPDF2.PdfFileWriter()  
pageobj=pr.getPage(0)  
pdfw.addPage(pageobj)  
  
  
for pageNum in range(pr2.numPages):  
    pageobj2=pr2.getPage(pageNum)  
    pdfw.addPage(pageobj2)  
      
pageobj3=pr3.getPage(0)  
pdfw.addPage(pageobj3)  
  
pdfout=open("c.pdf","wb")  
pdfw.write(pdfout)  
pdfout.close()  
pdff1.close()  
pdff2.close()  
pdf3.close() 
```

###Python解析指定文件夹下pdf文件读取需要的数据并写入数据库

```
from PyPDF2.pdf import PdfFileReader  
import pymysql  
import os  
import os.path  
from time import strftime,strptime  
  
def sqlupdate(sql):  
        conn = pymysql.connect(host="xxx.xxx.xx.xxx",port=3306,user="***",password="***",database="db_name")  
        cur=conn.cursor()  
        cur.execute(sql)  
        conn.commit()  
        conn.close()  
  
def getDataUsingPyPdf2(filename):  
    pdf = PdfFileReader(open(filename, "rb"))  
    content = ""  
    for i in range(0, pdf.getNumPages()):  
        extractedText = pdf.getPage(i).extractText()  
        content +=  extractedText + "\n"  
    #return content.encode("ascii", "ignore")  
    return content  
  
  
def removeBlankFromList(list_old):  
    list_new = []  
    for i in list_old:  
        if i != '':  
            list_new.append(i)  
    return list_new  
  
  
if __name__ == '__main__':  
    rootdir = '/root/'  
    for dirpath,dirnames,filenames in os.walk(rootdir):  
        filedir = dirpath.split('/')[-1]  
        for filename in filenames:  
            filename = filename  
            filename_long = dirpath+'/'+filename  
            outputString = getDataUsingPyPdf2(filename_long)  
            #outputString = getDataUsingPyPdf2("/root/a.pdf")  
            outputString = outputString.split('\n')  
            outputString_new = removeBlankFromList(outputString)  
            outputString = outputString_new  
            try:  
                rectime = strftime('%Y-%m-%d %H:%M:%S',strptime(outputString[1].rstrip(' '), "%a %b %d %Y %H:%M:%S"))  
            except:  
                rectime = strftime('%Y-%m-%d %H:%M:%S',strptime(outputString[-1].rstrip(' '), "%a %b %d %Y %H:%M:%S"))  
            pn = outputString[0].split()  
            if len(pn) > 1:  
                sn = pn[1]  
            else:  
                sn = ''  
            pn = pn[0].strip()  
            #print('[%s],[%s],[%s]' % (pn,topbut,rectime))  
            sql = "insert into tb_1(pn,sn,rectime,dir,filename) values ('%s','%s','%s','%s','%s')" % (pn,sn,rectime,filedir,filename)  
            print('sql=[%s]' % sql)  
            sqlupdate(sql)  
            print('done')  
    #print(getDataUsingPyPdf2("/root/a.pdf")) 
```

###官方示例：

```
from PyPDF2 import PdfFileWriter, PdfFileReader

output = PdfFileWriter()
input1 = PdfFileReader(open("document1.pdf", "rb"))

# print how many pages input1 has:
print "document1.pdf has %d pages." % input1.getNumPages()

# add page 1 from input1 to output document, unchanged
output.addPage(input1.getPage(0))

# add page 2 from input1, but rotated clockwise 90 degrees
output.addPage(input1.getPage(1).rotateClockwise(90))

# add page 3 from input1, rotated the other way:
output.addPage(input1.getPage(2).rotateCounterClockwise(90))
# alt: output.addPage(input1.getPage(2).rotateClockwise(270))

# add page 4 from input1, but first add a watermark from another PDF:
page4 = input1.getPage(3)
watermark = PdfFileReader(open("watermark.pdf", "rb"))
page4.mergePage(watermark.getPage(0))
output.addPage(page4)


# add page 5 from input1, but crop it to half size:
page5 = input1.getPage(4)
page5.mediaBox.upperRight = (
    page5.mediaBox.getUpperRight_x() / 2,
    page5.mediaBox.getUpperRight_y() / 2
)
output.addPage(page5)

# add some Javascript to launch the print window on opening this PDF.
# the password dialog may prevent the print dialog from being shown,
# comment the the encription lines, if that's the case, to try this out
output.addJS("this.print({bUI:true,bSilent:false,bShrinkToFit:true});")

# encrypt your new PDF and add a password
password = "secret"
output.encrypt(password)

# finally, write "output" to document-output.pdf
outputStream = file("PyPDF2-output.pdf", "wb")
output.write(outputStream)
```



