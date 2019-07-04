# word转html

## Mammoth

用于.NET平台.docx转HTML的Mammoth

Mammoth可用于将.docx文档（比如由Microsoft Word创建的）转换为HTML。Mammoth致力于通过文档中的语义信息生成简洁的HTML，而忽略一些其他细节。例如，Mammoth会把带有“Heading 1”样式的所有段落转换为“h1”元素，而不是试图精确地复制标题的所有样式（字体、字号、颜色等）。

.docx使用的结构与HMTL的结构有很多不匹配的地方，这意味着复杂文档的转换很难达到完美。但如果你仅使用样式进行文档的语义化标记，Mammoth将会工作得很好。

当前支持如下特性：

```
- 标题。
- 列表。
- 自定义从.docx样式到HTML的映射。比如，通过提供合适的样式映射，可以把“WarningHeading”样式转换为“h1.warning”类。
- 表格。表格自身的样式——比如边框——目前会被忽略，但对文本格式的处理与文档的其余部分一致。
- 脚注和尾注。
- 图像。
- 粗体、斜体、下划线、删除线、上标和下标。
- 链接。
- 换行。
- 文本框。文本框中的内容作为一个单独的段落处理，放在包含该文本框的段落之后。
```

## 安装

可从Nuget上获取。

```
Install-Package Mammoth
```

支持的其他平台

```
- JavaScript，包括浏览器和node.js。可从npm上获取。
- Python。可从PyPI上获取。
- WordPress。
- Java/JVM。可从Maven Central上获取。
```

## 使用

### 库

- 基本转换

要将一个已经存在的.docx文件转换为HTML，只需创建DocumentConverter的一个实例，并将文件路径传递给ConvertToHtml方法。例如：

```python
using Mammoth;

var converter = new DocumentConverter();
var result = converter.ConvertToHtml("document.docx");
var html = result.Value; // 生成的HTML
var warnings = result.Warnings; // 转换期间产生的所有警告
```

你也可以使用ExtractRawText方法提取文档的纯文本。这会忽略文档中的所有格式。每个段落后跟两个换行符。

```python
var converter = new DocumentConverter();
var result = converter.ExtractRawText("document.docx");
var html = result.Value; // 纯文本
var warnings = result.Warnings; // 转换期间产生的所有警告
```

- 自定义样式映射

默认情况下，Mammoth把一些普通的.docx样式映射为HTML元素。比如，带有名为“Heading 1”的样式的一个段落会被转换为一个“h1”元素。关于样式映射语法的描述包含在“编写样式映射配置”一节中。例如，将带有名为“Section Title”的样式的段落转换为“h1”元素，带有名为“Subsection Title”的样式的段落转换为“h2”元素：

```python
var converter = new DocumentConverter()
    .AddStyleMap("p[style-name='Section Title'] => h1:fresh")
    .AddStyleMap("p[style-name='Subsection Title'] => h2:fresh");
```

也可以将整个样式映射作为一个字符串传递，当样式映射存储在文本文件中时，这会很有用：

```python
var styleMap =
    "p[style-name='Section Title'] => h1:fresh\n" +
    "p[style-name='Subsection Title'] => h2:fresh";
var converter = new DocumentConverter()
    .AddStyleMap(styleMap);
```

后添加的样式映射拥有较高的优先级。用户定义的样式映射优于默认样式映射。如果要禁用所有的默认样式映射，可调用DisableDefaultStyleMap方法：

```python
var converter = new DocumentConverter()
    .DisableDefaultStyleMap();
```

- 粗体

默认情况下，粗体文本被包装在`“<strong>”`标签中。可以通过添加对“b”的样式映射改变这种行为。比如，要把粗体文本包装在`“<em>”`标签中：

```python
var converter = new DocumentConverter()
    .AddStyleMap("b => em");
```

- 斜体

默认情况下，斜体文本被包装在`“<em>”`标签中。可以通过添加对“i”的样式映射改变这种行为。比如，要把斜体文本包装在`“<strong>”`标签中：

```python
var converter = new DocumentConverter()
    .AddStyleMap("i => strong");
```

- 下划线

默认情况下，由于会与HTML文档中的链接引起混淆，所有文本的下划线均被忽略。可以通过添加对“u”的样式映射改变这种行为。比如，有一个源文档使用下划线表示强调。下面的代码会把所有带显式下划线的源文本包装在`“<em>”`标签中：

```python
var converter = new DocumentConverter()
    .AddStyleMap("u => em");
```

- 删除线

默认情况下，带删除线的文本被包装在`“<s>”`标签中。可以通过添加对`“strike”`的样式映射改变这种行为。比如，要把带删除线的文本包装在`“<del>”`标签中：

```python
var converter = new DocumentConverter()
    .AddStyleMap("strike => del");
```

### API
`DocumentConverter`

方法：

| name                                             | Desc                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `IResult<string> ConvertToHtml(string path)`     | 将由path参数指定的文件转换为一个HTML字符串                   |
| `IResult<string> ConvertToHtml(string path)`     | 将由path参数指定的文件转换为一个HTML字符串                   |
| `IResult<string> ConvertToHtml(Stream stream)`   | 将由stream参数指定的流转换为一个HTML字符串。注意，使用这个方法，而不是convertToHtml(File file)方法意味着指向其他文件——比如图片——的相对路径将无法解析 |
| `IResult<string> ExtractRawText(string path)`    | 提取文档中的纯文本。这将忽略文档中的所有格式。每个段落后跟两个换行符 |
| `IResult<string> ExtractRawText(Stream stream)`  | 提取文档中的纯文本。 这将忽略文档中的所有格式。每个段落后跟两个换行符 |
| `DocumentConverter AddStyleMap(string styleMap)` | 添加用于指定Word样式到HTML的映射的样式映射。最后添加的样式映射具有最高的优先级。“编写样式映射配置”一节提供了对其语法的说明 |
| `DocumentConverter DisableDefaultStyleMap()`     | 默认情况下，任何新添加的样式映射都会跟默认样式映射合并起来。调用这个方法可以停用默认样式映射 |
| `DocumentConverter PreserveEmptyParagraphs()`    | 默认情况下，空段落将会被忽略。调用这个方法可以在输出中保留空段落 |
| `DocumentConverter IdPrefix(string idPrefix)`    | 设置生成的任何ID的前缀，比如用于书签、脚注和尾注等的ID。默认为空字符串 |

`IResult<T>`

表示转换的结果。属性：

| name                    | desc                   |
| ----------------------- | ---------------------- |
| `T Value`               | 生成的文本             |
| `ISet<string> Warnings` | 转换期间生成的所有警告 |

## 编写样式映射配置

 一个样式映射配置由几个使用换行符分隔的样式映射组成。空行和由“#”开始的行会被忽略。

一个样式映射由两部分组成：

```
- 箭头左侧为文档元素匹配器。
- 箭头右侧为HTML路径。
```

每转换一个段落，Mammoth会查找文档元素匹配器匹配该段落的第一个样式映射，然后Mammoth确保满足HTML路径。

### 新建元素

当编写样式映射时，理解Mammoth中关于新建元素的概念是很有用的。在生成HTML的时候，Mammoth仅在必要的时候才会关闭一个HTML元素。否则，元素会被重用。

例如，有一个样式映射为`“p[style-name='Heading 1'] => h1”`。如果Mammoth遇到了一个包含名为“Heading 1”的样式的段落，这个段落会被转换为包含相同文本的“h1”元素。如果下一个段落也包含名为“Heading 1”的样式，那么这个段落的文本会被追加到已有的“h1”元素，而不是创建一个新的“h1”元素。

许多情况下，你可能希望生成一个新的“h1”元素。你可以通过使用“:fresh”修饰符指明这么做：

```
p[style-name='Heading 1'] => h1:fresh
```

然后两个连续的“Heading 1”段落会被转换为两个独立的“h1”元素。

当生成较为复杂的HTML结构的时候，重用元素就比较有用。例如，假设你的.docx文档包含旁注。每个旁注可能包含一个标题和一些正文文本，它们应该被包含在一个单独的“div.aside”元素中。这种情况下，类似于`“p[style-name='Aside Heading'] => div.aside > h2:fresh”`和`“p[style-name='Aside Text'] => div.aside > p:fresh”`的样式映射可能会有帮助。

### 文档元素匹配器

- 段落和内联文本

匹配所有段落：

```
p
```

匹配所有内联文本：

```
r
```

要匹配带有指定样式的段落和内联文本，你可以通过名称引用样式，即显示在Microsoft Word或LibreOffice中的样式名称。比如，要匹配一个带有样式名“Heading 1”的段落：

```
p[style-name='Heading 1']
```

也可以通过样式ID引用样式，即在.docx文件内部使用的ID。要匹配一个带有指定样式ID的段落或内联文本，追加一个点，后跟样式ID即可。比如，要匹配一个带有样式ID“Heading1”的段落：

```
p.Heading1
```

- 粗体

匹配显式的粗体文本：

```
b
```

注意，这只会匹配显式地应用了粗体样式的文本，而不会匹配由于其所属段落或内联文本的样式而显示为粗体的文本。

- 斜体

匹配显式的斜体文本：

```
i
```

注意，这只会匹配显式地应用了斜体样式的文本，而不会匹配由于其所属段落或内联文本的样式而显示为斜体的文本。

- 下划线

匹配显式的下划线文本：

```
u
```

注意，这只会匹配显式地应用了下划线样式的文本，而不会匹配由于其所属段落或内联文本的样式而带有下划线的文本。

- 删除线

匹配显式的删除线文本：

```
strike
```

注意，这只会匹配显式地应用了删除线样式的文本，而不会匹配由于其所属段落或内联文本的样式而带有删除线的文本。

### HTML路径

#### 单一元素

最简单的HTML路径只指定单一元素。比如，要指定一个“h1”元素：

```
h1
```

要给元素指定一个CSS类，追加一个点，后跟类名即可：

```
h1.section-title
```

如果要求新建元素，使用“:fresh”：

```
h1:fresh
```

必须按正确顺序使用修饰符：

```
h1.section-title:fresh
```

#### 嵌套元素

使用“>”指定嵌套元素。比如，要指定“h2”在“div.aside”中：

```
div.aside > h2
```

你可以嵌套任意深度的元素。

## 缺失的特性

与Mammoth的JavaScript和Python实现相比，如下特性暂缺：

```
- 自定义图片处理程序
- CLI
- 对嵌入样式映射配置的支持
- Markdown支持
- 文档变换
```

## 