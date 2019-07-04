# Mammoth

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

```
pip install mammoth
```

支持的其他平台

```
- JavaScript，包括浏览器和node.js。可从npm上获取。
- Python。可从PyPI上获取。
- WordPress。
- Java/JVM。可从Maven Central上获取。
```

## 使用

### CLI

您可以通过将路径传递给docx文件和输出文件来转换docx文件。例如：
```shell
mammoth document.docx output.html
```
如果未指定输出文件，则输出将写入stdout。
输出是一个HTML片段，而不是用UTF-8编码的完整HTML文档。由于未在片段中显式设置编码，因此如果浏览器未默认为UTF-8，则在Web浏览器中打开输出文件可能会导致Unicode字符被错误地呈现。
#### images

默认情况下，图像包含在输出HTML中。如果使用了`--output-dir`指定了输出目录，则会将图像写入单独的文件。若是文件已存在则被重写覆盖。例如：

```shell
mammoth document.docx --output-dir=output-dir
```
#### styles

可以使用`--style-map`从文件中读取自定义样式映射。例如

```shell
mammoth document.docx output.html --style-map=custom-style-map
```
其中，`custom-style-map`类似如下
```shell
p[style-name='Aside Heading'] => div.aside > h2:fresh
p[style-name='Aside Text'] => div.aside > p:fresh
```
#### Markdown

使用`--output-format=markdown`会产生Markdown格式

Markdown支持仍处于早期阶段，因此您可能会发现某些功能不受支持

```
mammoth document.docx --output-format=markdown
```

### 库

#### 基本转换

要将一个已经存在的.docx文件转换为HTML，将类文件对象传递给`mammoth.convert_to_html`，该文件应以二进制模式打开。例如：

```python
import mammoth

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file)
    html = result.value # The generated HTML
    messages = result.messages # Any messages, such as warnings during conversion
```

您还可以使用`mammoth.extract_raw_text`提取文档的原始文本。这将忽略文档中的所有格式。每个段落后跟两个换行符。

```python
with open("document.docx", "rb") as docx_file:
    result = mammoth.extract_raw_text(docx_file)
    text = result.value # The raw text
    messages = result.messages # Any messages
```

#### 自定义样式映射

默认情况下，Mammoth把一些普通的.docx样式映射为HTML元素。比如，带有名为`Heading 1`的样式的一个段落会被转换为一个`h1`元素。您可以通过将带有`style_map`属性的options对象作为`convert_to_html`的第二个参数传递来传递样式的自定义映射。关于样式映射语法的描述包含在“编写样式映射配置”一节中。例如，将带有名为`Section Title`的样式的段落转换为`h1`元素，带有名为`Subsection Title`的样式的段落转换为`h2`元素：

```python
import mammoth

style_map = """
p[style-name='Section Title'] => h1:fresh
p[style-name='Subsection Title'] => h2:fresh
"""

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, style_map=style_map)
```

用户定义的样式映射优先于默认样式映射使用。要完全停止使用默认样式映射，请传递`include_default_style_map = False`：

```python
result = mammoth.convert_to_html(docx_file, style_map=style_map, include_default_style_map=False)
```

#### 自定义图像处理程序

默认情况下，图像将转换为`<img>`元素，并且源包含在`src`属性中。通过将`convert_image`参数设置为图像转换器可以更改此行为。

例如，以下内容将复制默认行为

```python
def convert_image(image):
    with image.open() as image_bytes:
        encoded_src = base64.b64encode(image_bytes.read()).decode("ascii")
    
    return {
        "src": "data:{0};base64,{1}".format(image.content_type, encoded_src)
    }

mammoth.convert_to_html(docx_file, convert_image=mammoth.images.img_element(convert_image))
```

#### 粗体

默认情况下，粗体文本被包装在`<strong>`标签中。可以通过添加对`b`的样式映射改变这种行为。比如，要把粗体文本包装在`<em>`标签中：

```python
style_map = "b => em"

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, style_map=style_map)
```

#### 斜体

默认情况下，斜体文本被包装在`<em>`标签中。可以通过添加对`i`的样式映射改变这种行为。比如，要把斜体文本包装在`<strong>`标签中：

```python
style_map = "i => strong"

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, style_map=style_map)
```

#### 下划线

默认情况下，由于会与HTML文档中的链接引起混淆，所有文本的下划线均被忽略。可以通过添加对`u`的样式映射改变这种行为。比如，有一个源文档使用下划线表示强调。下面的代码会把所有带显式下划线的源文本包装在`<em>`标签中：

```python
import mammoth

style_map = "u => em"

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, style_map=style_map)
```

#### 删除线

默认情况下，带删除线的文本被包装在`<s>`标签中。可以通过添加对`strike`的样式映射改变这种行为。比如，要把带删除线的文本包装在`<del>`标签中：

```python
style_map = "strike => del"

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, style_map=style_map)
```

#### 注释

默认情况下，注释是被忽略的。为`comment-reference`添加一个类型映射，可以将注释生成在HTML中。

注释将附加到文档的末尾，并带有使用指定样式映射包装的注释的链接。

```python
style_map = "comment-reference => sup"

with open("document.docx", "rb") as docx_file:
    result = mammoth.convert_to_html(docx_file, style_map=style_map)
```

### [API](https://github.com/mwilliamson/python-mammoth)

| name                                             | Desc                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `mammoth.convert_to_html(fileobj, **kwargs)`     | 将源文档转换为html                                           |
| `mammoth.convert_to_markdown(fileobj, **kwargs)` | 将源文档转换为Markdown。这与`convert_to_html`的行为相同，只是结果的value属性包含Markdown而不是HTML。 |
| `mammoth.extract_raw_text(fileobj)`              | 提取文档的原始文本。这将忽略文档中的所有格式。每个段落后跟两个换行符 |
| `mammoth.embed_style_map(fileobj, style_map)`    | 将样式贴图style_map嵌入到fileobj中。当Mammoth读取文件对象时，它将使用嵌入式样式图。 |

#### message

每个消息都有如下属性

| name      | Desc                                |
| --------- | ----------------------------------- |
| `type`    | 表示消息类型的字符串，例如“warning” |
| `message` | 包含实际消息的字符串                |

#### 图片转换器

可以通过调用`mammoth.images.img_element(func)`来创建图像转换器。这会为原始docx中的每个图像创建一个`<img>`元素。`func`应该是一个具有一个参数图像的函数。此参数是要转换的图像元素，并具有以下属性：

```
open()				打开图像文件。返回类文件对象。
content_type	图像的内容类型，如image/png
```

`func`应该返回`<img>`元素的属性的dict。至少，这应该包括`src`属性。如果找到图像的任何替代文本，则会自动将其添加到元素的属性中。

例如，以下内容复制默认图像转换：

```python
def convert_image(image):
    with image.open() as image_bytes:
        encoded_src = base64.b64encode(image_bytes.read()).decode("ascii")
    
    return {
        "src": "data:{0};base64,{1}".format(image.content_type, encoded_src)
    }

mammoth.images.img_element(convert_image)
```

`mammoth.images.data_uri`是默认图像转换器

Mammoth默认不处理WMF图像。处理内容目录包含如何使用LibreOffice转换它们的示例，尽管转换的保真度完全取决于LibreOffice。

#### 文档转换

用于文档转换的API应被视为不稳定，并且可能在任何版本之间发生变化。如果您依赖此行为，则应该固定特定版本的Mammoth，并在更新前仔细测试。
Mammoth允许文档在转换之前进行修正。例如，假设文档没有进行语义标记，但您知道任何中心对齐的段落应该是标题。您可以使用`transform_document`参数来适当地修改文档：
```python
import mammoth.transforms

def transform_paragraph(element):
    if element.alignment == "center" and not element.style_id:
        return element.copy(style_id="Heading2")
    else:
        return element

transform_document = mammoth.transforms.paragraph(transform_paragraph)

mammoth.convert_to_html(fileobj, transform_document=transform_document)
```
或者，如果您希望已明确设置的段落使用等宽字体来表示代码：
```python
import mammoth.documents
import mammoth.transforms

_monospace_fonts = set(["courier new"])

def transform_paragraph(paragraph):
    runs = mammoth.transforms.get_descendants_of_type(paragraph, mammoth.documents.Run)
    if runs and all(run.font and run.font.lower() in _monospace_fonts for run in runs):
        return paragraph.copy(style_id="code", style_name="Code")
    else:
        return paragraph

convert_to_html(
    fileobj,
    transform_document=mammoth.transforms.paragraph(transform_paragraph),
    style_map="p[style-name='Code'] => pre:separator('\n')",
)
```
|  name    |   desc   |
| ---- | ---- |
| `mammoth.transforms.paragraph(transform_paragraph)` | 返回一个可用作`transform_document`参数的函数。这将将函数`transform_paragraph`应用于每个段落元素。`transform_paragraph`应返回新段落。 |
| `mammoth.transforms.run(transform_run)` | 返回一个可用作`transform_document`参数的函数。这将将函数`transform_run`应用于每个run元素。`transform_run`应该返回新的运行。 |
| `mammoth.transforms.get_descendants(element)` | 获取元素的所有后代。 |
| `mammoth.transforms.get_descendants_of_type(element, type)` | 获取特定类型元素的所有后代。例如，要获取元素段落中的所有运行 |

示例

```python
import mammoth.documents
import mammoth.transforms

runs = mammoth.transforms.get_descendants_of_type(paragraph, documents.Run)
```

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

例如，有一个样式映射为`p[style-name='Heading 1'] => h1`。如果Mammoth遇到了一个包含名为`Heading 1`的样式的段落，这个段落会被转换为包含相同文本的`h1`元素。如果下一个段落也包含名为`Heading 1`的样式，那么这个段落的文本会被追加到已有的`h1`元素，而不是创建一个新的`h1`元素。

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