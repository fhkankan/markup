# word另存为

原理是利用win32com接口直接调用office API，好处是简单、兼容性好，只要office能处理的，python都可以处理，处理出来的结果和office word里面“另存为”一致。

```python
from win32com import client as wc 
word = wc.Dispatch('Word.Application') 
doc = word.Documents.Open('/FilePath/test.docx') 
doc.SaveAs('/DestPath/test.pdf', 17) #17对应于下表中的pdf文件
doc.Close() 
word.Quit()
```

下面是office 2007支持的全部文件格式对应表：

```
wdFormatDocument = 0 
wdFormatDocument97 = 0 
wdFormatDocumentDefault = 16 
wdFormatDOSText = 4 
wdFormatDOSTextLineBreaks = 5 
wdFormatEncodedText = 7 
wdFormatFilteredHTML = 10 
wdFormatFlatXML = 19 
wdFormatFlatXMLMacroEnabled = 20 
wdFormatFlatXMLTemplate = 21 
wdFormatFlatXMLTemplateMacroEnabled = 22 
wdFormatHTML = 8 
wdFormatPDF = 17 
wdFormatRTF = 6 
wdFormatTemplate = 1 
wdFormatTemplate97 = 1 
wdFormatText = 2 
wdFormatTextLineBreaks = 3 
wdFormatUnicodeText = 7 
wdFormatWebArchive = 9 
wdFormatXML = 11 
wdFormatXMLDocument = 12 
wdFormatXMLDocumentMacroEnabled = 13 
wdFormatXMLTemplate = 14 
wdFormatXMLTemplateMacroEnabled = 15 
wdFormatXPS = 18
```

## 批量改pdf

```python
from win32com.client import Dispatch
from os import walk

wdFormatPDF = 17


def doc2pdf(input_file):
    word = Dispatch('Word.Application')
    doc = word.Documents.Open(input_file)
    doc.SaveAs(input_file.replace(".docx", ".pdf"), FileFormat=wdFormatPDF)
    doc.Close()
    word.Quit()


if __name__ == "__main__":
    doc_files = []
    directory = "C:\\Users\\xkw\\Desktop\\destData"
    for root, dirs, filenames in walk(directory):
        for file in filenames:
            if file.endswith(".doc") or file.endswith(".docx"):
                doc2pdf(str(root + "\\" + file))


```

## doc转docx

```python
from win32com import client
def doc2docx(doc_name,docx_name):
    """
    :doc转docx
    """
    try:
        # 首先将doc转换成docx
        word = client.Dispatch("Word.Application")
        doc = word.Documents.Open(doc_name)
        #使用参数16表示将doc转换成docx
        doc.SaveAs(docx_name,16)
        doc.Close()
        word.Quit()
    except:
        pass
if __name__ == '__main__':
    doc2docx(f:test.doc','f:/test.docx')
```

## docx转html

```python
#coding:utf-8
import docx
from docx2html import convert
import HTMLParser
def  docx2html(docx_name,new_name):
    """
    :docx转html
    """
    try:
        #读取word内容
        doc = docx.Document(docx_name,new_name)
        data = doc.paragraphs[0].text
        # 转换成html
        html_parser = HTMLParser.HTMLParser()
        #使用docx2html模块将docx文件转成html串，随后你想干嘛都行
        html = convert(new_name)
        #docx2html模块将中文进行了转义，需要将生成的字符串重新转义
        return html_parser.enescape(html)
    except:
        pass
if __name__ == '__main__':
    docx2html('f:/test.docx','f:/test1.docx')
```

