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

## 实例

PyPDF2 将读与写分成两个类来操作：

```
from PyPDF2 import PdfFileWriter, PdfFileReader

writer = PdfFileWriter()
reader = PdfFileReader(open("document1.pdf", "rb"))

```

如果是要修改一个已有的 pdf 文件，可以将 reader 的页面添加到 writer 中：

`writer.appendPagesFromReader(reader)`

添加书签：

`writer.addBookmark(title, pagenum, parent=parent)`

一个包含添加书签方法的类：

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

官方示例：

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

裁切(by pypdf)

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

把三个pdf文件合成一个

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

Python解析指定文件夹下pdf文件读取需要的数据并写入数据库

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

