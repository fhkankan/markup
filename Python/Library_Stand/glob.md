# glob

```python
glob.glob()
```
查找匹配文件或文件夹(目录)
```python
# 使用Unix shell的规则来查找：
# *:匹配任意个任意字符
# ?:匹配单个任意字符
# [字符列表]:匹配字符列表中的任意一个字符
# [!字符列表]:匹配除列表外的其他字符

# 查找以d开头的文件或文件夹
glob.glob('d*')
# 查找以d开头并且全长为5个字符的文件或文件夹
glob.glob('d????')
# 查找以abcd中任一字符开头的文件或文件夹
glob.glob('[abcd]*')
# 查找不以abc中任一字符开头的文件或文件夹
glob.glob('[!abc]')
```
应用
```python
import glob
for name in glob.glob('dir/*.py'):
    print(name)
```