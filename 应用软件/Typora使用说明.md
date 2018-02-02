---

---

# 区域元素

## YAML FONT Matters

在文章最上方输入---，按换行键产生，输入内容即可

##菜单

输入[toc]+换行键，产生标题，自动更新

```
[toc]

```

[TOC]

## 段落

按换行键建立新的一行<br/>可在行尾插入打断线，禁止向后插入

```
按换行键建立新的一行<br/>
```

## 标题

开头#的个数表示，空格+文字。标题有1~6个级别，#表示开始，按换行键结束

```
# H1
## H2
###### H6
```

## 引注

开头>表示，空格+文字，按换行键换行，双按换行跳出

```
> ni
>
> ni hao
```

> ni
>
> nini

## 序列

开头*/+/-，空格+文字，可以创建无序序列，换行键换行，删除键+shift+tab跳出

开头1.，空格+后接文字，可以创建有序序列

```
*   Red
+   Green
-   Blue

1.  Red
2. 	Green
3.	Blue
```

* 无序序列

1. 你

## 可选序列

开头序列+空格+[ ]+空格+文字，换行键换行，删除键+shift+tab跳出

```
- [ ] a
+ [ ] b
* [ ] c
- [x] completed
```

- [ ] a 

+ [ ] b

* [ ] c


* [x] d

## 代码块

开头```+语言名，开启代码块，换行键换行，光标下移键跳出

```
​```python
​```
```

## 数学块

使用MtathJax建立数学公式

开头$$+换行键，产生输入区域，输入Tex/LaTex格式的数学公式

```
$$

$$
```


$$
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix} 
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\
\end{vmatrix}
$$

## 表格

开头|+列名+|+列名+|+换行键，创建一个2*2表格

将鼠标放置其上，弹出编辑尺寸，个数，文字等

```
|first|second|

```

| first | second |
| :---: | ------ |
|       |        |

## 脚注

在需要添加脚注的文字后面+[+^+序列+]，注释的产生可以鼠标放置其上单击自动产生，添加信息

或人工添加+[+^+序列+]+:

```
脚注产生的方法[^footnote].
[^footnote]:这个就是*脚注*
```

脚注产生的方法[^1].

[^1]: 脚注

## 水平线

输入***/---，换行键换行

```
***
---

```

***

---

# 特征元素

## 链接

单击链接，展开后可编辑

ctr+单击，打开链接

### 超链接

用[]括住要超链接的内容，紧接着用()括住超链接源+名字，超链接源后面+超链接命名

```
This is [an example](http://example.com/ "Title") inline link.

[This link](http://example.net/) has no title attribute.
```

This is [an example](http://example.com/ "Title") inline link.

[This link](http://example.net/) has no title attribute.

### 内链接



### 相关链

使用[+超链接文字+]+[+标签+]，创建可定义链接

```
This is [an example][id] reference-style link.

Then, anywhere in the document, you define your link label like this, on a line by itself:

[id]: http://example.com/  "Optional Title Here"
```

[Baidu][id]
And then define the link:

[id]: http://baidu.com/	"Title"
[Google][]
And then define the link:

[Google]: http://google.com/

## URLs

用<>括住url，可手动设置url

对于标准URLs，可自动识别

```
<i@163.com>
www.baidu.com
```

<i@163.com>

www.baidu.com

## 图片

1. 手动添加：类似链接，前面需加！
2. 用鼠标拖图片进入，然后鼠标放置其上修改



```
![Alt text](/path/to/img.jpg)

![Alt text](/path/to/img.jpg "Optional title")
```

![Alt text](/path/to/img.jpg)

![Alt text](/path/to/img.jpg "Optional title")

## 斜体

以\*\*或\_\_括住，建议双*

```
*single asterisks*

_single underscores_
```

*single asterisks*

_single underscores_

## 加粗

开头双\*或双\_，结尾双\*或双\_，建议双\*

```
**double asterisks**

__double underscores__
```

**double asterisks**

__double underscores__

## 删除线

用两个~开头，两个~结尾

```
~~Mistaken text.~~
```

~~错误文字.~~

##下划线

使用HTML标签

```
<u>Underline</u>
```

<u>Underline</u>

## 代码

用两个`在正常段落总表示代码

```
Use the `printf()` function.
```

Use the `printf()` function.

## 数学式

需 `Preference` Panel -> `Markdown` Tab启动，

输入$，然后按ESC键，之后输入Tex命令，可预览

```
$\lim_{x\to\infty}\exp(-x)=0$
```

$\lim_{x\to\infty}\exp(-x)=0$

## 下标

需 `Preference` Panel -> `Markdown` Tab启动，

使用双~括住内容

```
H~2~O, X~long\ text~/
```

H~2~O`, `X~long\ text~

## 上标

需 `Preference` Panel -> `Markdown` Tab启动，

使用双^括住内容

```
X^2^
```

X^2^

## 高亮

需 `Preference` Panel -> `Markdown` Tab启动，

使用双==括住内容

```
==highlight==
```

==highlight==