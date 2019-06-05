# LaTeX

[参考](https://liam.page/2014/09/08/latex-introduction/#TeX_家族)

## 安装

- 安装



- 配置

TeXworks 为我们预设了若干排版工具（pdfTeX, pdfLaTeX, XeTeX, XeLaTeX 等），本文主要用到其中的 **XeLaTeX**。关于这些排版工具的细节，讲解起来会有些复杂。因此此处按下不表，若有兴趣，可以参看[后文](https://liam.page/2014/09/08/latex-introduction/#TeX-家族)。当你对 TeX 系统相当熟悉之后，也可以不使用 TeXworks 预设的工具，自己配置排版工具。

TeXworks 默认的排版工具是 pdfLaTeX。如果你希望更改这个默认值，可以在*编辑 - 首选项 - 排版 - 处理工具 - 默认* 中修改。

## 文档

- 文字

```latex
\documentclass{article}
% 这里是导言区
\begin{document}
hello, word
\end{document}
```

中英文混排

```latex
\documentclass[UTF8]{ctexart}
\begin{document}
你好，world
\end{document}
```

- 组织文章

作者、标题、日期

```Latex 
\documentclass[UTF8]{ctexart}
\title{你好，world!}
\author{Liam}
\date{\today}
\begin{document}
\maketitle
你好，world!
\end{document}
```

章节、段落

```Latex 
\documentclass[UTF8]{ctexart}
\title{你好，world!}
\author{Liam}
\date{\today}
\begin{document}
\maketitle
\section{你好中国}
中国在East Asia.
\subsection{Hello Beijing}
北京是capital of China.
\subsubsection{Hello Dongcheng District}
\paragraph{Tian'anmen Square}
is in the center of Beijing
\subparagraph{Chairman Mao}
is in the center of 天安门广场。
\subsection{Hello 山东}
\paragraph{山东大学} is one of the best university in 山东。
\end{document}
```

目录

```latex
\documentclass[UTF8]{ctexart}
\title{你好，world!}
\author{Liam}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
\section{你好中国}
中国在East Asia.
\subsection{Hello Beijing}
北京是capital of China.
\subsubsection{Hello Dongcheng District}
\paragraph{Tian'anmen Square}
is in the center of Beijing
\subparagraph{Chairman Mao}
is in the center of 天安门广场。
\subsection{Hello 山东}
\paragraph{山东大学} is one of the best university in 山东。
\end{document}
```

## 公式

需要将导言区中的相关部分加上，就可以同时使用中文和编写数学公式了。为了使用 AMS-LaTeX 提供的数学功能，我们需要在导言区加载 `amsmath` 宏包

```latex
\usepackage{amsmath}
```

### 数学模式

LaTeX 的数学模式有两种：行内模式 (inline) 和行间模式 (display)。前者在正文的行文中，插入数学公式；后者独立排列单独成行，并自动居中。

在行文中，使用 `$ ... $` 可以插入行内公式，使用 `\[ ... \]` 可以插入行间公式，如果需要对行间公式进行编号，则可以使用 `equation` 环境：

```
\begin{equation}
...
\end{equation}
```

> 行内公式也可以使用 `\(...\)` 或者 `\begin{math} ... \end{math}` 来插入，但略显麻烦。
> 无编号的行间公式也可以使用 `\begin{displaymath} ... \end{displaymath}` 或者 `\begin{equation*} ... \end{equation*}` 来插入，但略显麻烦。（`equation*` 中的 `*` 表示环境不编号）
> 也有 plainTeX 风格的 `$$ ... $$` 来插入不编号的行间公式。但是在 LaTeX 中这样做会改变行文的默认行间距，不推荐。



## 图片



## 表格



