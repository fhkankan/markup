# NLTK

[官网](http://www.nltk.org/index.html)

NTLK是著名的Python自然语言处理工具包，但是主要针对的是英文处理。

应用：

文本提取、词汇切、词频分析、词袋模型、情感分析

## 安装步骤

[参考](http://nltk.org/install.html)

- 下载NLTK包 
```
pip install nltk
```
- 运行Python，并输入下面的指令
```python
 import nltk
 nltk.download()  # 下载语料库
```
 - 在弹出的窗口，建议安装所有的包

# 自然语言文本处理流程
## 语料库的使用
nltk的都语料库包含在 nltk.corpus 中
```python
import nltk
# 需要下载brown语料库
from nltk.corpus import brown 
# 引用布朗大学的语料库

# 查看语料库包含的类别
print(brown.categories())

# 查看brown语料库
print('共有{}个句子'.format(len(brown.sents())))
print('共有{}个单词'.format(len(brown.words())))
```
## 分词(tokenize)

```
- 将句子拆分成 具有语言语义学上有意义的词
- 中、英文分词区别：
  - 英文中，单词之间是以空格作为自然分界符的
  - 中文中没有一个形式上的分界符，分词比英文复杂的多
- 中文分词工具，如：结巴分词 pip install jieba
- 得到分词结果后，中英文的后续处理没有太大区别
```

- 英文分词

```python
import nltk
# 需要事先安装 punkt 分词模型

text = "Python is a high-level programming language, and i like it!"

# 对文本进行分词
seg_list = nltk.word_tokenize(text)

print(seg_list)

# 运行结果：
# ['Python', 'is', 'a', 'high-level', 'programming', 'language', '!']
```
- 中文分词

```python
# 导入jieba分词
import jieba

# 全模式
seg_list = jieba.cut("我来到清华大学", cut_all=True)
print("全模式: " + "/ ".join(seg_list))  
# 精确模式
seg_list = jieba.cut("我来到清华大学", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))  
# 搜索引擎模式
seg_list = jieba.cut_for_search("小明硕士毕业于中国科技大学，后在美国斯坦福大学深造")
print("搜索引擎模式: "+"/".join(seg_list))
```
## 词形问题
- look, looked, looking
- 影响语料学习的准确度
- 词形归一化
### 词干提取(stemming)

- 英文

stemming，词干提取，如将ing, ed去掉，只保留单词主干

NLTK中常用的stemmer：

>PorterStemmer, SnowballStemmer, LancasterStemmer
```python
# PorterStemmer
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('looked'))
print(porter_stemmer.stem('looking'))

# 运行结果：
# look
# look

# SnowballStemmer
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')
print(snowball_stemmer.stem('looked'))
print(snowball_stemmer.stem('looking'))

# 运行结果：
# look
# look

# LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer

lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('looked'))
print(lancaster_stemmer.stem('looking'))

# 运行结果：
# look
# look
```
- 中文关键词提取

jeiba实现了TF_IDF和TextRank两种关键词提取算法，直接调用即可。这里的关键词前提是中文分词，会使用jieba自带的前缀词典和IDF权重字典

```python
import jieba.analyse

# 字符串前面加u表示使用unicode编码
content = u'十八大以来，国内外形势变化和我国各项事业发展都给我们提出了一个重大时代课题，这就是必须从理论和实践结合上系统回答新时代坚持和发展什么样的中国特色社会主义、怎样坚持和发展中国特色社会主义，包括新时代坚持和发展中国特色社会主义的总目标、总任务、总体布局、战略布局和发展方向、发展方式、发展动力、战略步骤、外部条件、政治保证等基本问题，并且要根据新的实践对经济、政治、法治、科技、文化、教育、民生、民族、宗教、社会、生态文明、国家安全、国防和军队、“一国两制”和祖国统一、统一战线、外交、党的建设等各方面作出理论分析和政策指导，以利于更好坚持和发展中国特色社会主义。'

# 参数1：待提取关键词的文本，参数2：返回关键词的数量，重要性从高到低排序
# 参数3：是否同时返回每个关键词的权重，参数4：词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词
keywords = jieba.analyse.extract_tags(content, topK=20, withWeight=True, allowPOS=())
# 访问提取结果
for item in keywords:
	# 分别为关键词和响应的权重
    print item[0], item[1]
    
# 同样式四个参数，但allowPOS默认为('ns','n','vn','v'),即仅提取地名、名词、动名词、动词
keywords = jieba.analyse.textrank(content, topK=20, withWeight=True, allowPOS=('ns','n','vn','v'))
# 访问提取结果
for item in keywords:
    # 分别为关键词和响应的权重
    print item[0], item[1]
```



### 词形归并(lemmatization) 

- lemmatization，词形归并，将单词的各种词形归并成一种形式，如am, is, are -> be, went->go
- NLTK中的lemma

> WordNetLemmatizer

- 指明词性可以更准确地进行lemma

> went 动词 -> go， 走
>
> Went 名词 -> Went，文特

```
# WordNetLemmatizer 示例：
from nltk.stem import WordNetLemmatizer 
# 需要下载wordnet语料库

wordnet_lematizer = WordNetLemmatizer()
print(wordnet_lematizer.lemmatize('cats'))
print(wordnet_lematizer.lemmatize('boxes'))
print(wordnet_lematizer.lemmatize('are'))
print(wordnet_lematizer.lemmatize('went'))

# 运行结果：
# cat
# box
# are
# went


# 指明词性可以更准确地进行lemma
# lemmatize 默认为名词
print(wordnet_lematizer.lemmatize('are', pos='v'))
print(wordnet_lematizer.lemmatize('went', pos='v'))

# 运行结果：
# be
# go
```

## 词性标注

- 英文

NLTK中的词性标注

> nltk.word_tokenize()

```python
import nltk

words = nltk.word_tokenize('Python is a widely used programming language.')
print(nltk.pos_tag(words)) # 需要下载 averaged_perceptron_tagger

# 运行结果：
# [('Python', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('widely', 'RB'), ('used', 'VBN'), ('programming', 'NN'), ('language', 'NN'), ('.', '.')]
```

- 中文

jie在进程中文分词的同时，可以完成词性标注任务。根据分词结果中每个词的词性，可以初步实现命名实体是被，即将标注为nr的词视为人名，将标注为ns的词视为地名等。所有标点符号都被标注为x，因此可以根据这个方法去除分词结果中的标点符号

```python
# 加载jie.posseg并取个别名，方便调用
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    # 格式化模板并传入参数
    print('%s, %s' % (word, flag)) 
```



## 去除停用词

- 为节省存储空间和提高搜索效率，NLP中会自动过滤掉某些字或词

- 停用词都是人工输入、非自动化生成的，形成停用词表

- 分类

  > 语言中的功能词，如the, is…
  >
  > 词汇词，通常是使用广泛的词，如want

- 中文停用词表

  > 中文停用词库
  >
  > 哈工大停用词表
  >
  > 四川大学机器智能实验室停用词库
  >
  > 百度停用词列表

- 其他语言停用词表

  > <http://www.ranks.nl/stopwords>

- 使用NLTK去除停用词

  > stopwords.words()

```
from nltk.corpus import stopwords # 需要下载stopwords

filtered_words = [word for word in words if word not in stopwords.words('english')]
print('原始词：', words)
print('去除停用词后：', filtered_words)

# 运行结果：
# 原始词： ['Python', 'is', 'a', 'widely', 'used', 'programming', 'language', '.']
# 去除停用词后： ['Python', 'widely', 'used', 'programming', 'language', '.']
```

# 典型的文本预处理流程

原始文本--->分词--->[词性标注]--->词形归一化--->去除停用词--->处理好的单词列表

```python
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 原始文本
# '生活就像一盒巧克力. 你永远也不知道下一个你会拿到什么.' ——《阿甘正传》经典台词
raw_text = 'Life is like a box of chocolates. You never know what you\'re gonna get.'

# 分词
raw_words = nltk.word_tokenize(raw_text)

# 词形归一化
wordnet_lematizer = WordNetLemmatizer()
words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

# 去除停用词
filtered_words = [word for word in words if word not in stopwords.words('english')]

print('原始文本：', raw_text)
print('预处理结果：', filtered_words)
```

## NLTK 中的分句与分词

```
import nltk

text = "The first time I heard that song was in Hawaii on radio.  I was just a kid, and loved it very much! What a fantastic song!"  

# 分句
sents = nltk.sent_tokenize(text)
print(sents)

# 按小写字母 分词
words = nltk.word_tokenize(text.lower()))
print(words)
```

# 词嵌入

词嵌入(Word Embedding)可以将文本和词语转换为机器能够接受的数值向量

- 语言的表示

> 符号主义

```
符号主义中典型的代表是Bag of words，即词袋模型。基于词袋模型可以方便地用一个N维向量表示任何一句话，每个维度的值即对应的词出现的次数。

优点：简单
缺点：当词典中词的数量增大时，向量的维度将随之增大；无论是词还是句子的表示，向量过于稀疏，除了少数维度外其他维度均为0；每个词对应的向量在空间上都两两正交，任意一堆向量之间的内积等数值特征为0，无法表达词语之间的语义关联和差异；句子的向量表示丢失了词序特征，"我很不高兴"和“不我很高兴”对应的向量相同
```

> 分布式表示

```
分布式表示的典型代表是Word Embedding，即词嵌入。使用低维、稠密、实值得词向量来表示每一个词，从而赋予词语丰富的语义含义，并使得计算词语相关度成为可能。两个词具有语义相关或相似，则它们对应得词向量之间距离相近，度量向量之间的距离可以使用经典的欧拉距离和余弦相似度等。

词嵌入可以将词典中的每个词映射成对应的词向量，好的词嵌入模型具有：相关性好，类比关联
```

- 训练词向量

词向量的训练主要是基于无监督学习，从大量文本语料中学习出每个词的最佳词向量。训练的核心思想是，语义相关或相似的词语，大多具有相似的上下文，即它们经常在相似的语境中出现。

词嵌入模型中的典型代表是Word2Vec

- 代码实现

gensim是开源python工具包，用于从非结构化文本中无监督地学习文本隐层的主题向量表示，支持包括TF-IDF,LSA,LDA和Word2Vec在内的多种主题模型算法，并提供了诸如相似度计算、信息检索等常用任务的API接口。

```python
# 加载包
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 训练模型
sentences = LineSentence('wiki.zh.word.text')
# size词向量的维度，window上下文环境的窗口大小，min_count忽略出现次数低于min_count的词
model = Word2Vec(sentences, size=128, window=5， min_count=5, workers=4)

# 保存模型
model.save('word_embedding_128')

# 若已经保存过模型，则直接加载即可
# 训练及保存代码可省略
# model = Word2Vec.load('word_embedding_128')

# 使用模型
# 返回一个词语最相关的多个词语及对应的相关度
items = model.most_similar(u'中国')
for item in items:
    # 词的内容，词的相关度
    print item[0], item[1]
# 返回连个词语之间的相关度
model.similarity(u'男人', u'女人')
```

