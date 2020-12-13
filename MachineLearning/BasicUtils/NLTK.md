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

## 语料库的使用
```python
import nltk
# 需要下载brown语料库
nltk.download('brown')

# nltk的都语料库包含在nltk.corpus中
from nltk.corpus import brown 
# 引用布朗大学的语料库

# 查看语料库包含的类别
print(brown.categories())

# 查看brown语料库
print('共有{}个句子'.format(len(brown.sents())))
print('共有{}个单词'.format(len(brown.words())))
```
## 分词

tokenize

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
nltk.download('punkt')

text = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."

# 句子解析器
from nltk.tokenize import sent_tokenize

sent_tokenize_list = sent_tokenize(text)
print("Sentence tokenizer:", sent_tokenize_list)

# 单词解析器
# 最基本的单词解析器
from nltk.tokenize import word_tokenize

print("Word tokenizer:", word_tokenize(text))

# Punktword单词解析器，以标点符号分割文本，如果是单词中的标点符号，则保留不做分割
from nltk.tokenize import PunktWordTokenizer

punkt_word_tokenizer = PunktWordTokenizer()
print("Punkt word tokenizer:", punkt_word_tokenizer.tokenize(text))

# wordPunct单词解析器，将标点符号保留到不同的句子标记中
from nltk.tokenize import WordPunctTokenizer

word_punct_tokenizer = WordPunctTokenizer()
print("Word punct tokenizer:", word_punct_tokenizer.tokenize(text))

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
- 同词不同形：look, looked, looking
- 影响语料学习的准确度
- 词形归一化
### 词干提取

stemming

- 英文

处理文本文档时，可能会碰到单词的不同形式。在文本分析中，提取这些单词的原形非常有用，有助于提取一些统计信息来分析整个文本。词干提取的目的是将不同词形的单词都变为其原形。词干提取使用启发式处理方法截取单词的尾部，以提取单词的原形。

NLTK中常用的stemmer：`PorterStemmer, SnowballStemmer, LancasterStemmer`，其中`Porter`提取规则最宽松，`Lancaster`提取规则最严格，会造成单词模糊难以理解，故常用`Snowball`.

```python
# PorterStemmer
from nltk.stem.porter import PorterStemmer

porter_stemmer = PorterStemmer()
print(porter_stemmer.stem('looked'))
print(porter_stemmer.stem('looking'))


# SnowballStemmer
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')
print(snowball_stemmer.stem('looked'))
print(snowball_stemmer.stem('looking'))


# LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer

lancaster_stemmer = LancasterStemmer()
print(lancaster_stemmer.stem('looked'))
print(lancaster_stemmer.stem('looking'))

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

### 词形归并

lemmatization

词形归并的目标也是将单词转换为其原形，但它是一个更结构化的方法。若用词干提取技术提取`wolves`，则结果`wolv`不是一个有意义的单词。词形归并通过对单词进行词汇和语法分析来实现，故可解决上述问题，得到结果`wolf`。

lemmatization，词形归并，将单词的各种词形归并成一种形式，如am, is, are -> be, went->go

NLTK中的lemma：`WordNetLemmatizer`，其中指明词性可以更准确地进行lemma

```python
import nltk
# 需要下载wordnet语料库
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer 


wordnet_lematizer = WordNetLemmatizer()
# lemmatize 默认为名词n
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
print(wordnet_lematizer.lemmatize('are', pos='v'))
print(wordnet_lematizer.lemmatize('went', pos='v'))

# 运行结果：
# be
# go
```

## 词性标注

- 英文

NLTK中的词性标注`nltk.word_tokenize()`

```python
import nltk

words = nltk.word_tokenize('Python is a widely used programming language.')
print(nltk.pos_tag(words)) # 需要下载 averaged_perceptron_tagger

# 运行结果：
# [('Python', 'NNP'), ('is', 'VBZ'), ('a', 'DT'), ('widely', 'RB'), ('used', 'VBN'), ('programming', 'NN'), ('language', 'NN'), ('.', '.')]
```

- 中文

jie在进程中文分词的同时，可以完成词性标注任务。根据分词结果中每个词的词性，可以初步实现命名实体识别，即将标注为nr的词视为人名，将标注为ns的词视为地名等。所有标点符号都被标注为x，因此可以根据这个方法去除分词结果中的标点符号

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

```python
import nltk
# 需要下载stopwords
nltk.download('stopwords')

from nltk.corpus import stopwords 

filtered_words = [word for word in words if word not in stopwords.words('english')]
print('原始词：', words)
print('去除停用词后：', filtered_words)

# 运行结果：
# 原始词： ['Python', 'is', 'a', 'widely', 'used', 'programming', 'language', '.']
# 去除停用词后： ['Python', 'widely', 'used', 'programming', 'language', '.']
```

## 分块划分文本

分块是指基于任意随机条件将输入文本分割成块。与标记解析不同的是，分块没有条件约束，分块的结果不需要有实际意义。当处理非常大的文本文档时，就需要将文本进行分块，以便进行下一步分析。

```python
import numpy as np
from nltk.corpus import brown

# 将文本分割成块
def splitter(data, num_words):
    words = data.split(' ')
    output = []

    # 初始化变量
    cur_count = 0
    cur_words = []
    # 对单词进行迭代
    for word in words:
        cur_words.append(word)
        cur_count += 1
        # 获得的单词数量与所需的单词数量相等时，重置相应变量
        if cur_count == num_words:
            output.append(' '.join(cur_words))
            cur_words = []
            cur_count = 0

    # 将块添加到输出变量列表的最后
    output.append(' '.join(cur_words) )

    return output 

if __name__=='__main__':
    # 从布朗语料库加载数据
    data = ' '.join(brown.words()[:10000])

    # 定义每块包含的单词数目 
    num_words = 1700

    chunks = []
    counter = 0
	# 调用分块逻辑
    text_chunks = splitter(data, num_words)
    print("Number of text chunks =", len(text_chunks))

```

## 词袋模型

如果要处理包含数百万单词的文本文档，需要将其转化成某种数值表示形式，以便让机器用这些数据来学习算法。这些算法需要数值数据，以便可以对这些数据进行分析，并输出有用的信息。这里需要用到词袋(bag-of-words)。词袋是从所有文档的所有单词中学习词汇的模型。学习之后，词袋通过构建文档中所有单词的直方图来对每篇文档进行建模

```python
import numpy as np
from nltk.corpus import brown
from chunking import splitter

if __name__ == '__main__':
    # 加载布朗语料库数据
    data = ' '.join(brown.words()[:10000])

    # 将文本按块划分
    num_words = 2000
    chunks = []
    counter = 0
    text_chunks = splitter(data, num_words)

    # 创建基于文本块的词典
    for text in text_chunks:
        chunk = {'index': counter, 'text': text}
        chunks.append(chunk)
        counter += 1

    # 提取一个文档-词矩阵：记录文档中每个单词出现的频次
    # 使用sklearn而不是nltk，由于sklearn更简洁
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(min_df=5, max_df=.95)
    doc_term_matrix = vectorizer.fit_transform([chunk['text'] for chunk in chunks])

    vocab = np.array(vectorizer.get_feature_names())
    print("Vocabulary:", vocab)

    print("Document term matrix:")
    chunk_names = ['Chunk-0', 'Chunk-1', 'Chunk-2', 'Chunk-3', 'Chunk-4']
    formatted_row = '{:>12}' * (len(chunk_names) + 1)  # 表格样式
    print(formatted_row.format('Word', *chunk_names))
    for word, item in zip(vocab, doc_term_matrix.T):
        # 'item'是压缩的稀疏矩阵(csr_matrix)数据结构 
        output = [str(x) for x in item.data]
        print(formatted_row.format(word, *output))

```

## 文本分类器

文本分类的目的是将文本文档分为不同的类，这里使用一种`tf-idf`的统计方法，表示词频-逆文档频率(`term frequency-inverse document frequency`)。这个统计工具有助于理解一个单词在一组文档中对某一个文档的重要性。可以作为特征向量来做文档分类。

```python
from sklearn.datasets import fetch_20newsgroups

# 创建一个类型列表，用词典映射的方式定义
category_map = {'misc.forsale': 'Sales', 'rec.motorcycles': 'Motorcycles',
                'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography',
                'sci.space': 'Space'}
# 基于定义的类型加载训练数据
training_data = fetch_20newsgroups(subset='train', categories=category_map.keys(), shuffle=True, random_state=7)

# 特征提取
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_termcounts = vectorizer.fit_transform(training_data.data)
print("Dimensions of training data:", X_train_termcounts.shape)

# 训练分类器
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

input_data = [
    "The curveballs of right handed pitchers tend to curve to the left",
    "Caesar cipher is an ancient form of encryption",
    "This two-wheeler is really good on slippery roads"
]

# 定义tf-idf变换器对象并训练 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_termcounts)

# 得到特征向量，使用该数据训练多项式朴素贝叶斯分类器 
classifier = MultinomialNB().fit(X_train_tfidf, training_data.target)
# 用词频统计转换输入数据
X_input_termcounts = vectorizer.transform(input_data)
# 用tf-idf变换器变换输入数据
X_input_tfidf = tfidf_transformer.transform(X_input_termcounts)

# 预测输入句子的输出类型 
predicted_categories = classifier.predict(X_input_tfidf)

for sentence, category in zip(input_data, predicted_categories):
    print('Input:', sentence, 'Predicted category:', category_map[training_data.target_names[category]])

```

## 性别识别

通过姓名识别性别，这里使用启发式方法，即姓名的最后几个字符可以界定性别特征。

```python
import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy


# 提取输入单词的特性
def gender_features(word, num_letters=2):
    return {'feature': word[-num_letters:].lower()}


if __name__ == '__main__':
    # 提取标记名称
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                     [(name, 'female') for name in names.words('female.txt')])

    # 设置随机种子，并混合搅乱训练数据
    random.seed(7)
    random.shuffle(labeled_names)
    input_names = ['Leonardo', 'Amy', 'Sam']

    # 搜索参数空间：由于不知需要多少个末尾字符，初步设置1～5
    for i in range(1, 5):
        print('Number of letters:', i)
        featuresets = [(gender_features(n, i), gender) for (n, gender) in labeled_names]
        # 训练集、测试集
        train_set, test_set = featuresets[500:], featuresets[:500]
        # 朴素贝叶斯分类
        classifier = NaiveBayesClassifier.train(train_set)  

        # 使用参数空间的每一个值评估分类器的效果
        print('Accuracy ==>', str(100 * nltk_accuracy(classifier, test_set)) + str('%'))

        # 预测
        for name in input_names:
            print(name, '==>', classifier.classify(gender_features(name, i)))

```

## 句子情感

情感分析是指确定一段歌诶定的文本是积极还是消极的过程。有一些场景中，会将"中性"作为第三个选项。情感分析常用于发现人们对于一个特定主题的看法。情感分析用于分析很多场景中用户的情绪，如营销活动、社交媒体、电子商务客户等。

```python
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


# 用于提取特征
def extract_features(word_list):
    return dict([(word, True) for word in word_list])


if __name__ == '__main__':
    # 加载积极与消极评论
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
    # 将评论分成积极和消极
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Negative') for f in negative_fileids]

    # 训练数据(80%)、测试数据
    threshold_factor = 0.8
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    # 提取特征
    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    print("Number of training datapoints:", len(features_train))
    print("Number of test datapoints:", len(features_test))

    # 训练朴素贝叶斯分类器
    classifier = NaiveBayesClassifier.train(features_train)
    print("Accuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))
    print("Top 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])

    # 输入一些评论进行预测
    input_reviews = [
        "It is an amazing movie",
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]

    print("Predictions:")
    for review in input_reviews:
        print("Review:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment), 2))

```

## 主题建模

主题建模指识别文本数据隐藏模式的过程，其目的是发现一组文档的隐藏主题结构。主题建模可以更好地组织文档，以便对这些文档进行分析。

主题建模通过识别文档中最有意义、最能表征主题的词来实现主题的分类。这些单词往往可以确定主题的内容。

```python
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
from nltk.corpus import stopwords


# 加载输入数据
def load_data(input_file):
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data.append(line[:-1])

    return data


# 类预处理文本
class Preprocessor(object):
    # 对各种操作进行初始化
    def __init__(self):
        # 创建正则表达式解析器，使用正则是因为只需要那些没有标点或其他标记的单词
        self.tokenizer = RegexpTokenizer(r'\w+')

        # 获取停用词列表，使用停用词可以减少干扰
        self.stop_words_english = stopwords.words('english')

        # 创建Snowball词干提取器
        self.stemmer = SnowballStemmer('english')

    # 标记解析、移除停用词、词干提取
    def process(self, input_text):
        # 标记解析(分词)
        tokens = self.tokenizer.tokenize(input_text.lower())

        # 移除停用词
        tokens_stopwords = [x for x in tokens if not x in self.stop_words_english]

        # 词干提取
        tokens_stemmed = [self.stemmer.stem(x) for x in tokens_stopwords]

        return tokens_stemmed


if __name__ == '__main__':
    input_file = 'data_topic_modeling.txt'

    data = load_data(input_file)

    # 创建预处理对象
    preprocessor = Preprocessor()

    # 创建一组经过预处理的文档
    processed_tokens = [preprocessor.process(x) for x in data]

    # 创建基于标记文档的词典
    dict_tokens = corpora.Dictionary(processed_tokens)

    # 创建文档-词矩阵
    corpus = [dict_tokens.doc2bow(text) for text in processed_tokens]

    # 假设文档可分为2个主题，使用隐含狄利克雷分布(LDA)做主题建模
    # 基于刚刚创建的语料库生成LDA模型
    num_topics = 2
    num_words = 4
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dict_tokens, passes=25)

    print("Most contributing words to the topics:")
    for item in ldamodel.print_topics(num_topics=num_topics, num_words=num_words):
        print("Topic", item[0], "==>", item[1])

```

