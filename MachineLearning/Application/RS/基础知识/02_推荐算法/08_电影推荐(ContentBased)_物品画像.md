## 基于内容的电影推荐：物品画像



物品画像构建步骤：

- 利用tags.csv中每部电影的标签作为电影的候选关键词
- 利用TF·IDF计算每部电影的标签的tfidf值，选取TOP-N个关键词作为电影画像标签
- 将电影的分类词直接作为每部电影的画像标签

## 基于TF-IDF的特征提取技术

前面提到，物品画像的特征标签主要都是指的如电影的导演、演员、图书的作者、出版社等结构话的数据，也就是他们的特征提取，尤其是体征向量的计算是比较简单的，如直接给作品的分类定义0或者1的状态。

但另外一些特征，比如电影的内容简介、电影的影评、图书的摘要等文本数据，这些被称为非结构化数据，首先他们本应该也属于物品的一个特征标签，但是这样的特征标签进行量化时，也就是计算它的特征向量时是很难去定义的。

因此这时就需要借助一些自然语言处理、信息检索等技术，将如用户的文本评论或其他文本内容信息的非结构化数据进行量化处理，从而实现更加完善的物品画像/用户画像。

TF-IDF算法便是其中一种在自然语言处理领域中应用比较广泛的一种算法。可用来提取目标文档中，并得到关键词用于计算对于目标文档的权重，并将这些权重组合到一起得到特征向量。

#### 算法原理

TF-IDF自然语言处理领域中计算文档中词或短语的权值的方法，是**词频**（Term Frequency，TF）和逆转文档频率（Inverse Document Frequency，IDF）的乘积。TF指的是某一个给定的词语在该文件中出现的次数。这个数字通常会被正规化，以防止它偏向长的文件（同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否）。IDF是一个词语普遍重要性的度量，某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。

TF-IDF算法基于一个这样的假设：若一个词语在目标文档中出现的频率高而在其他文档中出现的频率低，那么这个词语就可以用来区分出目标文档。这个假设需要掌握的有两点：
- 在本文档出现的频率高；
- 在其他文档出现的频率低。

因此，TF-IDF算法的计算可以分为词频（Term Frequency，TF）和逆转文档频率（Inverse Document Frequency，IDF）两部分，由TF和IDF的乘积来设置文档词语的权重。

TF指的是一个词语在文档中的出现频率。假设文档集包含的文档数为$$N$$，文档集中包含关键词$$k_i$$的文档数为$$n_i$$，$$f_{ij}$$表示关键词$$k_i$$在文档$$d_j$$中出现的次数，$$f_{dj}$$表示文档$$d_j$$中出现的词语总数，$$k_i$$在文档dj中的词频$$TF_{ij}$$定义为：$$TF_{ij}=\frac {f_{ij}}{f_{dj}}$$。并且注意，这个数字通常会被正规化，以防止它偏向长的文件（指同一个词语在长文件里可能会比短文件有更高的词频，而不管该词语重要与否）。

IDF是一个词语普遍重要性的度量。表示某一词语在整个文档集中出现的频率，由它计算的结果取对数得到关键词$$k_i$$的逆文档频率$$IDF_i$$：$$IDF_i=log\frac {N}{n_i}$$

由TF和IDF计算词语的权重为：$$w_{ij}=TF_{ij}$$**·**$$IDF_{i}=\frac {f_{ij}}{f_{dj}}$$**·**$$log\frac {N}{n_i}$$

**结论：TF-IDF与词语在文档中的出现次数成正比，与该词在整个文档集中的出现次数成反比。**

**用途：在目标文档中，提取关键词(特征标签)的方法就是将该文档所有词语的TF-IDF计算出来并进行对比，取其中TF-IDF值最大的k个数组成目标文档的特征向量用以表示文档。**

注意：文档中存在的停用词（Stop Words），如“是”、“的”之类的，对于文档的中心思想表达没有意义的词，在分词时需要先过滤掉再计算其他词语的TF-IDF值。

#### 算法举例

对于计算影评的TF-IDF，以电影“加勒比海盗：黑珍珠号的诅咒”为例，假设它总共有1000篇影评，其中一篇影评的总词语数为200，其中出现最频繁的词语为“海盗”、“船长”、“自由”，分别是20、15、10次，并且这3个词在所有影评中被提及的次数分别为1000、500、100，就这3个词语作为关键词的顺序计算如下。

1. 将影评中出现的停用词过滤掉，计算其他词语的词频。以出现最多的三个词为例进行计算如下：
    - “海盗”出现的词频为20/200＝0.1
    - “船长”出现的词频为15/200=0.075
    - “自由”出现的词频为10/200=0.05；

2. 计算词语的逆文档频率如下：
    - “海盗”的IDF为：log(1000/1000)=0
    - “船长”的IDF为：log(1000/500)=0.3
      “自由”的IDF为：log(1000/100)=1
3. 由1和2计算的结果求出词语的TF-IDF结果，“海盗”为0，“船长”为0.0225，“自由”为0.05。

通过对比可得，该篇影评的关键词排序应为：“自由”、“船长”、“海盗”。把这些词语的TF-IDF值作为它们的权重按照对应的顺序依次排列，就得到这篇影评的特征向量，我们就用这个向量来代表这篇影评，向量中每一个维度的分量大小对应这个属性的重要性。

将总的影评集中所有的影评向量与特定的系数相乘求和，得到这部电影的综合影评向量，与电影的基本属性结合构建视频的物品画像，同理构建用户画像，可采用多种方法计算物品画像和用户画像之间的相似度，为用户做出推荐。
    

#### 加载数据集

```python
import pandas as pd
import numpy as np
'''
- 利用tags.csv中每部电影的标签作为电影的候选关键词
- 利用TF·IDF计算每部电影的标签的tfidf值，选取TOP-N个关键词作为电影画像标签
- 并将电影的分类词直接作为每部电影的画像标签
'''

def get_movie_dataset():
    # 加载基于所有电影的标签
    # all-tags.csv来自ml-latest数据集中
    # 由于ml-latest-small中标签数据太多，因此借助其来扩充
    _tags = pd.read_csv("datasets/ml-latest-small/all-tags.csv", usecols=range(1, 3)).dropna()
    tags = _tags.groupby("movieId").agg(list)

    # 加载电影列表数据集
    movies = pd.read_csv("datasets/ml-latest-small/movies.csv", index_col="movieId")
    # 将类别词分开
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
    # 为每部电影匹配对应的标签数据，如果没有将会是NAN
    movies_index = set(movies.index) & set(tags.index)
    new_tags = tags.loc[list(movies_index)]
    ret = movies.join(new_tags)

    # 构建电影数据集，包含电影Id、电影名称、类别、标签四个字段
    # 如果电影没有标签数据，那么就替换为空列表
    # map(fun,可迭代对象)
    movie_dataset = pd.DataFrame(
        map(
            lambda x: (x[0], x[1], x[2], x[2]+x[3]) if x[3] is not np.nan else (x[0], x[1], x[2], []), ret.itertuples())
        , columns=["movieId", "title", "genres","tags"]
    )

    movie_dataset.set_index("movieId", inplace=True)
    return movie_dataset

movie_dataset = get_movie_dataset()
print(movie_dataset)
```

#### 基于TF·IDF提取TOP-N关键词，构建电影画像

```python
from gensim.models import TfidfModel

import pandas as pd
import numpy as np

from pprint import pprint

# ......

def create_movie_profile(movie_dataset):
    '''
    使用tfidf，分析提取topn关键词
    :param movie_dataset: 
    :return: 
    '''
    dataset = movie_dataset["tags"].values

    from gensim.corpora import Dictionary
    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据将每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TF-IDF值
    model = TfidfModel(corpus)

    movie_profile = {}
    for i, mid in enumerate(movie_dataset.index):
        # 根据每条数据返回，向量
        vector = model[corpus[i]]
        # 按照TF-IDF值得到top-n的关键词
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        # 根据关键词提取对应的名称
        movie_profile[mid] = dict(map(lambda x:(dct[x[0]], x[1]), movie_tags))

    return movie_profile

movie_dataset = get_movie_dataset()
pprint(create_movie_profile(movie_dataset))
```

#### 完善画像关键词

```python
from gensim.models import TfidfModel

import pandas as pd
import numpy as np

from pprint import pprint

# ......

def create_movie_profile(movie_dataset):
    '''
    使用tfidf，分析提取topn关键词
    :param movie_dataset:
    :return:
    '''
    dataset = movie_dataset["tags"].values

    from gensim.corpora import Dictionary
    # 根据数据集建立词袋，并统计词频，将所有词放入一个词典，使用索引进行获取
    dct = Dictionary(dataset)
    # 根据将每条数据，返回对应的词索引和词频
    corpus = [dct.doc2bow(line) for line in dataset]
    # 训练TF-IDF模型，即计算TF-IDF值
    model = TfidfModel(corpus)

    _movie_profile = []
    for i, data in enumerate(movie_dataset.itertuples()):
        mid = data[0]
        title = data[1]
        genres = data[2]
        vector = model[corpus[i]]
        movie_tags = sorted(vector, key=lambda x: x[1], reverse=True)[:30]
        topN_tags_weights = dict(map(lambda x: (dct[x[0]], x[1]), movie_tags))
        # 将类别词的添加进去，并设置权重值为1.0
        for g in genres:
            topN_tags_weights[g] = 1.0
        topN_tags = [i[0] for i in topN_tags_weights.items()]
        _movie_profile.append((mid, title, topN_tags, topN_tags_weights))

    movie_profile = pd.DataFrame(_movie_profile, columns=["movieId", "title", "profile", "weights"])
    movie_profile.set_index("movieId", inplace=True)
    return movie_profile

movie_dataset = get_movie_dataset()
pprint(create_movie_profile(movie_dataset))

```

为了根据指定关键词迅速匹配到对应的电影，因此需要对物品画像的标签词，建立**倒排索引**

**倒排索引介绍**

通常数据存储数据，都是以物品的ID作为索引，去提取物品的其他信息数据

而倒排索引就是用物品的其他数据作为索引，去提取它们对应的物品的ID列表

```python
# ......

'''
建立tag-物品的倒排索引
'''

def create_inverted_table(movie_profile):
    inverted_table = {}
    for mid, weights in movie_profile["weights"].iteritems():
        for tag, weight in weights.items():
            #到inverted_table dict 用tag作为Key去取值 如果取不到就返回[]
            _ = inverted_table.get(tag, [])
            _.append((mid, weight))
            inverted_table.setdefault(tag, _)
    return inverted_table

inverted_table = create_inverted_table(movie_profile)
pprint(inverted_table)
```

