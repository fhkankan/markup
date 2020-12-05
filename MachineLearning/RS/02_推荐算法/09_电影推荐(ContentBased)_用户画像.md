## 基于内容的电影推荐：用户画像



用户画像构建步骤：

- 根据用户的评分历史，结合物品画像，将有观影记录的电影的画像标签作为初始标签反打到用户身上
- 通过对用户观影标签的次数进行统计，计算用户的每个初始标签的权重值，排序后选取TOP-N作为用户最终的画像标签

#### 用户画像建立

```python
import pandas as pd
import numpy as np
from gensim.models import TfidfModel

from functools import reduce
import collections

from pprint import pprint

# ......

'''
user profile画像建立：
1. 提取用户观看列表
2. 根据观看列表和物品画像为用户匹配关键词，并统计词频
3. 根据词频排序，最多保留TOP-k个词，这里K设为100，作为用户的标签
'''

def create_user_profile():
    watch_record = pd.read_csv("datasets/ml-latest-small/ratings.csv", usecols=range(2), dtype={"userId":np.int32, "movieId": np.int32})

    watch_record = watch_record.groupby("userId").agg(list)
    # print(watch_record)

    movie_dataset = get_movie_dataset()
    movie_profile = create_movie_profile(movie_dataset)

    user_profile = {}
    for uid, mids in watch_record.itertuples():
        record_movie_prifole = movie_profile.loc[list(mids)]
        counter = collections.Counter(reduce(lambda x, y: list(x)+list(y), record_movie_prifole["profile"].values))
        # 兴趣词
        interest_words = counter.most_common(50)
        maxcount = interest_words[0][1]
        interest_words = [(w,round(c/maxcount, 4)) for w,c in interest_words]
        user_profile[uid] = interest_words

    return user_profile

user_profile = create_user_profile()
pprint(user_profile)
```

