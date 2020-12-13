## 基于内容的电影推荐：为用户产生TOP-N推荐结果



```python
# ......

user_profile = create_user_profile()

watch_record = pd.read_csv("datasets/ml-latest-small/ratings.csv", usecols=range(2),dtype={"userId": np.int32, "movieId": np.int32})

watch_record = watch_record.groupby("userId").agg(list)

for uid, interest_words in user_profile.items():
    result_table = {} # 电影id:[0.2,0.5,0.7]
    for interest_word, interest_weight in interest_words:
        related_movies = inverted_table[interest_word]
        for mid, related_weight in related_movies:
            _ = result_table.get(mid, [])
            _.append(interest_weight)    # 只考虑用户的兴趣程度
            # _.append(related_weight)    # 只考虑兴趣词与电影的关联程度
            # _.append(interest_weight*related_weight)    # 二者都考虑
            result_table.setdefault(mid, _)

    rs_result = map(lambda x: (x[0], sum(x[1])), result_table.items())
    rs_result = sorted(rs_result, key=lambda x:x[1], reverse=True)[:100]
    print(uid)
    pprint(rs_result)
    break
    
    # 历史数据  ==>  历史兴趣程度 ==>  历史推荐结果       离线推荐    离线计算
    # 在线推荐 ===>    娱乐(王思聪)   ===>   我 ==>  王思聪 100%  
    # 近线：最近1天、3天、7天           实时计算
```

