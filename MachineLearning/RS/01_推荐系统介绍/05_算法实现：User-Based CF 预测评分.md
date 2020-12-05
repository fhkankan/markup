## 案例--算法实现：User-Based CF 预测评分

**评分预测公式：**
$$
pred(u,i)=\hat{r}_{ui}=\cfrac{\sum_{v\in U}sim(u,v)*r_{vi}}{\sum_{v\in U}|sim(u,v)|}
$$
#### 算法实现

- 实现评分预测方法：`predict`

  ```python
  # ......
  
  def predict(uid, iid, ratings_matrix, user_similar):
      '''
      预测给定用户对给定物品的评分值
      :param uid: 用户ID
      :param iid: 物品ID
      :param ratings_matrix: 用户-物品评分矩阵
      :param user_similar: 用户两两相似度矩阵
      :return: 预测的评分值
      '''
      print("开始预测用户<%d>对电影<%d>的评分..."%(uid, iid))
      # 1. 找出uid用户的相似用户
      similar_users = user_similar[uid].drop([uid]).dropna()
      # 相似用户筛选规则：正相关的用户
      similar_users = similar_users.where(similar_users>0).dropna()
      if similar_users.empty is True:
          raise Exception("用户<%d>没有相似的用户" % uid)
  
      # 2. 从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
      ids = set(ratings_matrix[iid].dropna().index)&set(similar_users.index)
      finally_similar_users = similar_users.ix[list(ids)]
  
      # 3. 结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
      sum_up = 0    # 评分预测公式的分子部分的值
      sum_down = 0    # 评分预测公式的分母部分的值
      for sim_uid, similarity in finally_similar_users.iteritems():
          # 近邻用户的评分数据
          sim_user_rated_movies = ratings_matrix.ix[sim_uid].dropna()
          # 近邻用户对iid物品的评分
          sim_user_rating_for_item = sim_user_rated_movies[iid]
          # 计算分子的值
          sum_up += similarity * sim_user_rating_for_item
          # 计算分母的值
          sum_down += similarity
  
      # 计算预测的评分值并返回
      predict_rating = sum_up/sum_down
      print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
      return round(predict_rating, 2)
  
  if __name__ == '__main__':
      ratings_matrix = load_data(DATA_PATH)
  
      user_similar = compute_pearson_similarity(ratings_matrix, based="user")
      # 预测用户1对物品1的评分
      predict(1, 1, ratings_matrix, user_similar)
      # 预测用户1对物品2的评分
      predict(1, 2, ratings_matrix, user_similar)
  ```

- 实现预测全部评分方法：`predict_all`

  ```python
  # ......
  def predict_all(uid, ratings_matrix, user_similar):
      '''
      预测全部评分
      :param uid: 用户id
      :param ratings_matrix: 用户-物品打分矩阵
      :param user_similar: 用户两两间的相似度
      :return: 生成器，逐个返回预测评分
      '''
      # 准备要预测的物品的id列表
      item_ids = ratings_matrix.columns
      # 逐个预测
      for iid in item_ids:
          try:
              rating = predict(uid, iid, ratings_matrix, user_similar)
          except Exception as e:
              print(e)
          else:
              yield uid, iid, rating
  
  if __name__ == '__main__':
      ratings_matrix = load_data(DATA_PATH)
  
      user_similar = compute_pearson_similarity(ratings_matrix, based="user")
  
      for i in predict_all(1, ratings_matrix, user_similar):
          pass
  ```

- 添加过滤规则

  ```python
  def _predict_all(uid, item_ids, ratings_matrix, user_similar):
      '''
      预测全部评分
      :param uid: 用户id
      :param item_ids: 要预测的物品id列表
      :param ratings_matrix: 用户-物品打分矩阵
      :param user_similar: 用户两两间的相似度
      :return: 生成器，逐个返回预测评分
      '''
      # 逐个预测
      for iid in item_ids:
          try:
              rating = predict(uid, iid, ratings_matrix, user_similar)
          except Exception as e:
              print(e)
          else:
              yield uid, iid, rating
  
  def predict_all(uid, ratings_matrix, user_similar, filter_rule=None):
      '''
      预测全部评分，并可根据条件进行前置过滤
      :param uid: 用户ID
      :param ratings_matrix: 用户-物品打分矩阵
      :param user_similar: 用户两两间的相似度
      :param filter_rule: 过滤规则，只能是四选一，否则将抛异常："unhot","rated",["unhot","rated"],None
      :return: 生成器，逐个返回预测评分
      '''
  
      if not filter_rule:
          item_ids = ratings_matrix.columns
      elif isinstance(filter_rule, str) and filter_rule == "unhot":
          '''过滤非热门电影'''
          # 统计每部电影的评分数
          count = ratings_matrix.count()
          # 过滤出评分数高于10的电影，作为热门电影
          item_ids = count.where(count>10).dropna().index
      elif isinstance(filter_rule, str) and filter_rule == "rated":
          '''过滤用户评分过的电影'''
          # 获取用户对所有电影的评分记录
          user_ratings = ratings_matrix.ix[uid]
          # 评分范围是1-5，小于6的都是评分过的，除此以外的都是没有评分的
          _ = user_ratings<6
          item_ids = _.where(_==False).dropna().index
      elif isinstance(filter_rule, list) and set(filter_rule) == set(["unhot", "rated"]):
          '''过滤非热门和用户已经评分过的电影'''
          count = ratings_matrix.count()
          ids1 = count.where(count > 10).dropna().index
  
          user_ratings = ratings_matrix.ix[uid]
          _ = user_ratings < 6
          ids2 = _.where(_ == False).dropna().index
          # 取二者交集
          item_ids = set(ids1)&set(ids2)
      else:
          raise Exception("无效的过滤参数")
  
      yield from _predict_all(uid, item_ids, ratings_matrix, user_similar)
  
  if __name__ == '__main__':
      ratings_matrix = load_data(DATA_PATH)
  
      user_similar = compute_pearson_similarity(ratings_matrix, based="user")
  
      for result in predict_all(1, ratings_matrix, user_similar, filter_rule=["unhot", "rated"]):
          print(result)
  ```

- 根据预测评分为指定用户进行TOP-N推荐：

  ```python
  # ......
  
  def top_k_rs_result(k):
      ratings_matrix = load_data(DATA_PATH)
      user_similar = compute_pearson_similarity(ratings_matrix, based="user")
      results = predict_all(1, ratings_matrix, user_similar, filter_rule=["unhot", "rated"])
      return sorted(results, key=lambda x: x[2], reverse=True)[:k]
  
  if __name__ == '__main__':
      from pprint import pprint
      result = top_k_rs_result(20)
      pprint(result)
  
  ```

  