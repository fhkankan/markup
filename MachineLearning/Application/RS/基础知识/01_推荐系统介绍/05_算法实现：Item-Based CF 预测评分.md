## 案例--算法实现：Item-Based CF 预测评分

**评分预测公式：**
$$
pred(u,i)=\hat{r}_{ui}=\cfrac{\sum_{j\in I_{rated}}sim(i,j)*r_{uj}}{\sum_{j\in I_{rated}}sim(i,j)}
$$
#### 算法实现

- 实现评分预测方法：`predict`

  - 方法说明：

    利用原始评分矩阵、以及物品间两两相似度，预测指定用户对指定物品的评分。

    如果无法预测，则抛出异常

  ```python
  # ......
  
  def predict(uid, iid, ratings_matrix, item_similar):
      '''
      预测给定用户对给定物品的评分值
      :param uid: 用户ID
      :param iid: 物品ID
      :param ratings_matrix: 用户-物品评分矩阵
      :param item_similar: 物品两两相似度矩阵
      :return: 预测的评分值
      '''
      print("开始预测用户<%d>对电影<%d>的评分..."%(uid, iid))
      # 1. 找出iid物品的相似物品
      similar_items = item_similar[iid].drop([iid]).dropna()
      # 相似物品筛选规则：正相关的物品
      similar_items = similar_items.where(similar_items>0).dropna()
      if similar_items.empty is True:
          raise Exception("物品<%d>没有相似的物品" %id)
  
      # 2. 从iid物品的近邻相似物品中筛选出uid用户评分过的物品
      ids = set(ratings_matrix.ix[uid].dropna().index)&set(similar_items.index)
      finally_similar_items = similar_items.ix[list(ids)]
  
      # 3. 结合iid物品与其相似物品的相似度和uid用户对其相似物品的评分，预测uid对iid的评分
      sum_up = 0    # 评分预测公式的分子部分的值
      sum_down = 0    # 评分预测公式的分母部分的值
      for sim_iid, similarity in finally_similar_items.iteritems():
          # 近邻物品的评分数据
          sim_item_rated_movies = ratings_matrix[sim_iid].dropna()
          # uid用户对相似物品物品的评分
          sim_item_rating_from_user = sim_item_rated_movies[uid]
          # 计算分子的值
          sum_up += similarity * sim_item_rating_from_user
          # 计算分母的值
          sum_down += similarity
  
      # 计算预测的评分值并返回
      predict_rating = sum_up/sum_down
      print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
      return round(predict_rating, 2)
  
  if __name__ == '__main__':
      ratings_matrix = load_data(DATA_PATH)
  
      item_similar = compute_pearson_similarity(ratings_matrix, based="item")
      # 预测用户1对物品1的评分
      predict(1, 1, ratings_matrix, item_similar)
      # 预测用户1对物品2的评分
      predict(1, 2, ratings_matrix, item_similar)
  ```

- 实现预测全部评分方法：`predict_all`

  ```python
  # ......
  
  def predict_all(uid, ratings_matrix, item_similar):
      '''
      预测全部评分
      :param uid: 用户id
      :param ratings_matrix: 用户-物品打分矩阵
      :param item_similar: 物品两两间的相似度
      :return: 生成器，逐个返回预测评分
      '''
      # 准备要预测的物品的id列表
      item_ids = ratings_matrix.columns
      # 逐个预测
      for iid in item_ids:
          try:
              rating = predict(uid, iid, ratings_matrix, item_similar)
          except Exception as e:
              print(e)
          else:
              yield uid, iid, rating
  
  if __name__ == '__main__':
      ratings_matrix = load_data(DATA_PATH)
  
      item_similar = compute_pearson_similarity(ratings_matrix, based="item")
      for i in predict_all(1, ratings_matrix, item_similar):
          pass
  ```

- 添加过滤规则

  ```python
  def _predict_all(uid, item_ids,ratings_matrix, item_similar):
      '''
      预测全部评分
      :param uid: 用户id
      :param item_ids: 要预测物品id列表
      :param ratings_matrix: 用户-物品打分矩阵
      :param item_similar: 物品两两间的相似度
      :return: 生成器，逐个返回预测评分
      '''
      # 逐个预测
      for iid in item_ids:
          try:
              rating = predict(uid, iid, ratings_matrix, item_similar)
          except Exception as e:
              print(e)
          else:
              yield uid, iid, rating
  
  def predict_all(uid, ratings_matrix, item_similar, filter_rule=None):
      '''
      预测全部评分，并可根据条件进行前置过滤
      :param uid: 用户ID
      :param ratings_matrix: 用户-物品打分矩阵
      :param item_similar: 物品两两间的相似度
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
  
      yield from _predict_all(uid, item_ids, ratings_matrix, item_similar)
  
  if __name__ == '__main__':
      ratings_matrix = load_data(DATA_PATH)
  
      item_similar = compute_pearson_similarity(ratings_matrix, based="item")
  
      for result in predict_all(1, ratings_matrix, item_similar, filter_rule=["unhot", "rated"]):
          print(result)
  ```

- 为指定用户推荐TOP-N结果

  ```python
  # ......
  
  def top_k_rs_result(k):
      ratings_matrix = load_data(DATA_PATH)
  
      item_similar = compute_pearson_similarity(ratings_matrix, based="item")
      results = predict_all(1, ratings_matrix, item_similar, filter_rule=["unhot", "rated"])
      return sorted(results, key=lambda x: x[2], reverse=True)[:k]
  
  if __name__ == '__main__':
      from pprint import pprint
      result = top_k_rs_result(20)
      pprint(result)
  ```

  

