# peewee

## 概述

[参考](https://github.com/coleifer/peewee)

[文档](http://docs.peewee-orm.com/en/latest/peewee/quickstart.html#quickstart)

Peewee是一个简单而小型的ORM。它几乎没有（但富有表现力）概念，使其易于学习且使用直观。

- 一个小的表达ORM
- python 2.7+和3.4+（使用3.6开发）
- 支持sqlite，mysql，postgresql和cockroachdb
- 大量的[extensions](http://docs.peewee-orm.com/en/latest/peewee/playhouse.html)

```
pip install peewee
```

## 使用

### 定义类

```python
from peewee import *
import datetime

db = SqliteDatabase('my_database.db')

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    username = CharField(unique=True)

class Tweet(BaseModel):
    user = ForeignKeyField(User, backref='tweets')
    message = TextField()
    created_date = DateTimeField(default=datetime.datetime.now)
    is_published = BooleanField(default=True)
```

### 创建表

```python
db.connect()
db.create_tables([User, Tweet])
```

### 增加数据

```python
charlie = User.create(username='charlie')
huey = User(username='huey')
huey.save()

# No need to set `is_published` or `created_date` since they
# will just use the default values we specified.
Tweet.create(user=charlie, message='My first tweet')
```

### 查询

```python
# A simple query selecting a user.
User.get(User.username == 'charlie')

# Get tweets created by one of several users.
usernames = ['charlie', 'huey', 'mickey']
users = User.select().where(User.username.in_(usernames))
tweets = Tweet.select().where(Tweet.user.in_(users))

# We could accomplish the same using a JOIN:
tweets = (Tweet
          .select()
          .join(User)
          .where(User.username.in_(usernames)))

# How many tweets were published today?
tweets_today = (Tweet
                .select()
                .where(
                    (Tweet.created_date >= datetime.date.today()) &
                    (Tweet.is_published == True))
                .count())

# Paginate the user table and show me page 3 (users 41-60).
User.select().order_by(User.username).paginate(3, 20)

# Order users by the number of tweets they've created:
tweet_ct = fn.Count(Tweet.id)
users = (User
         .select(User, tweet_ct.alias('ct'))
         .join(Tweet, JOIN.LEFT_OUTER)
         .group_by(User)
         .order_by(tweet_ct.desc()))

# Do an atomic update
Counter.update(count=Counter.count + 1).where(Counter.url == request.url)
```

