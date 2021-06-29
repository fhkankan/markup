# 与python交互

## 同步pymongo

### 安装

```shell
pip install mongoengine
pip install pymongo
```

### 操作

```python
from pymongo import *


# 使用init方法创建连接MongoClient对象
client = MongoClient('主机ip',端口)

# 通过client对象获取获得数据库对象
db = client.数据库名称

# 通过db对象获取集合对象
collections = db.集合名称

# 主要方法
insert 			# 加入多条文档对象
insert_one 		# 加入一条文档对象
insert_many 	# 加入多条文档对象
delete_one 		# 删除一条文档对象
delete_many 	# 删除多条文档对象
update_one 		# 更新一条文档对象
update_many 	# 更新多条文档对象
find_one 		# 查找一条文档对象
find 			# 查找多条文档对象

# Cursor对象
# 当调用集合对象的find()方法时，会返回Cursor对象
# 结合for...in...遍历cursor对象
```

增加

```python
from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    
    # 增加一条文档对象
    doc={'name':'zhangsan','home':'henan'}
    db.stu.insert_one(doc)
    # 增加多条文档对象
    doc1={'name':'hr','home':'thd'}
    doc2={'name':'mnc','home':'njc'}
    doc=[doc1,doc2]
    db.stu.insert_many(doc)
    print("ok")
except Exception,e:
    print(e)
```

删除

```python
from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    
    # 删除一条文档对象
    db.stu.delete_one({'gender':True})
    # 删除多条文档对象
    db.stu.delete_many({'gender':False})
    print 'ok'
except Exception,e:
    print(e)
```

修改

```python
from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    
    # 修改一条文档对象
    db.stu.update_one({'gender':False},{'$set':{'name':'hehe'}})
    # 修改多条文档对象
    db.stu.update_many({'gender':True},{'$set':{'name':'haha'}})
    print('ok')
except Exception,e:
    print(e)
```

查询

```python
from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    
    # 查询一条文档对象
    doc=db.stu.find_one()
    print('%s--%s'%(doc['name'],doc['hometown']))
    # 查询多条文档对象
    cursor=db.stu.find({'hometown':'大理'})
    for doc in cursor:
        print('%s--%s'%(doc['name'],doc['hometown']))
except Exception,e:
    print(e)
```

### 集成Django

- 配置

方法一

```python
# settings.py
DBNAME = 'mymongo'

# models.py
from mongoengine import *
from .settings import DBNAME

connect(DBNAME)

class Post(Document):
	...
```

方法二

```python
# settings.py
INSTALLED_APPS =[
    'mongoengine',
]
MONGODB_DATABSES = {
    "default":{
        "name": "test",
        "host": '127.0.0.1',
        "tz_aware": True,  # 时区
    }
}
DATABASES= {
    'default':{
        'ENGINE': 'django.db.backends.dumpy'
    }
}
from mongine import connect
connect('test', host='127.0.0.1')

# modules.py
from mongoengine import Document

class Peom(Document):
	...
```

## 异步motor

### 安装

```shell
pip install motor
```

### 操作

```python
import motor

# 主机和端口号
client = motor.motor_asyncio.AsyncIOMotorClient('localhost', 27017)

# 用户名和密码
client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://root:123456@localhost:27017')

# 获取数据库
db = client.test_database
db = client['test_database']

# 获取集合
collection = db.test_collection
collection = db['test_collection']

# 主要方法
insert 			# 加入多条文档对象
insert_one 		# 加入一条文档对象
insert_many 	# 加入多条文档对象
delete_one 		# 删除一条文档对象
delete_many 	# 删除多条文档对象
update_one 		# 更新一条文档对象
update_many 	# 更新多条文档对象
find_one 		# 查找一条文档对象
find 			# 查找多条文档对象
count_documents # 查询集合中文档数量

async for ... in ...
```

增加

```python
async def do_insert():
	# 插入一条数据
    document = {'key': 'value'}
    result = await db.test_collection.insert_one(document)
    print('result %s' % repr(result.inserted_id))
    
    # 批量插入文档
    result = await db.test_collection.insert_many(
        [{'i': i} for i in range(2000)])
    print('inserted %d docs' % (len(result.inserted_ids),))
 
loop = asyncio.get_event_loop()
loop.run_until_complete(do_insert())
```

删除

```python
async def do_delete_many():
    coll = db.test_collection
    n = await coll.count_documents({})
    print('%s documents before calling delete_many()' % n)
    result = await db.test_collection.delete_many({'i': {'$gte': 1000}})
    print('%s documents after' % (await coll.count_documents({})))
 
loop = asyncio.get_event_loop()
loop.run_until_complete(do_delete_many())
# 2000 documents before calling delete_many()
# 1000 documents after
```

修改

```python
# replace_one()
# 除了_id不变，会对修改后文档的所有字段做更新操作， 慎用 
async def do_replace():
    coll = db.test_collection
    old_document = await coll.find_one({'i': 50})
    print('found document: %s' % pprint.pformat(old_document))
    _id = old_document['_id']
 
    old_document['i'] = -1  # 修改文档(dict)的key, value
    old_document['new'] = 'new'  # 增加文档(dict)的key, value
    del old_document['i']  # 删除文档(dict)的key, value
 
    result = await coll.replace_one(
        {'_id': _id}, old_document)  # replace_one第一个参数为查询条件, 第二个参数为更新后的文档
    print('replaced %s document' % result.modified_count)
    new_document = await coll.find_one({'_id': _id})
    print('document is now %s' % pprint.pformat(new_document))
 
loop = asyncio.get_event_loop()
loop.run_until_complete(do_replace())
# found document: {'_id': ObjectId('...'), 'i': 50}
# replaced 1 document
# document is now {'_id': ObjectId('...'), 'key': 'value'}


# update_one()
async def do_update():
    coll = db.test_collection
    result = await coll.update_one({'i': 51}, {'$set': {'key': 'value'}})  # 仅新增或更改该文档的某个key
    print('updated %s document' % result.modified_count)
    new_document = await coll.find_one({'i': 51})
    print('document is now %s' % pprint.pformat(new_document))
 
loop = asyncio.get_event_loop()
loop.run_until_complete(do_update())
# updated 1 document
# document is now {'_id': ObjectId('...'), 'i': 51, 'key': 'value'}

# update_many()
await coll.update_many({'i': {'$gt': 100}}, {'$set': {'key': 'value'}})
```

查询

```python
async def do_find():
    # 查询一条数据
    document = await db.test_collection.find_one({'i': {'$lt': 1}})  
    pprint.pprint(document)
    
    # 查询多个文档
    cursor = db.test_collection.find({'i': {'$lt': 5}}).sort('i')
    for document in await cursor.to_list(length=100):
        pprint.pprint(document)
        
    # 查询所有文档
    cursor = db.test_collection.find({'i': {'$lt': 4}})
    cursor.sort('i', -1).skip(1).limit(2)  # 对查询应用排序(sort)，跳过(skip)或限制(limit)
    async for document in cursor:
        pprint.pprint(document)

	# 查询集合中文档数量
    n = await db.test_collection.count_documents({})  
    print('%s documents in collection' % n)
    n = await db.test_collection.count_documents({'i': {'$gt': 1000}})
    print('%s documents where i > 1000' % n)

 
loop = asyncio.get_event_loop()
loop.run_until_complete(do_find())
```

命令

```python
from bson import SON
 
async def use_distinct_command():
    response = await db.command(
        SON([("distinct", "test_collection"),("key", "i")])
    )
 
loop = asyncio.get_event_loop()
loop.run_until_complete(use_distinct_command())

# 由于命令参数的顺序很重要，因此不要使用Python dict来传递命令的参数。相反，养成使用PyMongo附带bson.SON的 bson 模块的习惯。
```

