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

#### 运行

```python
from pymongo import MongoClient
from pprint import pprint

client = MongoClient('mongodb://localhost:27017/')

with client:
    db = client.testdb
    status = db.command("serverStatus")
    print(status)
    status = db.command("dbstats")
    print(status)
```

#### 增加

```python
with client:
    db=client.testdb
    
    # 增加一条文档对象
    car={'name':'zhangsan','price':'1'}
    db.cars.insert_one(car)
    # 增加多条文档对象
    car1={'name':'hr','price':'2'}
    car2={'name':'mnc','price':'3'}
    cars=[car1, car2]
    db.cars.insert_many(cars)
    print("ok")
except Exception,e:
    print(e)
```

#### 删除

- 删除集合

```python
with client
		db.cars.drop()
```

- 删除文档对象

```python
with client:
		# 删除一条文档对象
		db.cars.delete_one(car)
		# 删除多条文档对象
		db.stu.delete_many(cars)
```

#### 修改

```python
with clinet:
    condition = {"name": "vol"}  # 查询条件
  	car = db,cars.find_one(condition)
    car['price'] = 2
    # 修改一条文档对象
    result = db.cars.update_one(car, {'$set': car})
    print(result, result.matched_count, result.modified_count)  # matched_count,modified_count表示匹配和影响的数据条数
		condition = {"price": {"$gt": 2}}  #  查询条件
    restult = db.cars.update_one(condition, {'$inc': {'age': 1}})
    print(result, result.matched_count, result.modified_count)
    
    # 修改多行文档对象
    condition = {"price": {"$gt": 2}}  #  查询条件
    restult = db.cars.update_many(condition, {'$inc': {'age': 1}})
    print(result, result.matched_count, result.modified_count)   
```

#### 查询

- 列出集合

```python
with client:
    db = client.testdb
    print(db.collection_names())
```

- 游标

```python
with client:
		db = client.testdb
		doc = sb.stu.find()  # 返回一个pymongo游标
		print(doc.next())  # 从结果中获取下一个文档
    doc.rewind() # 将游标倒回其未评估状态
    print(doc.next())
    
    print(list(doc)) # 将游标转换为python列表，将所有数据加载到内存中
```

- 读取所有数据

```python
with client:
  	db = client.testdb
    cars = db.cars.find()
    for cat in cars:
      print(f"{car['name']}{car['price']}")
```

- 记数文件

```python
with client:
  	db = client.testdb
    n_cars = db.cars.find().count()
    print(n_cars)
```

- 过滤器

```python
with client:
    db = client.testdb
    expensive_cars = db.cars.find({'price': {'$gt': 50000}})
    for ecar in expensive_cars:
        print(ecar['name'])
```

- 投影

通过投影，我们可以从返回的文档中选择特定字段。 投影在`find()`方法的第二个参数中传递。

```python
with client:
    db = client.testdb
    cars = db.cars.find({}, {'_id': 1, 'name':1})
    for car in cars:
        print(car)
```

- 排序

```python
with client:
    db = client.testdb
    cars = db.cars.find().sort("price", DESCENDING)
    for car in cars:
        print('{0} {1}'.format(car['name'], 
            car['price']))
```

- 聚合

```python
with client:
    db = client.testdb
    # sum运算符计算并返回数值的总和。 group运算符通过指定的标识符表达式对输入文档进行分组，并将累加器表达式（如果指定）应用于每个组。
    agr = [ {'group': {'_id': 1, 'all': { 'sum': '$price' } } } ]
    # aggregate()方法将聚合操作应用于cars集合
    val = list(db.cars.aggregate(agr))
    print('The sum of prices is {}'.format(val[0]['all']))
    
    
with client:
    db = client.testdb
    # 计算奥迪和沃尔沃汽车的价格总和
    agr = [{ 'match': {'or': [ { 'name': "Audi" }, { 'name': "Volvo" }] }}, 
        { 'group': {'_id': 1, 'sum2cars': { 'sum': "$price" } }}]
    val = list(db.cars.aggregate(agr))
    print('The sum of prices of two cars is {}'.format(val[0]['sum2cars']))


```

- 限制输出

`limit`查询选项指定要返回的文档数量，`skip()`选项指定某些文档。

```python
with client:
    db = client.testdb
		# skip()方法跳过前两个文档，limit()方法将输出限制为三个文档
    cars = db.cars.find().skip(2).limit(3)
    for car in cars:
        print('{0}: {1}'.format(car['name'], car['price']))
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

