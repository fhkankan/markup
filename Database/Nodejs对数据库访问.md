# MySQL

## 安装驱动

本教程使用了[淘宝定制的 cnpm 命令](https://www.runoob.com/nodejs/nodejs-npm.html#taobaonpm)进行安装：

```
$ cnpm install mysql
```

## 连接数据库

在以下实例中根据你的实际配置修改数据库用户名、及密码及数据库名：

test.js 文件代码：
```javascript
var mysql      = require('mysql');
var connection = mysql.createConnection({
  host     : 'localhost',
  user     : 'root', 
  password : '123456',
  database : 'test'
}); 

connection.connect();
connection.query('SELECT 1 + 1 AS solution', function (error, results, fields) { 
  if (error) throw error; 
  console.log('The solution is: ', results[0].solution); 
});
```
执行以下命令输出结果为：
```
$ node test.js
The solution is: 2
```
- 数据库连接参数说明：

| 参数               | 描述                                                         |
| :----------------- | :----------------------------------------------------------- |
| host               | 主机地址 （默认：localhost）                                 |
| user               | 用户名                                                       |
| password           | 密码                                                         |
| port               | 端口号 （默认：3306）                                        |
| database           | 数据库名                                                     |
| charset            | 连接字符集（默认：'UTF8_GENERAL_CI'，注意字符集的字母都要大写） |
| localAddress       | 此IP用于TCP连接（可选）                                      |
| socketPath         | 连接到unix域路径，当使用 host 和 port 时会被忽略             |
| timezone           | 时区（默认：'local'）                                        |
| connectTimeout     | 连接超时（默认：不限制；单位：毫秒）                         |
| stringifyObjects   | 是否序列化对象                                               |
| typeCast           | 是否将列值转化为本地JavaScript类型值 （默认：true）          |
| queryFormat        | 自定义query语句格式化方法                                    |
| supportBigNumbers  | 数据库支持bigint或decimal类型列时，需要设此option为true （默认：false） |
| bigNumberStrings   | supportBigNumbers和bigNumberStrings启用 强制bigint或decimal列以JavaScript字符串类型返回（默认：false） |
| dateStrings        | 强制timestamp,datetime,data类型以字符串类型返回，而不是JavaScript Date类型（默认：false） |
| debug              | 开启调试（默认：false）                                      |
| multipleStatements | 是否许一个query中有多个MySQL语句 （默认：false）             |
| flags              | 用于修改连接标志                                             |
| ssl                | 使用ssl参数（与crypto.createCredenitals参数格式一至）或一个包含ssl配置文件名称的字符串，目前只捆绑Amazon RDS的配置文件 |

更多说明可参见：https://github.com/mysqljs/mysql

------

## 数据库操作

在进行数据库操作前，你需要将本站提供的 Websites 表 SQL 文件[websites.sql](https://static.runoob.com/download/websites.sql) 导入到你的 MySQL 数据库中。

本教程测试的 MySQL 用户名为 root，密码为 123456，数据库为 test，你需要根据自己配置情况修改。

### 查询数据

将上面我们提供的 SQL 文件导入数据库后，执行以下代码即可查询出数据：
```javascript
var mysql  = require('mysql');

var connection = mysql.createConnection({ 
  host     : 'localhost', 
  user     : 'root', 
  password : '123456', 
  port: '3306',
  database: 'test'
}); 
connection.connect(); 
var  sql = 'SELECT * FROM websites'; 

//查
connection.query(sql,function (err, result) {        
  if(err){ 
    console.log('[SELECT ERROR] - ',err.message);          
    return;
  }  
  console.log('--------------------------SELECT----------------------------');
  console.log(result); 
  console.log('------------------------------------------------------------\n\n');
}); 

connection.end();
```
执行以下命令输出就结果为：

```
$ node test.js
--------------------------SELECT----------------------------
[ RowDataPacket {
    id: 1,
    name: 'Google',
    url: 'https://www.google.cm/',
    alexa: 1,
    country: 'USA' },
  RowDataPacket {
    id: 2,
    name: '淘宝',
    url: 'https://www.taobao.com/',
    alexa: 13,
    country: 'CN' },
  RowDataPacket {
    id: 3,
    name: '菜鸟教程',
    url: 'http://www.runoob.com/',
    alexa: 4689,
    country: 'CN' },
  RowDataPacket {
    id: 4,
    name: '微博',
    url: 'http://weibo.com/',
    alexa: 20,
    country: 'CN' },
  RowDataPacket {
    id: 5,
    name: 'Facebook',
    url: 'https://www.facebook.com/',
    alexa: 3,
    country: 'USA' } ]
------------------------------------------------------------
```

### 插入数据

我们可以向数据表 websties 插入数据：
```javascript
var mysql  = require('mysql');

var connection = mysql.createConnection({
  host     : 'localhost', 
  user     : 'root', 
  password : '123456', 
  port: '3306',  
  database: 'test'
});

connection.connect();

var  addSql = 'INSERT INTO websites(Id,name,url,alexa,country) VALUES(0,?,?,?,?)'; 

var  addSqlParams = ['菜鸟工具', 'https://c.runoob.com','23453', 'CN'];

//增
connection.query(addSql,addSqlParams,function (err, result) {
  if(err){
    console.log('[INSERT ERROR] - ',err.message);         
    return; 
  } 
  console.log('--------------------------INSERT----------------------------'); 
  //console.log('INSERT ID:',result.insertId);  
  console.log('INSERT ID:',result);  
  console.log('-----------------------------------------------------------------\n\n');  
}); 

connection.end();
```
执行以下命令输出就结果为：

```
$ node test.js
--------------------------INSERT----------------------------
INSERT ID: OkPacket {
  fieldCount: 0,
  affectedRows: 1,
  insertId: 6,
  serverStatus: 2,
  warningCount: 0,
  message: '',
  protocol41: true,
  changedRows: 0 }
-----------------------------------------------------------------
```

执行成功后，查看数据表，即可以看到添加的数据

### 更新数据

我们也可以对数据库的数据进行修改：
```javascript
var mysql  = require('mysql'); 

var connection = mysql.createConnection({ 
  host     : 'localhost', 
  user     : 'root', 
  password : '123456', 
  port: '3306',  
  database: 'test'
});

connection.connect();

var modSql = 'UPDATE websites SET name = ?,url = ? WHERE Id = ?';

var modSqlParams = ['菜鸟移动站', 'https://m.runoob.com',6];

//改
connection.query(modSql,modSqlParams,function (err, result) {
  if(err){ 
    console.log('[UPDATE ERROR] - ',err.message);         
    return; 
  }
  console.log('--------------------------UPDATE----------------------------');
  console.log('UPDATE affectedRows',result.affectedRows); 
  console.log('-----------------------------------------------------------------\n\n');
});

connection.end();
```
执行以下命令输出就结果为：
```
--------------------------UPDATE----------------------------
UPDATE affectedRows 1
-----------------------------------------------------------------
```

执行成功后，查看数据表，即可以看到更新的数据：

### 删除数据

我们可以使用以下代码来删除 id 为 6 的数据:
```javascript
var mysql  = require('mysql'); 

var connection = mysql.createConnection({
  host     : 'localhost', 
  user     : 'root',
  password : '123456', 
  port: '3306',   
  database: 'test'
}); 

connection.connect();

var delSql = 'DELETE FROM websites where id=6'; 

//删
connection.query(delSql,function (err, result) {        
  if(err){ 
    console.log('[DELETE ERROR] - ',err.message);          
    return; 
  }                console.log('--------------------------DELETE----------------------------');       console.log('DELETE affectedRows',result.affectedRows);       console.log('-----------------------------------------------------------------\n\n');   });  connection.end();
```
执行以下命令输出就结果为：

```
--------------------------DELETE----------------------------
DELETE affectedRows 1
-----------------------------------------------------------------
```
执行成功后，查看数据表，即可以看到 id=6 的数据已被删除

# MongoDB

MongoDB是一种文档导向数据库管理系统，由C++撰写而成。

本章节我们将为大家介绍如何使用 Node.js 来连接 MongoDB，并对数据库进行操作。

如果你还没有 MongoDB 的基本知识，可以参考我们的教程：[MongoDB 教程](https://www.runoob.com/mongodb/mongodb-tutorial.html)。

## 安装驱动

本教程使用了[淘宝定制的 cnpm 命令](https://www.runoob.com/nodejs/nodejs-npm.html#taobaonpm)进行安装：

```
$ cnpm install mongodb
```

接下来我们来实现增删改查功能。

## 创建数据库

要在 MongoDB 中创建一个数据库，首先我们需要创建一个 MongoClient 对象，然后配置好指定的 URL 和 端口号。

如果数据库不存在，MongoDB将创建数据库并建立连接。

- 创建连接
```javascript
var MongoClient = require('mongodb').MongoClient; 

var url = "mongodb://localhost:27017/runoob";  MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {  if (err) throw err;  console.log("数据库已创建!");  db.close(); });
```

## 创建集合

我们可以使用 createCollection() 方法来创建集合
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = 'mongodb://localhost:27017/runoob'; 

MongoClient.connect(url, { useNewUrlParser: true }, function (err, db) { 
  if (err) throw err;  
  console.log('数据库已创建');
  var dbase = db.db("runoob"); 
  dbase.createCollection('site', function (err, res) {        
    if (err) throw err; 
    console.log("创建集合!");
    db.close(); 
  });
});
```
## 数据库操作

与 MySQL 不同的是 MongoDB 会自动创建数据库和集合，所以使用前我们不需要手动去创建。

### 插入数据

- 插入一条数据

以下实例我们连接数据库 runoob 的 site 表，并插入一条数据条数据，使用 **insertOne()**：

```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {
  if (err) throw err; 
  var dbo = db.db("runoob"); 
  var myobj = { name: "菜鸟教程", url: "www.runoob" };    
  dbo.collection("site").insertOne(myobj, function(err, res) { 
    if (err) throw err; 
    console.log("文档插入成功"); 
    db.close();  
  });
});
```
执行以下命令输出就结果为：

```
$ node test.js
文档插入成功
```

从输出结果来看，数据已插入成功。

我们也可以打开 MongoDB 的客户端查看数据，如：

```
> show dbs
runoob  0.000GB          # 自动创建了 runoob 数据库
> show tables
site                     # 自动创建了 site 集合（数据表）
> db.site.find()
{ "_id" : ObjectId("5a794e36763eb821b24db854"), "name" : "菜鸟教程", "url" : "www.runoob" }
> 
```

如果要插入多条数据可以使用 **insertMany()**：

- 插入多条数据
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {
  if (err) throw err;
  var dbo = db.db("runoob");
  var myobj =  [ 
    { name: '菜鸟工具', url: 'https://c.runoob.com', type: 'cn'}, 
    { name: 'Google', url: 'https://www.google.com', type: 'en'}, 
    { name: 'Facebook', url: 'https://www.google.com', type: 'en'} 
  ]; 
  dbo.collection("site").insertMany(myobj, function(err, res) { 
    if (err) throw err;  
    console.log("插入的文档数量为: " + res.insertedCount);        
    db.close(); 
  });
});
```
res.insertedCount 为插入的条数。

### 查询数据

可以使用 find() 来查找数据, find() 可以返回匹配条件的所有数据。 如果未指定条件，find() 返回集合中的所有数据。

- find

```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {
  if (err) throw err; 
  var dbo = db.db("runoob"); 
  dbo.collection("site"). find({}).toArray(function(err, result) { 
    // 返回集合中所有数据 
    if (err) throw err; 
    console.log(result); 
    db.close(); 
  });
});
```
以下实例检索 name 为 "菜鸟教程" 的实例：

- 查询指定条件的数据
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) { 
  if (err) throw err; 
  var dbo = db.db("runoob");
  var whereStr = {"name":'菜鸟教程'};
  // 查询条件   
  dbo.collection("site").find(whereStr).toArray(function(err, result) { 
    if (err) throw err;   
    console.log(result);
    db.close();
  });
});
```
执行以下命令输出就结果为：

```
[ { _id: 5a794e36763eb821b24db854,
    name: '菜鸟教程',
    url: 'www.runoob' } ]
```

### 更新数据

我们也可以对数据库的数据进行修改，以下实例将 name 为 "菜鸟教程" 的 url 改为 https://www.runoob.com：

- 更新一条数据
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";  

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) { 
  if (err) throw err; 
  var dbo = db.db("runoob"); 
  var whereStr = {"name":'菜鸟教程'};
  // 查询条件
  var updateStr = {$set: { "url" : "https://www.runoob.com" }};    
  dbo.collection("site").updateOne(whereStr, updateStr, function(err, res) {  
    if (err) throw err;        
    console.log("文档更新成功");        
    db.close(); 
  });
});
```
执行成功后，进入 mongo 管理工具查看数据已修改：

```
> db.site.find().pretty()
{
    "_id" : ObjectId("5a794e36763eb821b24db854"),
    "name" : "菜鸟教程",
    "url" : "https://www.runoob.com"     // 已修改为 https
}
```

如果要更新所有符合条的文档数据可以使用 **updateMany()**：

- 更新多条数据

```javascript

var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) { 
  if (err) throw err;  
  var dbo = db.db("runoob"); 
  var whereStr = {"type":'en'}; 
  // 查询条件
  var updateStr = {$set: { "url" : "https://www.runoob.com" }};    
  dbo.collection("site").updateMany(whereStr, updateStr, function(err, res) {      
    if (err) throw err;         
    console.log(res.result.nModified + " 条文档被更新");
    db.close(); 
  }); 
});
```
result.nModified 为更新的条数。

### 删除数据

以下实例将 name 为 "菜鸟教程" 的数据删除 

- 删除一条数据
```javascript
var MongoClient = require('mongodb').MongoClient; 
var url = "mongodb://localhost:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) { 
  if (err) throw err; 
  var dbo = db.db("runoob");
  var whereStr = {"name":'菜鸟教程'};  
  // 查询条件 
  dbo.collection("site").deleteOne(whereStr, function(err, obj) {        
    if (err) throw err;        
    console.log("文档删除成功");        
    db.close();
  });
});
```
执行成功后，进入 mongo 管理工具查看数据已删除：

```
> db.site.find()
> 
```

如果要删除多条语句可以使用 **deleteMany()** 方法

以下实例将 type 为 en 的所有数据删除 :

- 删除多条数据
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";  

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) { 
  if (err) throw err; 
  var dbo = db.db("runoob"); 
  var whereStr = { type: "en" };  
  // 查询条件
  dbo.collection("site").deleteMany(whereStr, function(err, obj) {        
    if (err) throw err; 
    console.log(obj.result.n + " 条文档被删除");      
    db.close();  
  });
});
```
obj.result.n 删除的条数。

### 排序

排序 使用 sort() 方法，该方法接受一个参数，规定是升序(1)还是降序(-1)。

例如：

```
{ type: 1 }  // 按 type 字段升序
{ type: -1 } // 按 type 字段降序
```

按 type 升序排列:
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";  

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {  
  if (err) throw err; 
  var dbo = db.db("runoob"); 
  var mysort = { type: 1 };    
  dbo.collection("site").find().sort(mysort).toArray(function(err, result) {   
    if (err) throw err;        
    console.log(result);        
    db.close(); 
  });
});
```
### 查询分页

如果要设置指定的返回条数可以使用 **limit()** 方法，该方法只接受一个参数，指定了返回的条数。

- limit()：读取两条数据
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {  
  if (err) throw err;
  var dbo = db.db("runoob");    
  dbo.collection("site").find().limit(2).toArray(function(err, result) {        
    if (err) throw err;        
    console.log(result);        
    db.close(); 
  }); 
});
```
如果要指定跳过的条数，可以使用 **skip()** 方法。

- skip(): 跳过前面两条数据，读取两条数据

```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";  

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {
  if (err) throw err;
  var dbo = db.db("runoob");    
  dbo.collection("site").find().skip(2).limit(2).toArray(function(err, result) {
    if (err) throw err;        
    console.log(result);        
    db.close(); 
  }); 
});
```
### 连接操作

mongoDB 不是一个关系型数据库，但我们可以使用` $lookup `来实现左连接。

例如我们有两个集合数据分别为：

集合1：orders

```
[
  { _id: 1, product_id: 154, status: 1 }
]
```

集合2：products

```
[
  { _id: 154, name: '笔记本电脑' },
  { _id: 155, name: '耳机' },
  { _id: 156, name: '台式电脑' }
]
```

- `$lookup` 实现左连接
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://127.0.0.1:27017/"; 

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) { 
  if (err) throw err; 
  var dbo = db.db("runoob"); 
  dbo.collection('orders').aggregate([ 
    { $lookup:      
     {         
       from: 'products', // 右集合 
       localField: 'product_id',// 左集合 join 字段
       foreignField: '_id',         // 右集合 join 字段 
       as: 'orderdetails'           // 新生成字段（类型array） 
     }  
    }   
  ]).toArray(function(err, res) {
    if (err) throw err; 
    console.log(JSON.stringify(res));
    db.close(); 
  });
});
```
### 删除集合

我们可以使用 **drop()** 方法来删除集合：
```javascript
var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://localhost:27017/";

MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {    
  if (err) throw err;
  var dbo = db.db("runoob"); 
  // 删除 test 集合 
  dbo.collection("test").drop(function(err, delOK) { 
    // 执行成功 delOK 返回 true，否则返回 false
    if (err) throw err; 
    if (delOK) console.log("集合已删除");
    db.close();
  });
});
```