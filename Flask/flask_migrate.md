# 数据库迁移

在Flask中可以使用Flask-Migrate扩展，来实现数据迁移。并且集成到Flask-Script中，所有操作通过命令就能完成。

为了导出数据库迁移命令，Flask-Migrate提供了一个MigrateCommand类，可以附加到flask-script的manager对象上。

**在虚拟环境中安装Flask-Migrate**

```
pip install flask-migrate
```

**创建模型类文件database.py**

```
#coding=utf-8
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate,MigrateCommand
from flask_script import Shell,Manager

# 创建Flask实例
app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@127.0.0.1:3306/Flask_test'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# 创建SQLAlchemy实例
db = SQLAlchemy(app)

# 数据库迁移
# 1.创建Manager管理器Flask-Script实例
manager = Manager(app)
# 2.创建迁移框架于程序的关联，第一个参数是Flask的实例，第二个参数是Sqlalchemy数据库实例
migrate = Migrate(app,db) 
# 3.添加迁移命令，manager是Flask-Script的实例，这条语句在flask-Script中添加一个db命令
manager.add_command('db',MigrateCommand)

# 定义模型Role
class Role(db.Model):
    # 定义表名
    __tablename__ = 'roles'
    # 定义列对象
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    def __repr__(self):
        return 'Role:'.format(self.name)

# 定义用户
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    def __repr__(self):
        return 'User:'.format(self.username)
        
        
if __name__ == '__main__':
    manager.run()
```

**创建迁移仓库**

```
# 这个命令会创建migrations文件夹，所有迁移文件都放在里面。
python 文件名.py db init
```

**创建迁移脚本**

```
# 创建自动迁移脚本
python 文件名.py db migrate -m 'initial migration'
```

**更新数据库**

```
python 文件名.py db upgrade
```

**回退数据库**

```
# 查看数据库历史版本
python 文件名.py db history

# 回退数据库至特定版本
python 文件名.py db downgrade 版本号
```