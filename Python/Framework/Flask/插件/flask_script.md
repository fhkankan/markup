# 扩展命令行

通过使用Flask-Script扩展，我们可以在Flask服务器启动的时候，通过命令行的方式传入参数。

**创建程序**

```python
from flask import Flask
from flask_script import Manager

app = Flask(__name__)

manager = Manager(app)

@app.route('/')
def index():
    return '床前明月光'

if __name__ == "__main__":
    manager.run()
```

**命令行启动**

```python
# 来查看参数
python hello.py runserver --help

# 启动服务
python hello.py runserver -h ip地址 -p 端口号
```

