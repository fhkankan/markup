# DotEnv

## 概述

Dotenv 文件（通常命名为` .env`）是一种常见的模式，它可以以独立于平台的方式轻松使用环境变量。

规则

```
1.使用键值对的形式定义
2.以  #  开头的为注释
3.若是使用变量，需要使用${}包裹
```

配置

```.env
# ignore comment
ENVIRONMENT="production"
REDIS_ADDRESS=localhost:6379
MEANING_OF_LIFE=42
MY_VAR='Hello world'
DOMAIN=example.org
ADMIN_EMAIL=admin@${DOMAIN}
ROOT_URL=${DOMAIN}/app
```

## python

- 安装

```
pip install python-dotenv
```

- 使用

`load_dotenv()`

```python
load_dotenv(
    dotenv_path: Union[str, _PathLike, None] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
)
# 将env中的变量写入到系统变量中

# 参数
# dotenv_path: 指定.env文件路径，当然如果不传该参数的话（默认为None）也会自定调用dotenv.find_dotenv()去查找文件位置的，但是你的文件名如果不是.env那就必须传递该参数了
# stream：如果dotenv_path是None，.env文件内容的文本流(例如io.StringIO)
# verbose：.env文件缺失时是否输出警告信息。默认Fasle
# override： 当.env文件中有变量与系统中原来的环境变量有冲突时，是否覆盖系统，默认Fasle
# encoding: 指定文件读取时的编码方式。
# 如果dotenv_path和stream都存在，则dotenv.find_dotenv()去查找文件位置
```

`dotenv_values`

```python
dotenv_values(
    dotenv_path: Union[str, _PathLike, None] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
)
# 获取env文件中的变量信息
```

获取系统变量

```python
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 方法一：自动搜索.env文件
load_dotenv(verbose=True)
# 方法二：自动搜索.env文件
load_dotenv(find_dotenv(), verbose=True)
# 方法三：指定.env文件位置
BASE_DIR = Path(__file__).absolute().parent
env_path = Path(BASE_DIR).joinpath(".env")
load_dotenv(dotenv_path=env_path, verbose=True)

# 通过load_dotenv ，你就可以访问像访问系统环境变量一样使用.env文件中的变量了
ENVIRONMENT = os.getenv("ENVIRONMENT")
ENVIRONMENT = os.environ.get("ENVIRONMENT")
print(ENVIRONMENT)
print(os.environ)  # 所有的系统变量
```

仅获取env变量

```python
from io import StringIO
from dotenv import dotenv_values

# 加载env文件
BASE_DIR = Path(__file__).absolute().parent
env_path = Path(BASE_DIR).joinpath(".env")
res = dotenv_values(dotenv_path=env_path)
print(res)

# 加载流对象
filelike = StringIO('SPAM=EGGS\n')
filelike.seek(0)
parsed = dotenv_values(stream=filelike)
res = parsed["SPAM"]
print(res)
```



