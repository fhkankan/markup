# web项目

## 目录结构

```
-api  # api文件夹
 |-user.py
-biz  # server文件夹
 |-user.py
-conf	# 配置文件，注意文件名与bash文件一致
 |-local.py
 |-test.py
 |-prod.py
-docs  # 接口文档
-logs  # 日志，项目启动自动生成
-models	# 模型类文件夹
 |-base.py
 |-model.py
main.py
run.sh
test.sh
prod.sh
```

## api

`user.py`

```python
from sanic import Blueprint
from sanic.log import logger

from common.const import RC
from common.utils import res_ng, res_ok, check_params, check_auth

from biz.user import register_api, login_api, veri_code_api

bp = Blueprint("user", url_prefix="/user")


@bp.post("/register")
async def register(request):
    try:
        params = request.json
        r_k = ["user_name", "gender", "birth_day", "phone", "veri_code"]
        r_k_v = check_params(r_k, params)
        o_k = ["province", "city"]
        o_k_v = {k: params.get(k) for k in o_k if params.get(k)}
        params = r_k_v
        params.update(o_k_v)
        flag, result = await register_api(request.app, params)
        return res_ok(**result) if flag else res_ng(**result)
    except AssertionError as e:
        return res_ng(code=RC.PARAMS_INVALID, msg=str(e))
    except Exception as e:
        logger.exception(e)
        return res_ng(code=RC.INTERNAL_ERROR, msg="服务器错误，请稍后再试")


@bp.post("/login")
async def login(request):
    try:
        params = request.json
        assert params.get("phone"), "phone参数有缺失"
        assert params.get("veri_code"), "veri_code参数有缺失"
        flag, result = await login_api(request, params)
        return res_ok(**result) if flag else res_ng(**result)
    except AssertionError as e:
        return res_ng(code=RC.PARAMS_INVALID, msg=str(e))
    except Exception as e:
        logger.exception(e)
        return res_ng(code=RC.INTERNAL_ERROR, msg="服务器错误，请稍后再试")
      
@bp.get("data_list")
@check_auth(role_ids=[1,2])
async def data_list(request):
    pass
```

## biz

`user.py`

```python
from sanic.log import logger
from playhouse.shortcuts import model_to_dict
from models.model import User


async def user_index(app, params):
    db = app.db
    result = await db.execute(User.select().where(User.is_delete==0).dicts())
    # 注意：
    # 1.result是一个可迭代对象，在response中自动转化为了list
    # 2.里面的create_time是datetime类型，在response中自动转化为了string
    return 1, dict(msg="ok", data=result)

async def register_api(app, params):
    pass


async def login_api(request, params):
    request.ctx.session["user_id"] = 1
    return 1, dict(msg="ok", data={"user_id": 1})
```

## common

`const.py`

```python
class RC:
    OK = 0

    NO_AUTH = 10005
    NOT_LOGIN = 10006

    PARAMS_INVALID = 11001

    INTERNAL_ERROR = 15001
    HTTP_ERROR = 15002
    DB_ERROR = 15003
    REDIS_ERROR = 15004

```

`utils.py`

```python
import string
import time as _time
import random
from datetime import datetime, date,timedelta
from functools import wraps
from inspect import isawaitable, getabsfile
from sanic.response import json
from sanic.request import Request
from sanic.exceptions import InvalidUsage
from sanic.views import HTTPMethodView
from sanic.log import logger
from sanic.kjson import json_loads, json_dumps
from common.const import RC


def res_ok(code=RC.OK, msg="ok", data=''):
    return json(dict(code=code, msg=msg, data=data))


def res_ng(code=RC.PARAMS_INVALID, msg='', data=''):
    return json(dict(code=code, msg=msg, data=data))


def build_random_str(l):
    t = str(int(_time.time()))
    x = string.digits + string.ascii_letters + string.punctuation
    k = l - len(t)
    return t + ''.join(random.choice(x) for i in range(k))

def check_params(r_k, params):
    l_k, r_k_v = [], {}
    for k in r_k:
        v = params.get(k, None)
        if v is not None:
            r_k_v[k] = v
        else:
            l_k.append(k)
    assert not l_k, f"{','.join(l_k)}参数缺失"
    return r_k_v


def get_rest_seconds():
    now = datetime.now()
    today_begin = datetime(now.year, now.month, now.day, 0, 0, 0)
    tomorrow_begin = today_begin + timedelta(days=1)
    rest_seconds = (tomorrow_begin - now).seconds
    return rest_seconds

def login_required(func):
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        prefix = "jwt"
        # 验证头信息的token信息是否合法
        if not auth_header:
            return res_ng(code=RC.NOT_LOGIN)
        auth_data = auth_header.partition(prefix)[-1].strip()
        if not auth_data:
            return res_ng(code=RC.NOT_LOGIN)
        # 解密
        result = parse_payload(auth_data)
        if not result['status']:
            return res_ng(code=RC.NO_AUTH)
        # 将解密后数据赋值给user_info
        request.ctx.user_id = result.get("data", {}).get("user_id")
        return await func(request, *args, **kwargs)

    return wrapper

def check_auth(role_ids=[], method="GET"):
    def _wrapper(func):
        @wraps(func)
        async def handler(*args, **kwargs):
            if len(args) >= 1 and isinstance(args[0], Request):
                request = args[0]
            elif len(args) >= 2 and isinstance(args[0], HTTPMethodView) and isinstance(args[1], Request):
                request = args[1]
            else:
                raise InvalidUsage("Can't decorate a bad handler")
  
            auth_user = request.ctx.auth_user
    				role_id = auth_user.get("role_id")
            brand_code = auth_user.get("brand_code")
            try:
                params = request.args if method == "GET" else request.json
            except Exception as e:
                params = {}
            if role_id in role_ids:
                r_brand_code = params.get("brand_code")
                if r_brand_code:
                    if brand_code != "99" and r_brand_code != brand_code:
                        return res_ng(code=RC.NO_AUTH, msg="无权限")
                return await func(*args, **kwargs)
            else:
                return res_ng(code=RC.NO_AUTH, msg="无权限")

        return handler

    return _wrapper

# 函数缓存
def cache_to_date(ttl=120):
    def warpper_(func):
        @wraps(func)
        async def handler(app, *args, to_date, **kwargs):
            key, use_cache = "", (to_date < date.today())
            if use_cache:
                key = f"{func.__name__}@{getabsfile(func)}={':'.join(map(str, args))}:{to_date}:{kwargs}"
                got = await app.redis.get(key)
                if got:
                    logger.info(f"cache-hit: {key}")
                    return json_loads(got)
            result = func(app, *args, to_date=to_date, **kwargs)
            got = await result if isawaitable(result) else result
            if use_cache:
                logger.info(f"cache-new: {key}")
                await app.redis.setex(key, ttl, json_dumps(got))
            return got

        return handler

    return warpper_
  
# 数据缓存->数据库
async def get_conf_content(cache_db, db, *keys):
    if not keys:
        return {}

    key_list = [item for item in keys]
    key_val = {}

    # 获取缓存
    contents = await cache_db.mget(*keys)
    no_val_keys = []
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"read {keys} from redis: {contents}")
    for i, x in enumerate(contents):
        if x:
            key_val[key_list[i]] = ujson.loads(x)
        else:
            no_val_keys.append(key_list[i])
    # 数据库
    if no_val_keys:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"need read from mysql: {no_val_keys}")
        db_res = await db.execute(ConfigModel.select().where(
            (RCKey.project_name + ConfigModel.conf_key) in no_val_keys, ConfigModel.is_delete == 0))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"read from mysql result: {len(db_res)}")
        for x in db_res:
            store_key = RCKey.project_name + x.conf_key
            key_val[store_key] = x.content
            await cache_db.setex(store_key, 1 * 60, ujson.dumps(x.content))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"setex: {x.conf_key}, {x.content}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"read from mysql: {x}")
    return key_val
```

`jwt.py`

```python
import jwt
import datetime
from jwt import exceptions

# 加的盐
JWT_SALT = "xxx"


def create_token(payload, timeout=20):
    # 声明类型，声明加密算法
    headers = {
        "type": "jwt",
        "alg": "HS256"
    }
    # 设置过期时间
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=20)
    result = jwt.encode(payload=payload, key=JWT_SALT, algorithm="HS256", headers=headers).decode("utf-8")
    # 返回加密结果
    return result


def parse_payload(token):
    """
    用于解密
    :param token:
    :return:
    """
    result = {"status": False, "data": None, "error": None}
    try:
        # 进行解密
        verified_payload = jwt.decode(token, JWT_SALT, True)
        result["status"] = True
        result['data'] = verified_payload
    except exceptions.ExpiredSignatureError:
        result['error'] = 'token已失效'
    except jwt.DecodeError:
        result['error'] = 'token认证失败'
    except jwt.InvalidTokenError:
        result['error'] = '非法的token'
    return result


if __name__ == '__main__':
    a = {"id": 10}
    res = create_token(a)
    print(res)
```

## conf

`/conf/test.py`

```python
PARAM_FOR_MYSQL = dict(
    database="db_iwc_badge",
    user='root',
    password='xxx',
    host='xx.xx.xx.xx',
    port=3306,
    charset='utf8mb4',
    max_connections=5,
)

PARAM_FOR_REDIS = dict(
    address=("xx.xx.xx.xx", 6379),
    db=5,
    password="xxx",
    minsize=8,
    maxsize=32,
)


PARAM_FOR_CLICKHOUSE = dict(
    url="xxxx",
    user="xxx",
    password="xxx",
    database="xxx",
)


COOKIE_NAME = "__sid__"
COOKIE_PREFIX = "iwc"
```

## models

`base.py`

```python
from mtkext.db import JSONField, MediumJSONField, patch_for_mysql
from peewee import Model, Proxy, SQL, fn
from peewee import AutoField, CharField, PrimaryKeyField, IntegerField, DateField, FloatField, SmallIntegerField, \
    TextField, BooleanField, DateTimeField, TimeField

auto_update = SQL("ON UPDATE CURRENT_TIMESTAMP")
auto_create = SQL("DEFAULT CURRENT_TIMESTAMP")

auto_create_update = SQL("DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP")

patch_for_mysql()
db_proxy = Proxy()


class BaseModel(Model):
    class Meta:
        databse = db_proxy
        db_table = "%"

    create_time = DateTimeField(null=False, index=True, constraints=[auto_create], help_text='创建时间')
    update_time = DateTimeField(null=True, index=True, constraints=[auto_update], help_text='更新时间')
```

`model.py`

```python
from .base import *


class User(BaseModel):
    class Meta:
        database = db_proxy
        db_table = "t_user"

    id = AutoField()
    user_name = CharField(max_length=32, help_text="姓名")
    gender = SmallIntegerField(default=1, help_text="性别1男2女")
    birth_day = DateField(help_text="出生年月日")
    province = CharField(max_length=16, help_text="省编码")
    city = CharField(max_length=16, help_text="城市编码")
    phone = CharField(index=True, unique=True, max_length=16, help_text="手机号")
```

## main

```python
from importlib import import_module

from sanic import Sanic, Blueprint
from sanic.log import logger
from sanic.exceptions import SanicException

from common.const import RC
from common.utils import res_ok, res_ng


app = Sanic("demo")


@app.listener('before_server_start')
async def init_server(app, loop):
    from sanic.config import Config
    conf = Config()
    conf.update_config("conf/" + app.cmd_args.env + ".py")
    app.conf = conf

    # init blueprints
    _url_prefix = f'{app.cmd_args.env}/iwc/badge'
    init_blueprint(app, _url_prefix)

    # init http
    from mtkext.hcp import HttpClientPool
    app.http = HttpClientPool(loop=loop)

    # init redis
    import aioredis
    app.redis = await aioredis.create_redis_pool(
        **conf.PARAM_FOR_REDIS, loop=loop
    )

    # init db
    from models.model import db_proxy
    from peewee_async import Manager, PooledMySQLDatabase
    pooled_db = PooledMySQLDatabase(**conf.PARAM_FOR_MYSQL)
    db_proxy.initialize(pooled_db)
    app.db = Manager(db_proxy, loop=loop)

    from models import model
    from mtkext.db import create_all_tables
    create_all_tables(model, [])
    # init chc
    from aiochclient import ChClient
    app.chc = ChClient(**conf.PARAM_FOR_CLICKHOUSE)

    # init session
    from mtkext.ses import install_session
    install_session(app, app.redis, prefix=conf.COOKIE_PREFIX, cookie_name=conf.COOKIE_NAME, expiry=86400)
    
    # init localCache
    from mtkext.cache import LocalCache
    app.cache = LocalCache

    # init aiojobs
    import aiojos
    app.jobs = await aiojobs.create_scheduler(close_timeout=0.5)
    await app.jobs.spawn(test(app))  # 项目启动时执行的异步任务
    await app.jobs.spawn(cron_job_by_daily_time(app, test))  # 每日定时执行的脚本
    await app.jobs.spawn(cron_job_by_interval(app, test, 60*60))  # 固定间隔执行的脚本
        

@app.listener('after_server_start')
async def check_server(app, loop):
    for uri, route in app.router.routes_all.items():
        method = "|".join(route.methods)
        logger.info(f"{uri} ==({method})==> {route.name}")


@app.listener('after_server_stop')
async def kill_server(app, loop):
    await app.jobs.close()
    await app.chc.close()
    await app.db.close()
    app.redis.close()
    await app.redis.wait_closed()
    await app.http.close()


@app.exception(Exception)
def catch_exceptions(request, ex):
    logger.exception(ex)
    if isinstance(ex, SanicException):
        status = 51000 + ex.status_code
        return res_ng(code=status, msg=str(ex))
    else:
        msg = ex.__class__.__name__ + ": " + str(ex)
        return res_ng(code=50000, msg=msg)
      

@app.middleware("request")
async def middle_request(request):
    path = request.path
    if path != "/":
        _uuid = uuid1().hex
        request.ctx.request_id = _uuid
        url = request.url
        logger.info(f"\n===========>{_uuid}\nurl:{url}\n===========\n")

    
@app.middleware('request')
async def auth(request):
    url = request.url
    no_login = ["/login", "/register"]
    if next((0 for x in no_login if x in url), 1):
        user_id = request.ctx.session.get('user_id')
        if not user_id:
            return res_ng(code=RC.NOT_LOGIN, msg="用户未登录")
        request.ctx.user_id = user_id
        
@app.middleware('response')
async def log_info(request, response):
    try:
        url = request.url
        if request.method == "POST" and request.headers.get("Content-Type").startswith("application/json"):
            req = request.json
        else:
            req = request.body
        rep = response.body
        rep = kjson.json_loads(rep.decode('utf-8')) if rep else ""
        logger.info(f"\n===========<{request_id}\nurl=>{url}\nrequest=>{req}\nresponse=>{rep}\n============\n")
    except Exception as e:
        logger.exception(e)
        

def init_blueprint(app, url_prefix):
    bps = []
    from glob import glob
    for subname in glob("api/*.py"):
        if not subname.endswith('.py'):
            continue
        modename = subname.replace('/', '.')[:-3]
        module = import_module(modename)
        bp = getattr(module, 'bp', None)
        if bp: bps.append(bp)
    bp_group = Blueprint.group(*bps, url_prefix=url_prefix)
    app.blueprint(bp_group)

        
async def cron_job_by_interval(app, method, interval=60):
    import time
    import asyncio
    p_stamp = 0
    while not app.jobs.closed:
        now = int(time.time())
        if now - p_stamp >= interval:
            try:
                await method(app)
            except Exception as ex:
                logger.exception(ex)
            p_stamp = now
        await asyncio.sleep(0.3)


async def cron_job_by_daily_time(app, method, delta_time=0):
    import time
    from datetime import datetime
    import asyncio
    while not app.jobs.closed:
        now = int(time.time())
        day_start = time.mktime(datetime.now().date().timetuple())
        if delta_time < now - day_start <= delta_time + 60:
            try:
                await method(app)
            except Exception as ex:
                logger.exception(ex)
            left_time = 86400 - (now - day_start) + delta_time + 30
            await asyncio.sleep(left_time)
        await asyncio.sleep(1)
        
async def fetch_token(request):
    cookie_name = request.app.conf.COOKIE_NAME
    token = request.cookies.get(cookie_name)
    if token == 'null':
        token = None
    if not token:
        token = request.headers.get('Authorization')
        if not token:
            return
        request.cookies[cookie_name] = token
        await request.app.extensions.get("session").setup(request, token)
    return token
```

后台执行任务

```python
# 方法一：使用aiojob
await app.jobs.spawn(test(app))
# 方法二：sanic支持add_task
app.add_task(func())
```

## bash

`run.sh`

```shell
#!/bin/bash

# === 1. Customize Arguments (DO CHANGES) ===
#module of app=Sanic(...)
SANICAPP=main.app

#environ key: debug|test|prod
MICROENV=prod

#host (ip) for listening
LISTENHOST=0.0.0.0

#base listening port
PORTBASE=9600

#reuse port when greater than 1
WORKERS=1

#DEBUG|INFO|WARNING|ERROR|NONE
LOGLEVEL=INFO

#debug mode of sanic framework, can be: [0,1]
DEBUGMODE=0

#extra path for importing, can be empty
INCLUDEDIR=


# === 2. Initialize Variables (DO NOT CHANGE) ===
MKTDIR=$(cat "$HOME/.path_microkit" 2>/dev/null)
if [ ! -d "$MKTDIR" ]; then
  echo "Missing or Invalid .path_microkit!"
  echo "Should enter into MTK and run setup.sh"
  exit 2
fi
source $MKTDIR/run.source
```

`test.sh`

```shell
#!/bin/bash

CURPATH=$(cd `dirname $0` && pwd)
export MTKEXPR='MICROENV=test PORTBASE=9700 DEBUGMODE=1 LOGLEVEL=DEBUG' && $CURPATH/run.sh $@
```

