# web项目

版本：20.12.6

## 目录结构

```
-api  # api文件夹
 |-user.py
-biz  # server文件夹
 |-user.py
-common
 |-const.py
 |-utils.py
 |-ext_cache.py
 |-ext_db.py
 |-ext_hcp.py
 |-ext_log.py
-conf	# 配置文件，注意文件名与bash文件一致
 |-dev.py
 |-prod.py
-docs  # 接口文档
-logs  # 日志，项目启动自动生成
-models	# 模型类文件夹
 |-base.py
 |-model.py
main.py
```

## api

`user.py`

```python
from sanic import Blueprint


from common.const import RC
from common.utils import res_ng, res_ok, check_params, check_auth
from common.ext_log import logger

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
from playhouse.shortcuts import model_to_dict
from models.model import User
from common.ext_log import logger


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

def check_params(r_k, o_k, params):
    l_k, r_k_v, o_k_v = [], {}, {}
    for k in r_k:
        v = params.get(k)
        if v is not None:
            r_k_v[k] = v
        else:
            l_k.append(k)
    assert not l_k, f"{','.join(l_k)}参数缺失"
    o_k_v = {k: params.get(k) for k in o_k if params.get(k) is not None}
    o_k_v.update(r_k_v)
    return o_k_v

def encrypt_passwd(salt, password):
    h = md5()
    h.update((salt + password).encode('utf-8'))
    password_encrypt = h.hexdigest()
    return password_encrypt


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

接口签名验证

```python
channel_vendor = {
    "1021": {
        "name": "test",
        "apps": {
            "mall": dict(key="Ba185qbFDuzPu", ip={"127.0.0.1"}),
        },
    },
}



def check_auth(request, table, myapp):
    hdr = request.headers
    ###
    vendor_id = hdr.get("X-EACH-VENDOR-ID")
    if not vendor_id: return 901, "http头缺少X-EACH-VENDOR-ID"
    vendor = table.get(vendor_id)
    if vendor is None: return 902, "错误的vendor_id"
    ###
    appid = hdr.get("X-EACH-APP-ID")
    if not appid: return 901, "http头缺少X-EACH-APP-ID"
    if appid != myapp: return 900, "appid不匹配"
    app = vendor["apps"].get(appid)
    if app is None: return 902, "错误的appid"
    ###
    signature = hdr.get("X-EACH-SIGNATURE")
    if not signature: return 901, "http头缺少X-EACH-SIGNATURE"
    ###
    logger.info((request.ip, request.remote_addr))
    if app["ip"] and request.ip not in app["ip"]:
        return 907, f"未被允许的ip: {request.ip}"
    ###
    todos = [request.path]
    if request.query_string:
        todos.append("?" + request.query_string)
    if request.method.upper() == "POST":
        todos.append(request.body.decode())
    todos.append(app["key"])
    bstr = "".join(todos).encode()
    hashcode = hashlib.sha256(bstr).hexdigest()
    if signature == hashcode: return 0, "ok"
    ###
    logger.error(f"check-sign-failed: {signature} != {hashcode} <== {bstr}")
    return 904, "签名不匹配"
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

`ext_cache.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
import time


class LocalCache(object):

    @staticmethod
    def current():
        return int(time.time())

    def __init__(self):
        self._storage = OrderedDict()
        self._purged = self.current()

    def __repr__(self):
        return f'LocalCache: {self._storage}'

    def clear(self):
        self._storage.clear()
        self._purged = self.current()

    def get(self, key, delta=60):
        item = self._storage.get(key)
        if item and self.current() < item[0] + delta:
            return item[1]

    def pop(self, key):
        return self._storage.pop(key, None)

    def set(self, key, value):
        self._storage[key] = (self.current(), value)
        self._storage.move_to_end(key)

    def purge(self, delta=300, interval=30):
        now = self.current()
        if now < self._purged + interval: return
        ###
        overdues = list()
        for key, item in self._storage.items():
            if now < item[0] + delta: break
            overdues.append(key)
        ###
        for key in overdues:
            del self._storage[key]
        self._purged = now

```

`ext_db.py`

```python
#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from peewee import TextField
from peewee import Model
import copy
import re
from functools import partial
from inspect import isclass
from binascii import crc32
from json import dumps, loads

import logging



class JSONField(TextField):

    def db_value(self, value):
        if value not in (None, ""):
            return dumps(value, ensure_ascii=False)

    def python_value(self, value):
        if value not in (None, ""):
            try:
                return loads(value)
            except Exception as ex:
                logging.error(f"{self}字段 {ex}: {value}")


class MediumJSONField(JSONField):
    field_type = "MEDIUMTEXT"


class MediumTextField(TextField):
    field_type = "MEDIUMTEXT"



def is_model(o):
    return isclass(o) and issubclass(o, Model)


def create_all_tables(models, excludes=[]):
    for subname in dir(models):
        t = getattr(models, subname)
        if not is_model(t): continue
        if t is Model or t in excludes: continue
        ###
        tname = t._meta.table_name
        if not tname: continue
        if tname.find("%") >= 0: continue
        if t.table_exists(): continue
        ###
        try:
            t.create_table(safe=True)
        except Exception as ex:
            logging.error(f"create-failed on {tname}")
            logging.exception(ex)
        b = getattr(t._meta, "auto_id_base", None)
        if type(b) == int:
            t.raw(f"ALTER TABLE {tname} AUTO_INCREMENT={b};").execute()


def get_bucket_id(s, bucket_size=32):
    v = s if type(s) == bytes else s.encode()
    code = (crc32(v) & 0xffffffff)
    return "%3.3d" % (code % bucket_size)


class FlexModel:
    """
    动态Model的类管理器，使用方式：
        cls = FlexModel.get(MyModel, 123)
    说明：
    1. 这里的MyModel即是模板Model，cid=123为编号，可以是分桶编号或日期等
    2. get时自动判断类MyModel_123是否存在：存在即返回，不存在则创建模型类和实体表
    3. 按模板创建的动态类，将继承MyModel模板类的所有Field和Meta类定义（除了table_name）
    4. 若希望获取表名，可以使用：cls._meta.table_name
    """
    _this_mapper = {}

    @staticmethod
    def get(TemplateModel, cid, db=None, create_if_not_exist=True):
        assert is_model(TemplateModel), "模板类必须继承于peewee.Model"
        ###
        base_class_name = TemplateModel.__name__
        class_name = "%s_%s" % (base_class_name, cid)
        ###
        one = FlexModel._this_mapper.get(class_name)
        if one: return one
        ###
        _meta = TemplateModel._meta
        _old = _meta.table_name
        _neu = ("%s_%s" % (_old, cid)) if _old.find("%") < 0 else (_old % cid)
        ###
        if not db: db = _meta.database
        class Meta:
            database = db
            schema = _meta.schema
            table_name = _neu
            indexes = copy.deepcopy(_meta.indexes or ())
        ###
        one = type(class_name, (TemplateModel,), dict(Meta=Meta))
        if create_if_not_exist:
            try:
                db.connection().ping(reconnect=True)
                one.create_table(safe=True) #注意：这里使用同步调用来创建
            except Exception as ex:
                logging.exception(ex)
                return None
        FlexModel._this_mapper[class_name] = one
        return one


try:
    import pymysql as _mysql
    def mysql_escape_string(s):
        return _mysql.escape_string(s)
except ImportError:
    pass


try:
    from psycopg2 import extensions as _pg_extensions
    def pg_escape_string(s):
        return _pg_extensions.adapt(s).getquoted()
except ImportError:
    pass


class safe_string(object):

    def __init__(self, source):
        self.__source = source

    def __str__(self):
        return str(self.__source)


async def sql_execute(db, sql):
    matched = re.search(r'\w+', sql)
    assert matched, "Invalid SQL, can not find keyword"
    kw = matched.group().upper()
    ###
    cursor = await db.cursor_async()
    try:
        await cursor.execute(sql)
    except Exception as ex:
        await cursor.release()
        if ex.__class__.__name__ == "OperationalError":
            await db.close_async()
        raise
    ###
    try:
        if kw == "SELECT":
            rows = []
            while True:
                one = await cursor.fetchone()
                if not one: return rows
                rows.append(one)
        elif kw == "INSERT":
            return await db.last_insert_id_async(cursor)
        elif kw in ("UPDATE", "DELETE"):
            return cursor.rowcount
    finally:
        await cursor.release()



def patch_for_mysql():
    from peewee import Field, SQL
    if getattr(Field, "_old_ddl", None) is None:
        Field._old_ddl = old_ddl = Field.ddl
        def new_ddl(self, ctx):
            result = old_ddl(self, ctx)
            if self.help_text:
                if self.help_text.find("%") >= 0:
                    self.help_text = self.help_text.replace('%', "@")
                    logging.warning(f"{self}字段的help_text包含百分号已被替换")
                if self.help_text.find("'") >= 0:
                    self.help_text = self.help_text.replace("'", '"')
                    logging.warning(f"{self}字段的help_text包含单引号已被替换")
                result.nodes.append(SQL("COMMENT '%s'" % self.help_text))
            return result
        Field.ddl = new_ddl


# PATCH for default cast() of the SQL clause: "" ==> 0
# In the case, we should fill NULL for empty string
the_cast_keys = ("INT", "BIGINT", "SMALLINT", "BOOL", "FLOAT", "DOUBLE", "DECIMAL")


def peewee_normalize_dict(ThatModel, obj, excludes=[]):
    record = {}
    for k, v in obj.items():
        if v is None or k in excludes: continue
        if getattr(Model, k, None):
            logging.warning(f"与Model关键字冲突，忽略key={k}：{obj}")
            continue
        ###
        key = getattr(ThatModel, k, None)
        if key is None or (v == "" and key.field_type in the_cast_keys):
            logging.debug(f"忽略key={k}，获取field.key={key}")
            continue
        ###
        record[key] = v
    return record


def pop_matched_dict(nodes, key, val):
    for i, tnode in enumerate(nodes):
        if tnode[key] == val:
            return nodes.pop(i)


def get_or_sep_list(args, key, conv_type=str):
    items = args.getlist(key, []) # args: request.args
    if len(items) == 1:
        items = [i.strip() for i in items[0].split(",")]
    if conv_type is not str:
        items = [conv_type(i) for i in items if i != ""]
    return items


# redis按通配符批量删除：采用了SCAN，避免KEYS指令
_script_batch_delete = """
local c = 0
local tlist = nil
local done = 0
repeat
    local resp = redis.call('SCAN', c, 'MATCH', ARGV[1], 'COUNT', 100)
    c = tonumber(resp[1])
    tlist = resp[2]
    if (tlist ~= nil and next(tlist) ~= nil) then
        local num = redis.call('DEL', unpack(tlist))
        if (num > 0) then done = done + num end
    end
until(c == 0)
return done
"""
async def redis_batch_delete(redis, wildcard):
    return await redis.eval(_script_batch_delete, args=[wildcard])


_script_incr_limit = """
local resp = redis.call('INCRBY', ARGV[1], ARGV[2])
local cnt = tonumber(resp)
if (cnt > tonumber(ARGV[3])) then
    redis.call('DECRBY', ARGV[1], ARGV[2])
    return false
end
return true
"""

async def redis_incr_with_limit(redis, key, delta, limit):
    return await redis.eval(_script_incr_limit, args=[key, delta, limit])


async def redis_fuzzy_find(redis, match, max_keys=1000):
    cur, results = b'0', []
    while cur:
        cur, matched = await redis.scan(cur, match=match, count=100)
        if matched: results.extend(matched)
        if len(results) >= max_keys: break
    return results


async def redis_find_get(redis, match, max_keys=1000):
    keys = await redis_fuzzy_find(redis, match, max_keys)
    tlist = await redis.mget(*keys)
    results = []
    for key, bstr in zip(keys, tlist):
        try:
            obj = loads(bstr)
            results.append((key.decode(), obj))
        except Exception as ex:
            logging.error(ex)
    return results

```

`ext_hcp.py`

```python
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import asyncio
from aiohttp import ClientSession, DummyCookieJar, FormData
from aiohttp import ServerTimeoutError, ClientResponseError
from sanic.log import logger
from json import dumps, loads
from base64 import b64encode
from io import BytesIO
from .utils import get_random_str, detect_image_type

__all__ = ["HttpClientPool"]


_ua_default = 'Mozilla/5.0 Gecko/20100101 Firefox/53.0 (PYMTK-HCP-2.1)'

_get_headers = {
    'User-Agent': _ua_default,
}

_post_headers = {
    "Content-Type": "application/json; charset=utf-8",
    'User-Agent': _ua_default,
}

def _merge_dict(input, default):
    if not input: return default
    return dict(default, **input)

def _format_error(ans, got):
    try:
        body = got.decode()
    except:
        body = b64encode(got).decode()
    ###
    return dict(errcode=ans.status, errmsg=ans.reason, body=body)


class HttpClientPool(object):

    def __init__(self, loop=None, client=None):
        self.__client = client or ClientSession(loop=loop,
            cookie_jar=DummyCookieJar(), json_serialize=dumps)
        self.__headers = {}
        self.__errinfo = {}
        self.__status = 500
        self.__reason = ""

    @property
    def raw(self):
        return self.__client

    @property
    def headers(self):
        return self.__headers

    @property
    def errinfo(self):
        return self.__errinfo

    @property
    def status(self):
        return self.__status

    @property
    def reason(self):
        return self.__reason

    async def close(self):
        await self.__client.close()
        self.__client = None

    def passthrough(method):
        def inner(self, *args, **kwargs):
            if self.__client is None:
                raise AttributeError('Cannot use NULL client.')
            return getattr(self.__client, method)(*args, **kwargs)
        return inner

    # Allow to be used as a context-manager: async with ... as ...
    __aenter__ = passthrough('__aenter__')
    __aexit__ = passthrough('__aexit__')

    async def _request_with_retry(self, method, url, parse_with_json, **kwargs):
        kw = dict(timeout=5, ssl=kwargs.pop("verify_ssl", False)) # deprecated!
        kw.update(kwargs)
        default = _post_headers if method == "POST" else _get_headers
        kw["headers"] = _merge_dict(kw.get("headers"), default)
        ###
        retry = kw.pop("retry", 0)
        retryWait = kw.pop("retryWait", 0.5)
        ctype = None if kw.pop("disableType", 0) else "application/json"
        ###
        while True:
            try:
                async with self.__client.request(method=method, url=url, **kw) as ans:
                    if parse_with_json:
                        if ans.status in (200, 206):  # 跳过204
                            got = await ans.json(loads=loads, content_type=ctype)
                            self.__headers = ans.headers
                            self.__status = ans.status
                            self.__reason = ans.reason
                            return got
                    got = await ans.read()
                    self.__headers = ans.headers
                    self.__status = ans.status
                    self.__reason = ans.reason
                    if ans.status in (200, 204, 206): return got
                    ###
                    self.__errinfo = _format_error(ans, got)
                    logger.warning(f"request-failed ({url}): {self.__errinfo}")
            except (asyncio.TimeoutError, ServerTimeoutError) as ex:
                logger.error(f"{ex}: {url}")
                self.__errinfo = dict(errcode=504, errmsg="Service is timeout")
            except ClientResponseError as ex:
                logger.error(ex)
                self.__errinfo = dict(errcode=500, errmsg=ex.message)
            except Exception as ex:
                logger.exception(ex)
                self.__errinfo = dict(errcode=500, errmsg=str(ex) or "Network failed")
            ###
            if retry <= 0: break
            await asyncio.sleep(retryWait)
            retry -= 1
            retryWait *= 2

    async def request(self, url, method="GET", parse_with_json=False, **kwargs):
        return await self._request_with_retry(method.upper(), url, parse_with_json, **kwargs)

    async def get(self, url, parse_with_json=True, **kwargs):
        return await self._request_with_retry("GET", url, parse_with_json, **kwargs)

    # If `obj` is not an object, we can set obj=None, and data=payload
    async def post(self, url, obj, parse_with_json=True, **kwargs):
        return await self._request_with_retry("POST", url, parse_with_json, json=obj, **kwargs)

    # `fields` is a list with dict elements like:
    # {"name":"field_key", "value": fp, "filename":"可选", "content_type": "可选", }
    async def upload(self, url, fields, parse_with_json=True, **kwargs):
        assert type(fields) is list, "only for: type(fields)=list"
        data = FormData()
        for one in fields: data.add_field(**one)
        headers = kwargs.pop("headers", {})
        headers["Content-Type"] = f"multipart/form-data; boundary={data._writer.boundary}"
        return await self._request_with_retry("POST", url, parse_with_json, data=data, headers=headers, **kwargs)

    async def upload_as_file(self, url, name, content, content_type=None, parse_with_json=True, **kwargs):
        fmt = detect_image_type(content) or "dat"
        filename = f"{get_random_str(12)}.{fmt}" # mock up a filename
        fp = BytesIO(content)
        form_data = FormData()
        form_data.add_field(name=name, value=fp, filename=filename, content_type=content_type)
        data = form_data() # build form-data into payload
        headers = kwargs.pop("headers", {})
        headers.update(data.headers)
        return await self._request_with_retry("POST", url, parse_with_json, data=data, headers=headers, **kwargs)


```

`ext_log`

```python
import logging
import logging.config
import sys

logger = logging.getLogger("sanic.access")


def _make_log_config(filename, count):
    current = dict(
        version=1,
        disable_existing_loggers=False,
        root={
            "level": "INFO",
            "handlers": ["root_console_handler"]
        },
        loggers={},
        handlers={
            "root_console_handler": {
                "class": "logging.StreamHandler",
                "formatter": "generic",
                "stream": sys.stdout,
            },
        },
        formatters={
            "generic": {
                "format": "[%(asctime)s] %(levelname)s (%(process)d) (%(pathname)s:%(lineno)d) %(message)s",
                "class": "logging.Formatter"
            },
        }
    )
    if filename:
        current.update(loggers={
                "sanic.access": {
                    "level": "DEBUG",
                    "handlers": ["access_file_handler"],
                    "propagate": False,
                    "qualname": "sanic.access",
                },
            })
        current["handlers"].update({
                "access_file_handler": {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "formatter": "generic",
                    "filename": filename,
                    "when": "midnight",
                    "interval": 1,
                    "backupCount": count,
                    "encoding": 'utf-8',
                },
            })
    return current


def get_log_filename(logfile, i):

    if logfile.find("{}") >= 0:
        return True, logfile.format(i)

    if logfile.find("%") >= 0:
        return True, logfile % i

    return False, logfile



_default_handler = {
    "class": "logging.handlers.TimedRotatingFileHandler",
    "formatter": "generic",
    "encoding": 'utf-8',
    "when": "midnight",
    "interval": 1,
    "backupCount": 30,
    "filename": "default-output.log",
}

def set_logger(param=None, level=logging.DEBUG, filename=None, count=30, more=None):

    if filename is not None:
        param = _make_log_config(filename, count)

    if type(param) is dict:
        if more is not None:
            gdict = getattr(more, "LOGGER_DICT", None)
            if gdict:
                for logger_name, level in gdict.items():
                    handler_name = f"{logger_name}_handler"
                    param["loggers"][logger_name] = dict(
                        level=level, handlers=[handler_name], propagate=False)
                    param["handlers"][handler_name] = dict(
                        _default_handler, filename=f"logs/{logger_name}.log")
            ###
            gloggers = getattr(more, "LOGGERS", {})
            ghandlers = getattr(more, "HANDLERS", {})
            if gloggers and ghandlers:
                param["loggers"].update(gloggers)
                for hander_name, kwargs in ghandlers.items():
                    param["handlers"][hander_name] = dict(_default_handler, **kwargs)
            ###
        logging.config.dictConfig(param)

    if level: logger.setLevel(level)
    return logger

init_logger = set_logger


"""
Some shortcut functions for invoke loggers quickly as:
```
if enabled_info():
    plog_info(f"Functions is called ok: {test()}")
```
Note that the argument of `plog_xxx()` will not be evaluated,
unless the testing function `enabled_xxx()` return `true`.
"""
def enabled_debug():
    return logger.isEnabledFor(logging.DEBUG)

def enabled_info():
    return logger.isEnabledFor(logging.INFO)

def enabled_warn():
    return logger.isEnabledFor(logging.WARNING)

def enabled_error():
    return logger.isEnabledFor(logging.ERROR)

def enabled_critical():
    return logger.isEnabledFor(logging.CRITICAL)


def plog_debug(msg, *args, **kwargs):
    logger._log(logging.DEBUG, msg, args, **kwargs)

def plog_info(msg, *args, **kwargs):
    logger._log(logging.INFO, msg, args, **kwargs)

def plog_warn(msg, *args, **kwargs):
    logger._log(logging.WARNING, msg, args, **kwargs)

def plog_error(msg, *args, **kwargs):
    logger._log(logging.ERROR, msg, args, **kwargs)

def plog_critical(msg, *args, **kwargs):
    logger._log(logging.CRITICAL, msg, args, **kwargs)


enabled_warning = enabled_warn
enabled_fatal = enabled_critical
plog_warning = plog_warn
plog_fatal = plog_critical

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
from common.ext_db import JSONField, MediumJSONField, patch_for_mysql
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
from sanic.exceptions import SanicException

from common.const import RC
from common.utils import res_ok, res_ng
from common.ext_log import logger

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
async def middle_response(request, response):
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
        subname.replace("\\", "/")  # windows
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

