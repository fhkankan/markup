#3-替换模板

##dynamic/my_web.py

```
import time
import os
import re

template_root = "./templates"


def index(file_name):
    """返回index.py需要的页面内容"""
    # return "hahha" + os.getcwd()  # for test 路径问题
    try:
        file_name = file_name.replace(".py", ".html")
        f = open(template_root + file_name)
    except Exception as ret:
        return "%s" % ret
    else:
        content = f.read()
        f.close()

        # --------更新-------
        data_from_mysql = "数据还没有敬请期待...."
        content = re.sub(r"\{%content%\}", data_from_mysql, content)

        return content


def center(file_name):
    """返回center.py需要的页面内容"""
    # return "hahha" + os.getcwd()  # for test 路径问题
    try:
        file_name = file_name.replace(".py", ".html")
        f = open(template_root + file_name)
    except Exception as ret:
        return "%s" % ret
    else:
        content = f.read()
        f.close()

        # --------更新-------
        data_from_mysql = "暂时没有数据,,,,~~~~(>_<)~~~~ "
        content = re.sub(r"\{%content%\}", data_from_mysql, content)

        return content

def application(environ, start_response):
    status = '200 OK'
    response_headers = [('Content-Type', 'text/html')]
    start_response(status, response_headers)

    file_name = environ['PATH_INFO']
    if file_name == "/index.py":
        return index(file_name)
    elif file_name == "/center.py":
        return center(file_name)
    else:
        return str(environ) + '==Hello world from a simple WSGI application!--->%s\n' % time.ctime()
```