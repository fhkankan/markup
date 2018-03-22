#mysql-改

##my_web.py(修改)

```
import pymysql
import time
import os
import re
import sys

template_root = "./templates"

# 用来存放url路由映射
# url_route = {
#   "/index.py":index_func,
#   "/center.py":center_func
# }
g_url_route = dict()


def route(url):
    def func1(func):
        # 添加键值对，key是需要访问的url，value是当这个url需要访问的时候，需要调用的函数引用
        g_url_route[url]=func
        def func2(file_name):
            return func(file_name)
        return func2
    return func1


@route(r"/index.html")
def index(file_name, url=None):
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

        # data_from_mysql = "暂时没有数据，请等待学习mysql吧，学习完mysql之后，这里就可以放入mysql查询到的数据了"
        db = pymysql.connect(host='localhost',port=3306,user='root',password='mysql',database='stock_db',charset='utf8')
        cursor = db.cursor()
        sql = """select * from info;"""
        cursor.execute(sql)
        data_from_mysql = cursor.fetchall()
        cursor.close()
        db.close()

        html_template = """
            <tr>
                <td>%d</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>
                    <input type="button" value="添加" id="toAdd" name="toAdd" systemidvaule="%s">
                </td>
                </tr>"""

        html = ""

        for info in data_from_mysql:
            html += html_template % (info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[7], info[1])


        content = re.sub(r"\{%content%\}", html, content)

        return content


@route(r"/center.html")
def center(file_name, url=None):
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

        # data_from_mysql = "暂时没有数据,,,,~~~~(>_<)~~~~ "
        db = pymysql.connect(host='localhost',port=3306,user='root',password='mysql',database='stock_db',charset='utf8')
        cursor = db.cursor()
        sql = """select i.code,i.short,i.chg,i.turnover,i.price,i.highs,j.note_info from info as i inner join focus as j on i.id=j.info_id;"""
        cursor.execute(sql)
        data_from_mysql = cursor.fetchall()
        cursor.close()
        db.close()

        html_template = """
            <tr>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>%s</td>
                <td>
                    <a type="button" class="btn btn-default btn-xs" href="/update/%s.html"> <span class="glyphicon glyphicon-star" aria-hidden="true"></span> 修改 </a>
                </td>
                <td>
                    <input type="button" value="删除" id="toDel" name="toDel" systemidvaule="%s">
                </td>
            </tr>
            """

        html = ""

        for info in data_from_mysql:
            html += html_template % (info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[0], info[0])

        content = re.sub(r"\{%content%\}", html, content)

        return content


@route(r"/update/(\d*)\.html")
def update(file_name, url):
    """显示 更新页面的内容"""
    try:
        template_file_name = template_root + "/update.html"
        f = open(template_file_name)
    except Exception as ret:
        return "%s,,,没有找到%s" % (ret, template_file_name)
    else:
        content = f.read()
        f.close()

        ret = re.match(url, file_name)
        if ret:
            stock_code = ret.group(1)
        else:
            stock_code = 0

        # ------ 添加 -------
        db = pymysql.connect(host='localhost',port=3306,user='root',password='mysql',database='stock_db',charset='utf8')
        cursor = db.cursor()
        # 会出现sql注入，怎样修改呢？ 参数化
        sql = """select focus.note_info from focus inner join info on focus.info_id=info.id where info.code=%s;""" % stock_code
        cursor.execute(sql)
        stock_note_info = cursor.fetchone()
        cursor.close()
        db.close()

        # ------ 修改 -------
        content = re.sub(r"\{%code%\}", stock_code, content)
        content = re.sub(r"\{%note_info%\}", str(stock_note_info[0]), content)

        return content


# ------ 添加 -------
@route(r"/update/(\d*)/(.*)\.html")
def update_note_info(file_name, url):
    """进行数据的真正更新"""
    stock_code = 0
    stock_note_info = ""

    ret = re.match(url, file_name)
    if ret:
        stock_code = ret.group(1)
        stock_note_info = ret.group(2)

    db = pymysql.connect(host='localhost',port=3306,user='root',password='mysql',database='stock_db',charset='utf8')
    cursor = db.cursor()
    # 会出现sql注入，怎样修改呢？ 参数化
    sql = """update focus inner join info on focus.info_id=info.id set focus.note_info="%s" where info.code=%s;""" % (stock_note_info, stock_code)
    cursor.execute(sql)
    db.commit()
    cursor.close()
    db.close()

    return "修改成功"


def app(environ, start_response):
    status = '200 OK'
    response_headers = [('Content-Type', 'text/html')]
    start_response(status, response_headers)

    file_name = environ['PATH_INFO']
    try:
        for url, call_func in g_url_route.items():
            print(url)
            ret = re.match(url, file_name)
            if ret:
                return call_func(file_name, url)
                break

        else:
            return "没有访问的页面--->%s" % file_name

    except Exception as ret:
        return "%s" % ret

    else:
        return str(environ) + '-----404--->%s\n'
```