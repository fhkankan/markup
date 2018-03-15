#组装数据为html格式

##my_web.py(更新)

```
import pymysql
import time
import os
import re

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

@route("/index.html")
def index(file_name):
    """返回index.html需要的页面内容"""
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


@route("/center.html")
def center(file_name):
    """返回center.html需要的页面内容"""
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


def application(environ, start_response):
    status = '200 OK'
    response_headers = [('Content-Type', 'text/html')]
    start_response(status, response_headers)

    file_name = environ['PATH_INFO']
    try:
        return g_url_route[file_name](file_name)
    except Exception as ret:
        return "%s" % ret
    else:
        return str(environ) + '-----404--->%s\n'
```