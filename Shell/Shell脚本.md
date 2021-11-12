# shell脚本

## 概述

- shell类型

```shell
cat /etc/default/useradd
# 里面有
SHELL=/bin/bash
```

- 创建脚本

常规创建

```shell
# 创建工具
vim/vi

# 脚本注释
单行注释	---> # 注释内容
多行注释	---> :<<! 注释内容！

# 示例
#!/bin/bash
# demo
echo '1'
:<<!
echo '2'
!
echo '3'
```

- 脚本执行

```shell
# 脚本文件本身没有可执行权限或者脚本首行没有命令解释器(推荐)
bash /path/to/script-name	或 /bin/bash /path/to/script-name

# 脚本文件具有可执行权限时使用
/path/to/script-name		或 ./script-name

# 加载shell脚本文件内容
source script-name			或 . script-name
```

 - 退出脚本

退出状态码

```shell
# 查看退出状态码
date
echo $?		# 展示上个命令的退出状态码

# 退出状态码值
0	# 命令成功结束
1  	# 一般性未知错误
2	# 不适合的shell命令
126	# 命令不可知性
127	# 没找到命令
128	# 无效的退出参数
128+x	# 与linux信号x相关的严重错误
130		# 通过Ctrl+C终止的命令
255		# 正常范围之外的退出状态码
```

脚本中指定退出状态码

```shell
exit 5
```

 - 脚本开发规范

```
- 脚本命令要有意义，文件后缀.sh
- 脚本文件首行是且必须是脚本解释器`#!bin/bash`
- 脚本文件解释器后面要有脚本的基本信息等内容
- 脚本文件常见执行方式`bash 脚本名`
- 脚本内容执行：从上到下，一次执行
- 代码书写优秀习惯(成对内容一次性写完，[]两端要有空格，流程控制语句一次性写完 )
- 通过缩进让代码容易易读
```
## 使用变量

### 环境变量

```shell
# 查看
env
set

# 定义
# 方法一：
变量=值
export 变量
# 方法二：
export 变量=值
```

### 用户变量

```shell
# 临时存储数据并在整个脚本中使用，可以是任何由字母数字或下划线组成的文本字符串，长度不超过20个
# 整体,中间无特殊字符
变量名=变量值
# 原样输出
变量名='变量值'
# 先解析，将结果与其他合成整体
变量名="变量值"
```

### 内置变量

```shell
$0	# 获取当前执行的shell脚本文件名，包括脚本路径
$n	# 获取当前执行的shell脚本的第n个参数值，0表示文件名，大于9就$(10)
$#	# 获取当前shell命令中参数的总个数
$?	# 获取执行上一个指令的返回值(0表示成功，非0为失败)
$$  # 当前shell的PID
```

### 变量操作

```shell
# 查看变量
# 方式一
$变量名
# 方式二
"$变量名"
# 方式三
${变量名}
# 方式四(推荐)
"${变量名}"

# 命令替换
变量名=`命令`
变量名=$(命令)
# 执行流程
1.执行'后者$()范围内的命令
2.将命令执行后的结果，赋值给新的变量名

# 取消变量
unset 变量名

# 默认值
# 若变量有内容，输出变量值；若变量内无内容，输出默认值
${变量名:-默认值}
# 无论变量是否有内容，均输出默认值
${变量名+默认值}

# 字符串精确截取
${变量名：起始位置：截取长度}
```

## 表达式

### 测试语句

```shell
# 方式一
test 条件表达式
# 方式二
[ 条件表达式 ]

# 条件成立，状态返回值是0，条件不成立，状态返回值是1
```

### 条件表达式

- 逻辑表达式

```shell
命令1 && 命令2
# 若命令1执行成功，才执行命令2；若命令1执行失败，那么命令2也不执行

命令1 || 命令2
# 若命令1执行成功，则命令2不执行；若命令1执行失败，则命令2执行
```

- 文件比较

```shell
-d file		# 检查file是否存在并是一个目录
-e file		# 检查file是否存在
-f file		# 检查file是否存在并是一个文件
-r file		# 检查file是否存在并可读
-s file		# 检查file是否存在并非空
-w file		# 检查file是否存在并可写
-x file		# 检查file是否存在并可执行
-O file		# 检查file是否存在并属当前用户所有
-G file		# 检查file是否窜在并默认组与当前用户相同

file1 -nt file2	# 检查file1是否比file2新
file1 -ot file2	# 检查file1是否比file2旧
```

- 数值比较

```shell
# 主要根据给定的两个值，判断第一个与第二个数的关系，如是否大于、小于、等于第二个数。常见选项如下：
n1 -eq n2	# 相等
n1 -ge n2	# 大于等于
n1 -gt n2 	# 大于
n1 -le n2   # 小于等于
n1 -lt n2 	# 小于
n1 -ne n2 	# 不等于
```

- 字符串比较

```shell
str1 = str2		# str1和str2字符串内容一致
str1 != str2	# str1和str2字符创内容不一致
str1 < str2		# 检查str1是否比str2小
str1 > str2		# 检查str1是否比str2大
-n str1			# 检查str1的长度是否为非0
-z str1			# 检查str1的到长度是否为0

# 注意
# 大于小于号必须转义，否则shell会把他们当作重定向符号，把字符串值当作文件名
# 大于小于顺序和sort命令所采用不同：比较测试中大写字母小于小写字母，sort中相反
```

### 计算表达式

```shell
# 方式一：
$(( 计算表达式 ))	# $((100/5))

# 方式二：
let 计算表达式		# let i=i+7
```

## 常见符号

- 重定向

```shell
# 输出重定向
>		# 表示将符号左侧的内容，以覆盖的方式输入到右侧文件中
>>		# 表示将符号左侧的内容，以追加的方式输入到右侧文件的末尾行中

# 输入重定向
<		# 表示将文件的内容重定向到命令
<<		# 内联输入重定向，无需使用文件进行重定向，只需在命令行中指定用于输入重定向的数据即可
```

- 管道符

```shell
|		# 管道符左侧命令执行的结果，传递给管道右侧的命令使用
```

- 后台展示

```
&		---> 将一个命令从前台转到后台执行
```

- 全部信息符号

```
2>&1
1		--->表示正确输出的信息
2		--->表示错误输出的信息
2>&1	--->代表所有输出的信息
```

- 括号

```shell
((expression))
# 双括号允许在比较过程中使用高级数学表达式
val++	# 后增
val--	# 后减
++val	# 先增
--val	# 先减
!		# 逻辑求反
~		# 位求反
**		# 幂运算
<<		# 左位移
>>		# 右位移
&		# 位布尔和
|		# 位布尔或
&&		# 逻辑和
||		# 逻辑或

[[expression]]
# 双方括号提供了对字符串比较的高级特性
```

## 数学运算

```shell
# expr
expr 1 + 5

# 方括号
var1 = $[1 + 5]
echo var1

# 浮点解决方案
# bash脚本中数学运算符只支持整数运算，若要进行浮点，需要使用内建的bash计算器
# 在命令中使用
bc		# 进入bash计算器
quit	# 退出bash计算器
scale=4	# 设置小数点位数，默认为0
# 在脚本中使用
# 简单计算
variable = $(echo "options; expression" | bc)  # 格式
var1 = $(echo "scale=4, 3.44/5" | bc)
echo the answer is $var1
# 复杂计算
# 方式一：将表达式存放到文件中，使用<将一个文件重定向到bc命令
var1=$(bc < test)
# 方式二：使用<<直接在命令行中重定向数据
variable=$(bc << EOF
options
statements
expressions
EOF  # EOF文本字符串标识了内联重定向数据的起止
)
# 示例
var1=10.46
var2=43.67
var3=33.2
var4=71
var5=$(bc <<EOF
sacle = 4
a1 = ($var1 * $var2)
b1 = ($var3 * $var4)
a1 + b1
EOF
)
```



## 流程控制

### 选择

- 单分支if

```
if [条件]
then
	指令
fi
```

- 双分支if

```
if [条件]
then
	指令1
else
	指令2
fi	
```

- 多分支if

```
if	[条件1]
then
	指令1
elif [条件2]
then
	指令2
else
	指令3
fi
```

- case选择

```shell
case 变量名 in
	值1）
		指令1
			;;
	值2）
		指令2
			;;
	值3）
		指令3
			;;
esac
```

### 循环

- 循环

for循环

```
for 值 in 列表
do
	执行语句
done
```

while循环

```
while 条件
do
	执行语句
done
```

until循环

```
until 条件
do
	执行语句
done
```

- 打断

```shell
break
# 自动终止所在的最内层循环
break n
# 跳出外部循环，n制定了要跳出的循环逻辑

continue
# 提前中止某次循环中的命令
```

## 处理输入

### 多命令

使用多个命令输入，以分号隔开

```shell
date; who 
```

## 数据展示

### 数据展示

```shell
echo
# 显示消息
echo This is a test

# 显示内容中有引号
echo "Let's go!"

# 需要date和标题在一行，无-n时自动换行
echo -n "The time and date are: "
date
```

## 函数

- 简单函数

```
# 定义
函数名(){
    函数体
}

# 调用
函数名
```

- 传参数函数

```
# 传参数
函数名  参数

# 函数体调用参数
函数名(){
    函数体 $n
}
```

- 实践

```
# 简单函数定义和调用
dayin(){
    echo "wo de ming zi shi 111"
}
dayin

# 函数传参和函数体内调用参数
dayin(){
    echo "wo de ming zi shi $1"
}
dayin 111

# 函数调用脚本传参
dayin(){
    echo "wo de ming zi shi $1"
}
dayin $1

# 脚本传多参，函数分别调用
dayin(){
    echo "wo de ming zi shi $1"
    echo "wo de ming zi shi $2"
    echo "wo de ming zi shi $3"
}
dayin 111 df dfs
```

## 实践

- zonghe.sh 脚本执行时候需要添加参数才能执行

  参数和功能详情如下：

  ```
  参数			执行效果
  start		服务启动中...
  stop		服务关闭中...
  restart		服务重启中...
  *			脚本帮助信息...

  ```

- 参数的数量有限制，只能是1个，多余一个会提示脚本的帮助信息

- 帮助信息使用函数来实现

  信息内容：脚本 zonghe.sh 使用方式 zonghe.sh [ start|stop|restart ]

**实践**

```
#！/bin/bash

# 定义本地变量
arg="$1"

# 脚本帮助信息
usage(){
    echo "脚本$0的使用方式是：$0[ start|stop|restart ]"
}

# 函数主框架
if [ $# -eq 1 ]
then
	case "${arg}" in
		start)
			echo "服务启动中..."
			;;
		stop)
			echo "服务关闭中..."
			;;
		restart)
			echo "服务重启中..."
			;;
		*)
			usage
			;;
		esac
else
	usage
fi
```

## 用户输入





# 代码发布

​	获取 -- 打包 -- 传输 -- 关闭 -- 解压 -- 放置 -- 开启 -- 自检 --对外

案例

```python
1. 在主机1上创建一个目录/data/tar-ceshi/，在目录里面创建两个文件，内容分别如下：
	文件名 	内容
	file1.txt file1
	file2.txt file2
2、对目录tar-ceshi进行压缩
3、对目录tar-ceshi进行时间戳备份
4、将压缩包文件传输到远端主机2
5、在主机2上解压 压缩包文件
6、在主机2上修改压缩包文件内容。然后再次压缩
7、在主机1上拉取主机2的压缩包文件
8、使用命令查看压缩包文件的内容
```

命令

```python
# 主机1操作命令
mkdir /data/tar-ceshi -p
cd /data
echo 'file1' > tar-ceshi/file1.txt
echo 'file1' > tar-ceshi/file1.txt

tar zcvf tar-ceshi.tar.gz tar-ceshi

mv tar-ceshi tar-ceshi-$(date +%Y%m%d%H%M%S)

scp tar-ceshi.tar.gz root@192.168.8.15:/tmp

# 主机2操作命令
cd /tmp
tar xf tar-ceshi.tar.gz

echo 'file3' >> tar-ceshi/file1.txt
tar zcvf tar-ceshi-1.tar.gz tar-ceshi

# 主机1操作命令
scp root@192.168.8.15:/tmp/tar-ceshi-1.tar.gz ./
zcat tar-ceshi-1.tar.gz
```

# 环境部署

## 基础环境

- 基础目录环境

```
1. 创建基本目录
# mkdir /data/{server,logs,backup,softs,virtual,scripts,codes} -p
# ls /data/
backup logs scripts server softs virtual codes

2. 查看
admin-1@ubuntu:/data# tree -L 1 /data/
/data/
```

- 主机网络环境

  - 要求

  主机间免密码认证

  - 步骤

  ```
  1. 本机生成秘钥对
  2. 对端机器使用公钥文件认证
  3. 验证
  ```

  - 方案

```python
# 生成秘钥对
ssh-keygen -t rsa
- 参数
-t	指定秘钥类型
rsa 秘钥类型
- 秘钥目录
/root/.ssh/
私钥 id_rsa	钥匙
公钥 id_rsa.pub	锁

# 编辑认证文件
1、创建认证文件，文件内容是线上主机 id_rsa.pub文件的内容						/root/.ssh/authorized_keys
2、ssh认证：
vim /etc/ssh/sshd_config					
AuthorizedKeysFile  %h/.ssh/authorized_keys
3、ssh服务重启
/etc/init.d/ssh  restart

# 验证
scp root@192.168.8.15:/root/python.tar.gz ./
```

 ## 方案分析

```python
# 需求
部署一个环境，支持我们的django项目正常运行

# 需求分析
分析：
2、python环境 ---> 3、python虚拟环境
1、django环境部署
4、django软件安装
5、项目基本操作
6、应用基本操作
7、view和url配置
8、问题：只有本机能访问
9、方案代理---- 10、nginx
11、nginx实现代理
13、pcre软件安装
12、nginx软件安装
14、nginx基本操作
15、nginx代理的配置
16、目录结构
17、查看配置文件
18、找到对应的代理配置项
19、启动django
20、启动nginx
21、整个项目调试

# 部署方案
一、django环境部署
1.1 python虚拟环境
1.2 django环境部署
1.2.1 django软件安装
1.2.2 项目基本操作
1.2.3 应用基本操作
1.2.4 view和url配置
二、nginx代理django
2.1 nginx软件安装
2.1.1 pcre软件安装
2.1.2 nginx软件安装
2.1.3 nginx基本操作
2.2 nginx代理配置
2.2.1 目录结构查看
2.2.2 配置文件查看
2.2.3 编辑代理配置项
三、项目调试
3.1 启动软件
3.1.1 启动django
3.1.2 启动nginx
3.2 整个项目调试
```

## 项目部署

- python虚拟环境

```python
1. 安装
apt-get install python-virtualenv -y
2. 创建
virtualenv -p /usr/bin/python3.5 venv
3. 进入
source venv/bin/activate
4. 退出
deactivate
5. 删除
rm -rf venv
```

- django环境

```python
# django软件安装
1. 进入虚环境
2. 解压
cd /data/soft
tar xf Django-1.10.7.tar.gz
3. 查看
cd Django-1.10.7
cat NSTALL or README
4. 安装
python setup.py install

# django项目操作
1.创建项目
cd /data/server
django-admin startproject itcast

# django应用操作
1.创建应用
cd /data/server/itcast
python manage.py startapp test1
2.注册应用
vim  itcast/settings.py
INSTALL_APP = [
    ...
    'test1',
]

# view和url配置
1. 需求：
访问django的页面请求为：127.0.0.1:8000/hello/,页面返回效果为：itcast v1.0

2. 分析：
views文件定制逻辑流程函数
urls文件定制路由跳转功能

3. view 配置文件生效:
admin-1@ubuntu:/data/soft# cat /data/server/itcast/test1/views.py
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def hello(resquest):
return HttpResponse("itcast V1.0")

4. url文件配置
admin-1@ubuntu:/data/soft# cat /data/server/itcast/itcast/urls.py
...
from test1.views import *
urlpatterns = [
url(r'^admin/', admin.site.urls),
url(r'^hello/$', hello),
]

5. 启动djago
cd /data/server/itcast
python manage.py runserver
```

- nginx环境

```python
# pcre软件安装
1. 解压
cd /data/soft/
tar xf pcre-8.39.tar.gz
2. 查看帮助
cd pcre-8.39
INSTALL 或者 README
3. 配置
./configure
4.编译
make
5.安装
make install

# nginx软件安装
1. 解压
cd /data/soft/
tar xf nginx-1.10.2.tar.gz
2. 配置
cd nginx-1.10.2/
./configure --prefix=/data/server/nginx --without-http_gzip_module
3. 编译
make
4. 安装
make install

# nginx简单操作
1. 检查
/data/server/nginx/sbin/nginx -t
2. 开启
/data/server/nginx/sbin/nginx
3. 关闭
/data/server/nginx/sbin/nginx -s stop
4. 重载
/data/server/nginx/sbin/nginx -s reload
```

- nginx代理django

```shell
# nginx的目录结构
admin-1@ubuntu:/data/server/nginx# tree -L 2 /data/server/nginx/
/data/server/nginx/
├── ...
├── conf 配置文件目录
│ ...
│ ├── nginx.conf 默认的配置文件
│ ...
├── ...
├── html 网页文件
│ ├── 50x.html
│ └── index.html
├── logs 日志目录
│ ├── access.log
│ └── error.log
├── ...
├── sbin 执行文件目录
│ └── nginx
├── ...

# nginx 配置文件介绍
全局配置段
http配置段
server配置段 项目或者应用
location配置段 url配置

# nginx 代理配置
案例需求：
访问地址 192.168.8.14/hello/ 跳转到 127.0.0.1:8000/hello/的django服务来处理hello请求
    
# 编辑配置文件实现代理功能
62: location /hello/ {
63: proxy_pass http://127.0.0.1:8000;
64: }

# 配置文件生效
/data/server/nginx/sbin/nginx -t
/data/server/nginx/sbin/nginx -s reload
```

# 手工发布代码

- 方案分析

```shell
1. 获取代码
sed -i 's#文件原始的内容#替换后的内容#g' 要更改到文件名
2. 打包代码
3. 传输代码
4. 关闭应用
5. 解压代码
6. 放置代码
7. 备份老文件
8. 放置新文件
9. 开启应用
10.检查

注意：
获取代码和打包代码在代码仓库主机上进行操作
其他操作，都在线上服务器进行操作
```

- 方案实施

```shell
# 获取代码
mkdir /data/codes -p
cd /data/codes
sed -i 's#1.0#1.1#' django/views.py
sed -i 's#原内容#替换后内容#g' 文件

# 打包代码
cd /data/codes/
tar zcf django.tar.gz django

# 传输代码
cd /data/codes/
scp root@192.168.8.15:/data/codes/django.tar.gz ./

# 关闭应用
1.关闭nginx应用
/data/server/nginx/sbin/nginx -s stop
2.关闭django应用
根据端口查看进程号，
lsof -Pti :8000
杀死进程号
kill 56502
一条命令搞定它：
kill $(lsof -Pti :8000)

# 解压代码
cd /data/codes
tar xf django.tar.gz

# 放置代码
1. 备份老文件
mv /data/server/itcast/test1/views.py /data/backup/views.py-$(date +%Y%m%d%H%M%S)
2. 放置新文件
cd /data/codes
mv django/views.py /data/server/itcast/test1/

# 开启应用
1. 开启djago
source /data/virtual/venv/bin/activate
cd /data/server/itcast/
python manage.py runserver >> /dev/null 2>&1 &
deactivate
2. 开启nginx
/data/server/nginx/sbin/nginx

# 检查
netstat -tnulp | grep ':80'
```

# 脚本发布代码

## 简单脚本

- 命令罗列

实现代码仓库主机上的操作命令功能即可

```
#!/bin/bash
# 功能：打包代码
# 脚本名：tar_code.sh
# 作者：itcast
# 版本：V 0.1
# 联系方式：www.itcast.cn
cd /data/codes
[ -f django.tar.gz ] && rm -f django.tar.gz
tar zcf django.tar.gz django
```

测试

```
sed -i 's#1.1#1.2#' /data/codes/django/views.py
bash /data/scripts/tar_code.sh
```

查看压缩文件内容

```
zcat django.tar.gz
```

- 固定内容变量化

```shell
#!/bin/bash
# 功能：打包代码
# 脚本名：tar_code.sh
# 作者：itcast
# 版本：V 0.2
# 联系方式：www.itcast.cn
FILE='django.tar.gz'
CODE_DIR='/data/codes'
CODE_PRO='django'
cd "${CODE_DIR}"
[ -f "${FILE}" ] && rm -f "${FILE}"
tar zcf "${FILE}" "${CODE_PRO}"
```

测试

```
sed -i 's#1.2#1.3#' /data/codes/django/views.py
bash /data/scripts/tar_code.sh
```

查看压缩文件

```
zcat django.tar.gz
```

- 功能函数实现

```shell
#!/bin/bash
# 功能：打包代码
# 脚本名：tar_code.sh
# 作者：itcast
# 版本：V 0.3
# 联系方式：www.itcast.cn
FILE='django.tar.gz'
CODE_DIR='/data/codes'
CODE_PRO='django'
code_tar(){
cd "${CODE_DIR}"
[ -f "${FILE}" ] && rm -f "${FILE}"
tar zcf "${FILE}" "${CODE_PRO}"
}
code_tar
```

测试

```
sed -i 's#1.2#1.3#' /data/codes/django/views.py
bash /data/scripts/tar_code.sh
```

查看压缩文件内容

```
zcat /data/codes/django.tar.gz
```

- 远程执行命令

格式：

```
ssh 远程主机登录用户名@远程主机ip地址 "执行命令"
```

效果

```shell
admin-1@ubuntu:/data/server/itcast# ssh root@192.168.8.15 "ifconfig eth0"

eth0 Link encap:Ethernet HWaddr 00:0c:29:f7:ca:d4

inet addr:192.168.8.15 Bcast:192.168.56.255 Mask:255.255.255.0

...
```

远程执行脚本测试

```shell
# 远程更新文件内容
ssh root@192.168.8.15 "sed -i /'s#1.4#1.5#' /data/codes/django/views.py"
# 远程查看脚本
ssh root@192.168.8.15 "ls /data/scripts"
# 远程
```

## 大型脚本

- 脚本框架

```shell
#!/bin/bash
# 功能：打包代码
# 脚本名：deploy.sh
# 作者：itcast
# 版本：V 0.1
# 联系方式：www.itcast.cn
# 获取代码
get_code(){
echo "获取代码"
}
# 打包代码
tar_code(){
echo "打包代码"
}
# 传输代码
scp_code(){
echo "传输代码"
}
# 关闭应用
stop_serv(){
echo "关闭应用"
echo "关闭 nginx 应用"
echo "关闭 django 应用"
}
# 解压代码
untar_code(){
echo "解压代码"
}
# 放置代码
fangzhi_code(){
echo "放置代码"
echo "备份老文件"
echo "放置新文件"
}
# 开启应用
start_serv(){
echo "开启应用"
echo "开启 django 应用"
echo "开启 nginx 应用"
}
# 检查
check(){
echo "检查项目"
}
# 部署函数
deploy_pro(){
get_code
tar_code
scp_code
stop_serv
untar_code
fangzhi_code
start_serv
check
}
# 主函数
main(){
deploy_pro
}
# 执行主函数
main
```

- 命令填充

```shell
#!/bin/bash
# 功能：打包代码
# 脚本名：deploy.sh
# 作者：itcast
# 版本：V 0.2
# 联系方式：www.itcast.cn
# 获取代码
get_code(){
echo "获取代码"
}
# 打包代码
tar_code(){
echo "打包代码"
ssh root@192.168.8.15 "/bin/bash /data/scripts/tar_code.sh"
}
# 传输代码
scp_code(){
echo "传输代码"
cd /data/codes
[ -f django.tar.gz ] && rm -f django.tar.gz
scp root@192.168.8.15:/data/codes/django.tar.gz ./
}
# 关闭应用
stop_serv(){
echo "关闭应用"
echo "关闭 nginx 应用"
/data/server/nginx/sbin/nginx -s stop
echo "关闭 django 应用"
kill $(lsof -Pti :8000)
}
# 解压代码
untar_code(){
	echo "解压代码"
cd /data/codes
tar xf django.tar.gz
}
# 放置代码
fangzhi_code(){
echo "放置代码"
echo "备份老文件"
mv /data/server/itcast/test1/views.py /data/backup/views.py-$(date +%Y%m%d%H%M%S)
echo "放置新文件"
mv /data/codes/django/views.py /data/server/itcast/test1/
}
# 开启应用
start_serv(){
echo "开启应用"
echo "开启 django 应用"
source /data/virtual/venv/bin/activate
cd /data/server/itcast/
python manage.py runserver >> /dev/null 2>&1 &
deactivate
echo "开启 nginx 应用"
/data/server/nginx/sbin/nginx
}
# 检查
check(){
echo "检查项目"
netstat -tnulp | grep ':80'
}
...	
```

- 增加日志功能

```shell
#!/bin/bash
...
LOG_FILE='/data/logs/deploy.log'
# 增加日志功能
write_log(){
DATE=$(date +%F)
TIME=$(date +%T)
buzhou="$1"
echo "${DATE} ${TIME} $0 : ${buzhou}" >> "${LOG_FILE}"
}
# 获取代码
get_code(){
...
write_log "获取代码"
}
# 打包代码
tar_code(){
...
write_log "打包代码"
}
# 传输代码
scp_code(){
...
write_log "传输代码"
}
# 关闭应用
stop_serv(){
...
write_log "关闭应用"
...
write_log "关闭 nginx 应用"
...
write_log "关闭 django 应用"
}
# 解压代码
untar_code(){
...
write_log "解压代码"
}
# 放置代码
fangzhi_code(){
...
write_log "放置代码"
...
write_log "备份老文件"
...
write_log "放置新文件"
}
# 开启应用
start_serv(){
...
write_log "开启应用"
...
write_log "开启 django 应用"
...
write_log "开启 nginx 应用"
}
# 检查
check(){
...
write_log "检查项目"
}
...
```

- 增加锁文件

```shell
#!/bin/bash
...
PID_FILE='/tmp/deploy.pid'
...
# 增加锁文件功能
add_lock(){
echo "增加锁文件"
touch "${PID_FILE}"
write_log "增加锁文件"
}
# 删除锁文件功能
del_lock(){
echo "删除锁文件"
rm -f "${PID_FILE}"
write_log "删除锁文件"
}
# 部署函数
deploy_pro(){
add_lock
...
del_lock
}
# 脚本报错信息
err_msg(){
echo "脚本 $0 正在运行，请稍候..."
}
# 主函数
main(){
if [ -f "${PID_FILE}" ]
then
err_msg
else
deploy_pro
fi
}
# 执行主函数
main
```

## 脚本知识点填充

需求

```shell
如果我给脚本输入的参数是deploy，那么脚本才执行，否则的话，提示该脚本的使用帮助信息，然后退出
提示信息：脚本 deploy.sh 的使用方式： deploy.sh [ deploy ]
```

分析

```shell
1、脚本传参，就需要在脚本内部进行调用参数
2、脚本的帮助信息
3、脚本内容就需要对传参的内容进行判断
```

方案

```shell
1、脚本的传参
脚本执行：bash deploy.sh deploy
位置参数的调用： $1
2、脚本的帮助信息
定义一个usage函数，然后调用。
提示信息格式：
脚本 deploy.sh 的使用方式： deploy.sh [ deploy ]
3、内容判断
main函数体调用函数传参: $1
在main函数中，结合case语句，对传入的参数进行匹配
如果传入参数内容是"deploy"，那么就执行代码部署流程
如果传入参数内容不是"deploy"，那么输出脚本的帮助信息
if语句和case语句的结合
if语句在外，case语句在内
case语句在外，if语句在内
```

脚本

```shell
#!/bin/bash
...
# 脚本帮助信息
usage(){
echo "脚本 $0 的使用方式: $0 [deploy]"
exit
}
# 主函数
main(){
case "$1" in
"deploy")
if [ -f "${PID_FILE}" ]
then
err_msg
else
deploy_pro
fi
;;
*)
usage
;;
esac
}
# 执行主函数
main $1
```

## 输入参数安全优化

需求

```
对脚本传入的参数的数量进行判断，如果参数数量不对，提示脚本的使用方式，然后退出
```

分析

```
1、脚本参数数量判断
2、条件判断
数量对，那么执行主函数
数量不对，那么调用脚本帮助信息
```

脚本

```shell
#!/bin/bash
...
# 执行主函数
if [ $# -eq 1 ]
then
main $1
else
usage
fi
```

# 脚本调试

```shell
-n 检查脚本中的语法错误
-v 先显示脚本所有内容， 然后执行脚本，结果输出，如果执行遇到错误，将错误输出。
-x 将执行的每一条命令和执行结果都打印出来
```

# 脚本案例

有密码交互

```shell
# /bin/bash
pwd = '1234'
echo pwd | sudo list
```

