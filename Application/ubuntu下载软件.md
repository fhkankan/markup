# axel

linux下轻量级下载加速工具,axel是一个多线程分段下载工具，可以从ftp或http服务器进行下载。

## 安装

```
sudo apt-get install axel
```

## 格式  

```
axel [OPTIONS] url1 url2
```

##参数

 至少要制定一个参数，如果是FTP下载的话，可以指定通配符进行下载， 程序会自行解析完整的文件名。可以指定多个URL进行下载，但是程序不会区分这些URL是否为同一个文件，换而言之，同一个URL指定多次，就会进行多次的下载。
**详细参数**  

```
--max-speed=x, -s x

指定最大下载速度。

 --num-connections=x, -n x

指定链接的数量。

 --output=x, -o x

指定下载的文件在本地保存的名字。如果指定的参数是一个文件夹，则文件会下载到指定的文件夹下。

--search[=x], -S[x]

Axel将会使用文件搜索引擎来查找文件的镜像。缺省时用的是filesearching.com。可以指定使用多少个不同的镜像来下载文件。
检测镜像将会花费一定的时间，因为程序会测试服务器的速度，以及文件存在与否。

--no-proxy, -N

不使用代理服务器来下载文件。当然此选项对于透明代理来说无意义。

--verbose

如果想得到更多的状态信息，可以使用这个参数。

--quiet, -q

不向标准输出平台(stdout)输入信息。

--alternate, -a

指定这个参数后将显示一个交替变化的进度条。它显示不同的线程的进度和状态，以及当前的速度和估计的剩余下载时间。

--header=x, -H x

添加HTTP头域，格式为“Header: Value”。

--user-agent=x, -U x

有些web服务器会根据不同的User-Agent返回不同的内容。这个参数就可以用来指定User-Agent头域。缺省时此头域值包括“Axel”，它的版本号以及平台信息。

--help, -h

返回参数的简要介绍信息。

--version, -V

显示版本信息。

注意：

除非系统支持getopt_long，否则最好不要使用长选项（即以两个短横杠开头的选项）。

返回值：

下载成功返回0，出错返回1，被中止返回2。其它为出现bug。

```
**配置文件**
```
全局配置文件为：/etc/axelrc或/usr/local/etc/axelrc。

个人配置文件为：~/.axelrc。
```

**举例**
```

axel http://www.baidu.com

需要指定http://或ftp://。
```
# utorrent

## 安装

```
1.下载
http://www.utorrent.com/intl/zh/downloads/linux
2.解压至安装路径
3.命令行cd到utserver所在目录下
4.命令行运行
./utserver
5.浏览器打开
浏览器输入http://localhost:8080/gui/，账号admin密码无  
```

