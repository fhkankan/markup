# FastDFS

## 简介

```
FastDFS 是用 c 语言编写的一款开源的分布式文件系统。FastDFS 为互联网量身定制， 充分考虑了冗余备份、负载均衡、线性扩容等机制，并注重高可用、高性能等指标，使用 FastDFS 很容易搭建一套高性能的文件服务器集群提供文件上传、下载等服务。 

FastDFS 架构包括 Tracker server 和 Storage server。客户端请求 Tracker server 进行文 件上传、下载，通过 Tracker server 调度最终由 Storage server 完成文件上传和下载。 

Tracker server 作用是负载均衡和调度，通过 Tracker server 在文件上传时可以根据一些策略找到 Storage server 提供文件上传服务。可以将 tracker 称为追踪服务器或调度服务 器。 
Storage server 作用是文件存储，客户端上传的文件最终存储在 Storage 服务器上， Storageserver 没有实现自己的文件系统而是利用操作系统 的文件系统来管理文件。可以将 storage 称为存储服务器。

Tracker:管理集群，tracker 也可以实现集群。每个 tracker 节点地位平等。收集 Storage 集群的状态。 
Storage:实际保存文件 Storage 分为多个组，每个组之间保存的文件是不同的。每 个组内部可以有多个成员，组成员内部保存的内容是一样的，组成员的地位是一致的，没有 主从的概念。
```

## 上传与下载

上传

```
1. storage server定时向tracker server上传状态信息
2. client向tracker server上传连接请求
3. tracker server查询可用storage
4. tracker server向client返回信息（storage的ip和端口）
5. client向storage server上传文件
6. storage server生成file_id
7. storage server将上传内容写入磁盘
8. storage server向client返回file_id
9. 在client中存储文件信息
```

下载

```
1. strrage server定时向tracker server上传状态信息
2. client向tracker server下载连接请求
3. tracker server查询可用storage（检验同步状态）
4. tracker server向client返回信息（storage的ip和端口） 
5. client向storage server请求file_id
6. storage server根据file_id查找文件
7. storage server向client返回file_content
```

## 安装与配置

安装fastdfs依赖包

```
1. 解压缩libfastcommon-master.zip
2. 进入到libfastcommon-master的目录中
3. 执行 ./make.sh
4. 执行 sudo ./make.sh install
```

安装fastdfs

```
1. 解压缩fastdfs-master.zip
2. 进入到 fastdfs-master目录中
3. 执行 ./make.sh
4. 执行 sudo ./make.sh install
```

安装nginx

安装fastdfs-nginx-module

```
1. 解压缩 nginx-1.8.1.tar.gz
2. 解压缩 fastdfs-nginx-module-master.zip
3. 进入nginx-1.8.1目录中
4. 执行
sudo ./configure --prefix=/usr/local/nginx/ --add-module=fastdfs-nginx-module-master解压后的目录的绝对路径/src
sudo ./make
sudo ./make install


7. sudo cp 解压缩的fastdfs-master目录中的http.conf  /etc/fdfs/http.conf
8. sudo cp 解压缩的fastdfs-master目录中的mime.types /etc/fdfs/mime.types
```

配置跟踪服务器tracker

```
1. sudo cp /etc/fdfs/tracker.conf.sample /etc/fdfs/tracker.conf
2. 在/home/python/目录中创建目录 fastdfs/tracker      
mkdir –p /home/python/fastdfs/tracker
3. 编辑/etc/fdfs/tracker.conf配置文件  
sudo vim /etc/fdfs/tracker.conf
修改 base_path=/home/python/fastdfs/tracker
```

配置存储服务器storage

```
1. sudo cp /etc/fdfs/storage.conf.sample /etc/fdfs/storage.conf
2. 在/home/python/fastdfs/ 目录中创建目录 storage
mkdir –p /home/python/fastdfs/storage
3. 编辑/etc/fdfs/storage.conf配置文件  
sudo vim /etc/fdfs/storage.conf
修改内容：
base_path=/home/python/fastdfs/storage
store_path0=/home/python/fastdfs/storage
tracker_server=自己ubuntu虚拟机的ip地址:22122
```

配置mod_fastdfs

```
1. sudo cp fastdfs-nginx-module-master解压后的目录中src下的mod_fastdfs.conf  /etc/fdfs/mod_fastdfs.conf
2. 编辑
sudo vim /etc/fdfs/mod_fastdfs.conf
修改内容：
connect_timeout=10
tracker_server=自己ubuntu虚拟机的ip地址:22122
url_have_group_name=true
store_path0=/home/python/fastdfs/storage
```
配置client

```
1. sudo cp /etc/fdfs/client.conf.sample /etc/fdfs/client.conf
2. 编辑/etc/fdfs/client.conf配置文件  
sudo vim /etc/fdfs/client.conf
修改内容：
base_path=/home/python/fastdfs/tracker
tracker_server=自己ubuntu虚拟机的ip地址:22122
```
配置nginx

```
sudo vim /usr/local/nginx/conf/nginx.conf
在http部分中添加配置信息如下：
server {
            listen       8888;
            server_name  localhost;
            location ~/group[0-9]/ {
                ngx_fastdfs_module;
            }
            error_page   500 502 503 504  /50x.html;
            location = /50x.html {
            root   html;
            }
        }
```

启动tracker 、storage和nginx

```
sudo service fdfs_trackerd start
sudo service fdfs_storaged start
sudo /usr/local/nginx/sbin/nginx
或sudo /usr/local/nginx/sbin/nginx -s reload
```

测试是否安装成功

```
命令行输入测试文件上传：
fdfs_upload_file /etc/fdfs/client.conf 要上传的图片文件 
如果返回类似group1/M00/00/00/rBIK6VcaP0aARXXvAAHrUgHEviQ394.jpg的文件id则说明文件上传成功


浏览器访问：127.0.0.1:8888/文件id
```

# python对接fastdfs

文档 https://github.com/jefforeilly/fdfs_client-py

1. workon django_py3
2. 进入fdfs_client-py-master.zip所在目录
3. pip install fdfs_client-py-master.zip
4. 在python中运行

```
>>> from fdfs_client.client import Fdfs_client
>>> client = Fdfs_client('/etc/fdfs/client.conf')
>>> ret = client.upload_by_filename('test')
>>> ret
{'Group name':'group1','Status':'Upload successed.', 'Remote file_id':'group1/M00/00/00/	wKjzh0_xaR63RExnAAAaDqbNk5E1398.py','Uploaded size':'6.0KB','Local file name':'test'
	, 'Storage IP':'192.168.243.133'}

```

# Django二次开发对接fastdfs

<http://python.usyiyi.cn/translate/django_182/ref/files/storage.html>

<http://python.usyiyi.cn/translate/django_182/howto/custom-file-storage.html>

步骤

```
1.	定义文件存储类
2.	在类中实现_save() 和_open()
3.	在settings文件进行配置
```

