# Nginx

## 安装

- mac

```
brew install nginx
```

开机启动

```
sudo cp /usr/local/opt/nginx/*.plist  /Library/LaunchDaemons
sudo launchctl load -w /Library/LaunchDaemons/homebrew.mxcl.nginx.plist
```

重要文件位置

```
/usr/local/cellar/nginx/版本  # 核心安装目录
/usr/local/cellar/nginx/版本/bin  # 启动文件
/usr/local/cellar/nginx/版本/html  # 欢迎页面在html下        
/usr/local/etc/nginx   # 核心配置文件路径
/usr/local/var/www  # 服务器默认路径
```

- ubuntu

```
sudo apt-get install nginx
```

重要文件位置

```
/etc/nginx	# 配置文件
/usr/sbin/nginx	# 程序文件
/var/log/nginx # 日志文件
/etc/nginx/sites-available	# 虚拟主机位置
/var/www/nginx-default		# 默认虚拟主机目录
/etc/init.d/nginx	# 启动脚本
```

- docker

```
docker run \
  --name myNginx \
  -d -p 90:80 \
  -v /usr/docker/myNginx/html:/usr/share/nginx/html \
  -v /etc/docker/myNginx/nginx.conf:/etc/nginx/nginx.conf \
  -v /etc/docker/myNginx/conf.d:/etc/nginx/conf.d \
  nginx
```

注意

```
/usr/docker/myNginx/html			# 挂载项目文件

/etc/docker/myNginx/nginx.conf		# 挂载的主配置文件"nginx.conf"，注意"nginx.conf"文件内有一行"include /etc/nginx/conf.d/*.conf;"，这个include指向了子配置文件的路径，此处注意include后所跟的路径一定不要出错。

/etc/docker/myNginx/conf.d			# 子配置文件的路径也挂载了出来，注意要与（2）中include指向路径一致

nginx.conf是挂载了一个文件（docker是不推荐这样用的），conf.d挂载的是一个目录
```

- 源码

```
# 下载后解压缩
tar zxvf nginx-1.6.3.tar.gz
# 安装
./configure
make
sudo make install
# 启动
# 默认安装到/usr/local/nginx/目录
cd /usr/local/nginx/
sudo sbin/nginx
```

## 使用

Mac

```shell
# 启动
brew services start nginx
# 停止
brew services stop nginx
# 重新加载配置
brew services reload nginx
# 重启
brew services restart nginx
```

ubuntu

```
nginx {start|stop|restart|reload|force-reload|status|configtest|rotate|upgrade}
```

其他

```shell
sudo nginx -s reload	# 重新加载配置文件
sudo nginx -s restart	# 重启
sudo nginx -s stop    # 快速停止nginx
sudo nginx -s quit 		# 不再接受新的请求，等正在处理的请求出完成后在进行停止（优雅的关闭）
```

测试

```shell
nginx -t -c xxx		# 测试nginx配置是否ok
```

浏览器访问

```
localhost:8080
```

## 配置

### 默认配置

```
#user  nobody;
worker_processes  1;
#error_log  logs/error.log;
#error_log  logs/error.log  notice;
#error_log  logs/error.log  info;
#pid        logs/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       mime.types;
    default_type  application/octet-stream;
    #log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
    #                  '$status $body_bytes_sent "$http_referer" '
    #                  '"$http_user_agent" "$http_x_forwarded_for"';
    #access_log  logs/access.log  main;
    sendfile        on;
    #tcp_nopush     on;
    #keepalive_timeout  0;
    keepalive_timeout  65;
    #gzip  on;

    server {
        listen       8080;
        server_name  localhost;
        #charset koi8-r;
        #access_log  logs/host.access.log  main;
        location / {
            root   html;
            index  index.html index.htm;
        }
        #error_page  404              /404.html;
        # redirect server error pages to the static page /50x.html
        #
        error_page   500 502 503 504  /50x.html;
        location = /50x.html {
            root   html;
        }
        # proxy the PHP scripts to Apache listening on 127.0.0.1:80
        #
        #location ~ \.php$ {
        #    proxy_pass   http://127.0.0.1;
        #}
        # pass the PHP scripts to FastCGI server listening on 127.0.0.1:9000
        #
        #location ~ \.php$ {
        #    root           html;
        #    fastcgi_pass   127.0.0.1:9000;
        #    fastcgi_index  index.php;
        #    fastcgi_param  SCRIPT_FILENAME  /scripts$fastcgi_script_name;
        #    include        fastcgi_params;
        #}
        # deny access to .htaccess files, if Apache's document root
        # concurs with nginx's one
        #
        #location ~ /\.ht {
        #    deny  all;
        #}
    }

    # another virtual host using mix of IP-, name-, and port-based configuration
    #
    #server {
    #    listen       8000;
    #    listen       somename:8080;
    #    server_name  somename  alias  another.alias;

    #    location / {
    #        root   html;
    #        index  index.html index.htm;
    #    }
    #}


    # HTTPS server
    #
    #server {
    #    listen       443 ssl;
    #    server_name  localhost;
    #    ssl_certificate      cert.pem;
    #    ssl_certificate_key  cert.key;
    #    ssl_session_cache    shared:SSL:1m;
    #    ssl_session_timeout  5m;
    #    ssl_ciphers  HIGH:!aNULL:!MD5;
    #    ssl_prefer_server_ciphers  on;
    #    location / {
    #        root   html;
    #        index  index.html index.htm;
    #    }
    #}
    include servers/*;
}
```

### 修改配置

修改服务器网站根目录

```
server {
        # listen       8080;  # 端口号
        listen       0.0.0.0:8080;
        server_name  localhost;  # 可为ip地址
        location / {
            root   html;  # 根目录位置
            index  index.html index.htm;
            autoindex on;  # 显示目录
        }
}
```

上传大文件

```shell
http{
	server{
		location /trail {
				# 防止Entity Too Large
				client_max_body_size 501m; # 客户端上传文件大小500M，默认1m，若超过返回413错误
				client_body_buffer_size 1m; # 缓存大小
				# 防止504 gateway time-out
			 	proxy_connect_timeout  1800s; #nginx跟后端服务器连接超时时间(代理连接超时)，默认60s
    		proxy_send_timeout  1800s;#后端服务器数据回传时间(代理发送超时)，默认60s
    		proxy_read_timeout  1800s;#连接成功后，后端服务器响应时间(代理接收超时)，默认60s
    		fastcgi_connect_timeout 1800s;#指定nginx与后端fastcgi server连接超时时间
    		fastcgi_send_timeout 1800s;#指定nginx向后端传送请求超时时间（指已完成两次握手后向fastcgi传送请求超时时间）
    		fastcgi_read_timeout 1800s;#指定nginx向后端传送响应超时时间（指已完成两次握手后向fastcgi传送响应超时时间）
		}
	}
}
```

静态文件支持跨域配置

```shell
server {
        listen       80;
        server_name  localhost;
        location / {
            root   html;
            index  index.html index.htm;
            try_files $uri $uri/ /index.html
        }
        location /filedata{
        		add_header 'Access-Control-Allow-Origin' '*';
        		alias /NginxData;
        		allow all;
        		autoindex on;
        }
}
```

反向代理

```shell
upstream burn_backend {
    server xx.xx.xx.xx:12000;
    server xx.xx.xx.xx:12001;
    keepalive 100;
}

server {
    charset utf-8;
    client_max_body_size 128M;
    proxy_headers_hash_bucket_size 6400;
    proxy_headers_hash_max_size 51200;

    listen 80;
    server_name  api.xx.cn;

    root /opt/www/api.xx.cn;
    access_log /opt/log/api.xx.cn.log main;
		
		location /micro/burn {
        proxy_pass http://burn_backend/prod;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Server $server_name;
        proxy_set_header X-Forwarded-For $http_x_forwarded_for;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

https

```
server {
    charset utf-8;
    client_max_body_size 128M;
    proxy_headers_hash_bucket_size 6400;
    proxy_headers_hash_max_size 51200;


    listen 443;
    server_name  stg.xx.cn;

    root /home/apollo/coupon_backendV2/public;
    index frontend.php index.html index.htm;

    ssl on;
    ssl_certificate  stg_cert/1_stg.miniapp.mcdonalds.com.cn_bundle.crt;
    ssl_certificate_key  stg_cert/2_stg.miniapp.mcdonalds.com.cn.key;
    ssl_session_timeout 5m;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE:ECDH:AES:HIGH:!NULL:!aNULL:!MD5:!ADH:!RC4;
    ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
    ssl_prefer_server_ciphers on;

    access_log /opt/log/stg-443.log main;
    #access_log off;
    #resolver 8.8.8.8;

    include ./conf.d/flyer-trial-debug.part;


    # 网站后台
    location /backend.php {
        #root /home/apollo/coupon_backend/public;
        rewrite ^/backend.php(.*)$ /backend.php;
    }
    
    location ~ /trial/imcd/openapi/token/member_info {
        proxy_pass http://xx.xx.xx.xx:28000;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Server $server_name;
        proxy_set_header X-Forwarded-For $http_x_forwarded_for;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 配置指南

### 全局模块

```shell
#user  nobody;
worker_processes  1;
#error_log  logs/error.log;
#error_log  logs/error.log  notice;
#error_log  logs/error.log  info;
#pid        logs/nginx.pid;
```

配置说明

| 配置指令         | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| user             | 配置worker进程的用户和组，若忽略group，则group的名字等于该参数指定用户的用户组 |
| worker_processes | 指定woeker进程启动的数量。用于处理客户的所有连接，经验法则是设置该参数的值与cpu绑定的负载处理器核心数数量相同，并用1.5~2之间的数乘以这个数作为I/O密集型负载，不确定时可设置为auto |
| error_log        | 所有错误写入的日志，第二个参数指定了错误的级别(debug,info,notice,warn,error,crit,alert,emerg)。debug级别的错误需在编译时配置了--with-debug |
| pid              | 设置记录主进程ID的文件，会覆盖编译时的默认配置               |

### events

```shell
events {
    worker_connections  1024;
}
```

`events`块涉及的指令主要影响Nginx服务器与用户的网络连接，常用的设置包括是否开启对多work process下的网络连接进行序列化，是否允许同时接受多个网络连接，选取哪种事件驱动模型来处理连接请求，每个work process可以同时支持的最大连接数等。

并发总数为：max_clients = worker_processes * worker_connections，设置了反向代理时，max_clients=worker_processes * worker_connections / 4。worker_connections 值的设置跟物理内存大小有关，因为并发受IO约束，max_clients的值须小于系统可以打开的最大文件数

配置说明

| 配置指令             | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| `use`                | 用于指示使用什么样的连接方式。会覆盖编译时的默认配置。`kqueue,rtsig,epoll,/dev/poll,select,poll`，epoll模型是Linux 2.6以上版本内核中的高性能网络I/O模型，如果跑在FreeBSD上面，就用kqueue模型。
| `worker_connections` | 配置一个工作进程能够接受并发连接的最大数。这个连接包括客户连接和向上游服务器的连接，但并不限于此。对于反向代理服务器尤为重要，为达到这个并发性连接数量，需要在操作系统层面进行一些额外调整。 |

### Http

添加位置不同，限制不同

```shell
# 添加在http{ }中，控制所有请求报文大小;
# 添加在server{ }中，控制该server的所有请求报文大小;
# 添加在location{ }中，控制满足该路由规则的请求报文大小;
```

#### 客户端指令

| http客户端指令               | 说明                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| chunked_transfer_encodeing   | 在发送给客户端的响应中，该指令允许禁用http/1.1标准的块传输编码 |
| client_body_buffer_size      | 为了阻止临时文件写到磁盘，可以通过该指令为客户端请求体设置缓存大小，默认大小为两个内存页面 |
| client_body_in_file_only     | 用于调试或进一步处理客户端请求体。该指令设置为“on”能够将客户端请求体强制写入到磁盘文件 |
| client_body_in_single_buffer | 为了减少复制的操作，使用该指令强制Nginx将整个客户端请求保存在单个缓存中 |
| client_body_temp_path        | 定义一个命令路径用于保存客户端请求体                         |
| `client_body_timeout`        | 指定客户体成功读取的两个操作之间的时间间隔，超过返回413错误  |
| client_header_buffer_size    | 为客户端请求头指定一个缓存大小，当请求头大于1kb时会用到这个设置 |
| `client_header_timeout`      | 该超时是读取整个客户端头的时间长度，超过返回408错误          |
| `client_max_body_size`       | 定义允许最大的客户端请求头，默认1m，若超过返回413错误        |
| keepalive_disable            | 对某些类型的客户禁用keep-alive请求功能                       |
| keepalive_requests           | 定义在一个keep-alive关闭之前可以接受多少个请求               |
| keepalive_timeout            | 指定keep-alive连接持续多久，第二个参数也可以设置，用于在响应头中设置"keepalive"头 |
| large_client_header_buffers  | 定义最大数量和最大客户端请求头的大小                         |
| msize_padding                | 为了填充响应的大小至512字节，对于MSIE客户端，大于400的状态码会被添加注释以便满足512字节，通过启用该命令可以阻止这种行为 |
| msize_refresh                | 对于MSIE客户端，该指令可启用发送一个refresh头，而不是redirect |

#### 文件I/O指令

用于控制Nginx如何投递静态文件以及如何管理文件描述符

| 文件I/O指令              | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| aio                      | 启用异步文件I/O。对于现代版本的FreeBSD和所有Linux发行版都有效。在FreeBSD下，aio可能被用于sendfile预加载数据。在Linux下，需要directio指令，自动禁用sendfile. |
| directio                 | 用于启用操作系统特定的标志或功能提供大于给定参数的文件。在Linux中使用aio时需要 |
| directio_alignment       | 设置directio的算法，默认值为512，通常足够，但在Linux的XFS下推荐增为4kB |
| open_file_cache          | 配置一个缓存用于存储打开的文件描述符、目录查询和文件查询错误 |
| open_file_cache_error    | 按照open_file_cache,启用文件查询错误缓存                     |
| open_file_cache_min_uses | open_file_cache缓存的文件描述符保留在缓存中，使用该指令配置最少使用文件描述符的次数 |
| open_file_cache_valid    | 该指令指定对open_file_cache缓存有效性检查的时间间隔          |
| postpone_output          | 指定Nginx发送给客户端最小的数值，若可能，没有数据发送，直到达到此值 |
| read_ahead               | 若可能， 内核将预读文件到设定的参数大小。                    |
| sendfile                 | 使用sendfile(2)直接复制数据从一个到另一个文件描述符          |
| sendfile_max_chunk       | 设置在一个sendfile(2)中复制最大数据的大小，阻止worker“贪婪”  |

#### Hash指令

控制Nginx分配给某些变量多大的静态内存。在启动和重新配置时，会计算需要的最小值。在Nginx发出警告时，只需要调整一个`*_hash_max_size`指令的参数值就可以达到效果。`*_hash_bucket_size`变量被设置了默认值，以便满足多处理器缓存行降低检索需要的检索查找。

| hash指令                      | 说明                                  |
| ----------------------------- | ------------------------------------- |
| server_names_hash_bucket_size | 指定用于保存server_name散列表大小的桶 |
| server_names_hash_max_size    | 指定server_name散列表的最大大小       |
| types_hash_bucket_size        | 指定用于存储散列表的桶的大小          |
| types_hash_max_size           | 指定散列类型比饿哦的最大大小          |
| variables_hash_bucket_size    | 指定用于存储保留变量桶大小            |
| variables_hash_max_size       | 指定存储保留变量最大散列值得大小      |

#### socket指令

描述了如何设置创建TCP套接字的变量选项

| socket指令               | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| lingering_close          | 指定如何保持客户端的连接，以便于更多数据的传输               |
| lingering_time           | 在使用lingering_close指令的连接中，该指令指定客户端连接为了处理更多的数据需要保持打开连接的时间 |
| lingering_timeout        | 结合lingering_close，该指令显示Nginx在关闭客户端连接之前，为获得更多的连接会等待多久 |
| reset_timeout_connection | 使用后，超时的连接将会立即关闭，释放相关的内存。默认的状态是处于FIN_WAIT1，状态会一直保持连接 |
| send_lowat               | 若非零，Nginx会子啊客户端套接字尝试减少发送操作              |
| send_tiemout             | 在两次成功的客户端接收响应的写操作之间设置一个超时时间       |
| tcp_nodelay              | 启用或禁用tcp_nodelay，用于keep-alive连接                    |
| tcp_push                 | 仅依赖于sendfile的使用，使Nginx在一个数据包中尝试发送响应头以及在数据包中发送一个完成的文件 |

#### server指令

虚拟主机，就是将一台物理服务器虚拟为多个服务器来使用，从而实现在一台服务器上配置多个站点，即可以在一台物理主机上配置多个域名。Nginx 中，一个 server 标签就是一台虚拟主机，配置多个 server 标签就虚拟出了多台主机。Nginx 虚拟主机的实现方式有两种：域名虚拟方式与端口虚拟方式。域名虚拟方式是指不同的虚拟机使用不同的域名，通过不同的域名虚拟出不同的主机；端口虚拟方式是指不同的虚拟机使用相同的域名不同的端口号，通过不同的端口号虚拟出不同的主机。基于端口的虚拟方式不常用。

指令server开始了一个新的上下文，在Nginx中，默认服务器是指特定配置文件中监听同一IP地址、同一端口作为另一个服务器中的第一个服务器。默认服务器可以通过为listen指令配置default_server参数来实现

默认服务器定义一组通用指令，监听在相同IP地址和端口的随后的服务器将会重复利用这些指令

```
server {
    listen 127.0.0.1:80;
    server_name default.example.com;
    server_name_in_redirect on;
}
server {
	liaten 127.0.0.1:80;
	server_name www.example.com;
}
```

服务器指令

| HTTP Server指令         | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| Port_in_redirect        | 确定Nginx是否对端口指定重定向                                |
| server                  | 该指令创建一个新的配置区段，定义一个虚拟主机。listen指令指定IP地址和端口号，server_name指令列举用于匹配的Host头值 |
| server_name             | 配置用于响应请求的虚拟主机名称                               |
| Server_name_in_redirect | 在该context中，对任何由Nginx发布的重定向，该指令都使用server_name指令中的第一个值来激活 |
| Server_tokens           | 在错误信息中，该指令禁止发送Nginx的版本号和server响应头(默认值为on) |

#### Access_log

配置文件的灭一个级别都可以有访问日志，在每一个级别上可以指定多个访问日志，每个日志用一个不同的log_format指令。log_format指令允许你明确指定记录要记在的内容，该指令需要在http部分内定义。

```
http{
    log_format vhost '$host $remote_addr - $remote_user [$time_local]'
    '"$request" $status $body_bytes_sent'
    '"$http_referer" "$http_user_agent"';
    
    log_format downloads '$time_iso8601 $host $remote_addr'
    '"$request" $status $body_bytes_sent $request_tiem';
    
    open_log_file_cache max=1000 inactive=60s;
    access_log logs/access.log;
   
    server{
        server_name ~^(www\.)?(.+)$;
        access_log logs/combined.log vhost;
        access_log logs/$2/accesslog;
        location /downloads{
            access_log logs/downloads.log downloads;
        }
    }
}
```

日志指令

| HTTP日志指令        | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| Access_log          | 描述在哪里、怎么样写入访问日志。第一个参数是日志文件被存储位置的路径。在构建的路中可使用变量，特殊值off可以禁止记录访问日志。第二个可选参数用于指定log_format指令设定的日志格式。若未配置第二个参数，则试用预定义的combined格式。若写缓存用于记录日志，第三个可选参数则知名了写缓存的大小。若使用写缓存，则这个大小不能超过写文件系统的原子磁盘大小。若第三个参数是gzip，那么缓冲日志将会被动态压缩，在构建Nginx二进制时需要提供zlib库。最后一个采纳数时flush，表明在将缓冲日志数据冲洗到磁盘之前，它们能够在内存中停留的最大时间 |
| log_format          | 指定出现在日志文件的字段和采用的格式。日志中指定日志变量参考下表 |
| Log_not_found       | 禁止在错误日志中报告404错误(默认on)                          |
| log_subrequest      | 在访问日志中启用记录子请求(默认off)                          |
| Open_log_file_cache | 存储access_logs在路径中使用到的打开的变量文件描述符的缓存。用到如下参数：max:指定文件描述符在缓存中的最大数量<br>inactive:在文件描述符被关闭前，使用该参数表明Nginx将会等待一个时间间隔用于写入该日志<br>min_uses:使用该参数表明文件描述符被使用的次数，在inactive时间内达到指定的次数，该文件描述符将会被保持打开<br>valid:使用该参数表明Nginx将经常检查次文件描述符是否仍有同名文件匹配<br>off:该参数禁止缓存 |

当指定了gzip后，则不可选用log_format参数

```
# 日志条目使用gzip压缩为4级，缓存默认64kb，至少每分钟都将缓存刷新到磁盘
access_log /var/log/nginx/access.log.gz combined gzip=4 flush=1m;
```

日志格式变量名称

| 日子格式变量名称          | 值                                                           |
| ------------------------- | ------------------------------------------------------------ |
| `$body_bytes_sent`        | 指定发送到客户端的字节数，不包括响应头                       |
| `$bytes_sent`             | 指定发送到客户端的字节数                                     |
| `$connection`             | 指定一个串号，用于标识一个唯一的连接                         |
| `$connection_requests`    | 指定通过一个特定连接的请求数                                 |
| `$msec`                   | 指定以秒为单位的时间，毫秒级别                               |
| `$pipe `                  | 指示请求是否是管道(p)                                        |
| `$request_length`         | 指定请求的长度，包括HTTP方法、URI、HTTP协议、头和请求体      |
| `$request_time`           | 单位秒，从接受用户请求的第一个字节到发送完响应数据的时间，包括接收客户端请求数据的时间、后端程序响应的时间、发送响应数据给客户端的时间(不包含写日志的时间) |
| `$upstream_response_time` | 单位秒，从Nginx向后端建立连接开始到接受完数据然后关闭连接为止的时间 |
| `$status`                 | 指定响应状态                                                 |
| `$time_iso8601`           | 指定本地时间，ISO8601格式                                    |
| `$time_local`             | 指定本地时间普通日志格式(%d/%b/%y:%H:%M:%S %z)               |

#### listen指令

```
listen address[:port];
listen port;
listen unix:path;
```

参数

| Listen指令的参数 | 说明                                                 | 注解                                                         |
| ---------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| default_server   | 定义一个组合:`address:port`默认的请求被绑定在此      |                                                              |
| setfib           | 为套接字监听设置相应的FIB                            | 仅支持FreeBSD，不支持UNIX域套接字                            |
| backlog          | 在listen()调用中设置backlog参数调用                  | 在FreeBSD系统中默认为-1，在其他中为511                       |
| rcvbuf           | 在套接字监听中，该参数设置SO_RCVBUF参数              |                                                              |
| sndbuf           | 在套接字监听中，该参数设置SO_SNDBUF参数              |                                                              |
| accept_filter    | 设置接受的过滤器，dataready或httpready dataready     | 仅支持FreeBSD                                                |
| deferred         | 该参数使用延迟的acept()调用设置TCP_DEFER_ACCEPT选项  | 仅支持Linux                                                  |
| bind             | 该参数为address:port套接字对打开一个单独的bind()调用 | 若其他特定套接字参数被使用，则一个单独的bind()将会被隐式地调用 |
| ipv6only         | 设置IPV6——V6ONLY参数的值                             | 只能在一个全新的开始设置，不支持UNIX域套接字                 |
| ssl              | 表明该端口仅接受HTTPS的链接                          | 允许更紧凑的配置                                             |
| so_keepalive     | 为TCP监听套接字配置keepalive                         |                                                              |

#### Location指令

用在虚拟服务器的server部分，提供来自客户端的URI或者内部的重定向访问

```
# 定义
location [modifier] uri ...
# 重命名
location @name ...
```

- 匹配规则

```shell
# 默认选择最长前缀
location = / {
   #规则A
}
location = /login {
   #规则B
}
location ^~ /static/ {
   #规则C
}
location ~ \.(gif|jpg|png|js|css)$ {
   #规则D，注意：是根据括号内的大小写进行匹配。括号内全是小写，只匹配小写
}
location ~* \.png$ {
   #规则E
}
location !~ \.xhtml$ {
   #规则F
}
location !~* \.xhtml$ {
   #规则G
}
location / {
   #规则H
}


# 含义
http://localhost/ 将匹配规则A
http://localhost/login 将匹配规则B
http://localhost/register 则匹配规则H
http://localhost/static/a.html 将匹配规则C
http://localhost/a.gif, http://localhost/b.jpg 将匹配规则D和规则E，但是规则D顺序优先，规则E不起作用， 而 http://localhost/static/c.png 则优先匹配到规则C
http://localhost/a.PNG 则匹配规则E， 而不会匹配规则D，因为规则E不区分大小写。
http://localhost/a.xhtml 不会匹配规则F和规则G
http://localhost/a.XHTML不会匹配规则G，（因为!）。规则F，规则G属于排除法，符合匹配规则也不会匹配到，所以想想看实际应用中哪里会用到。
http://localhost/category/id/1111 则最终匹配到规则H，因为以上规则都不匹配，这个时候nginx转发请求给后端应用服务器，比如FastCGI（php），tomcat（jsp），nginx作为方向代理服务器存在。
```
- 指令

| location的指令 | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| root           | root将外部的URI拼接在其后                                    |
| alias          | 定义location的其他名字，在文件系统中能够找到。若location指定了一个正则表达式，alias将会引用正则表达式中定义的捕获。alias指令替代location中匹配的URI部分，没有匹配的部分会在文件系统中搜索。当配置改变一点，配置中使用alias指令则会有脆弱的表现，因此推荐用root。除非为了找文件而需要修改URI |
| internal       | 指定一个仅用于内部请求的location(其他指定定义的重定向、rewrite请求、error请求等) |
| limit_except   | 限定一个location可以执行的HTTP操作(GET也包括HEAD)            |
| proxy_pass     | 对location的uri，proxy_pass不带URI时进行拼接，带URI时则进行替换 |
示例

```shell
location ^~ /t/ {
    root /www/root/html/;  # 进行拼接
}
# 请求是/t/a.html时，返回服务器上的/www/root/html/t/a.html的文件。

location ^~ /t/ {
 	alias /www/root/html/new_t/;   # 进行替换
}
# 请求有/t/a.html时，返回服务器上的/www/root/html/new_t/a.html的文件。

location /api/ {
	proxy_pass http://localhost:8080;  # 不带URI，进行拼接
}
# 访问http://localhost/api/xxx时，代理到http://localhost:8080/api/xxx

location /api/ {
	proxy_pass http://localhost:8080/;  # 带URI，进行替换
}
# 访问http://localhost/api/xxx时，代理到http://localhost:8080/xxx
```

## 反向代理

Nginx能够作为一个反向代理来终结来自客户端的请求，并且向上游服务器打开一个新的请求

```shell
# 在请求传递到上游服务器时，url发生变化
location /uri{
    proxy_pass http://localhost:8080/newuri;  # 带URI，进行替换
}
location /api/ {
	proxy_pass http://localhost:8080;  # 不带URI，进行拼接
}

# 有以下情况例外,不发生转换
# 正则
location ~ ^/local{
    proxy_pass http://localhost:8080/foreign;
}
# rewrite
location / {
    rewrite /(.*)$ /index.php?page=$1 break;
    proxy_pass http://localhost:8080/index;
}
```

### 代理模块 

| Proxy模块指令                 | 说明                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `proxy_connect_timeout`       | 指明Nginx从接受请求到连接到上游服务器的最长等待时间，默认60s |
| proxy_cookie_domain           | 替代从上游服务器来的Set-Cookie头中的domain属性；domain被替换为一个字符串、一个正则表达式，或者是引用的变量 |
| proxy_cookie_path             | 替代从上游服务器来的Set-Cookie头中的path属性；path被替换为一个字符串、一个正则表达式，或者是引用的变量 |
| proxy_headers_has_bucket_size | 指定头名字的最大值                                           |
| proxy_headers_hash_max_size   | 指定从上游服务器接收到头的总大小                             |
| proxy_hide_header             | 指定不应该传递给客户端头的列表                               |
| proxy_http_version            | 指定用于同上游服务器通信的HTTP协议版本                       |
| proxy_ignore_client_abort     | 若该指令设置为on，则当客户端放弃连接后，Nginx将不会放弃同上游服务器的连接 |
| proxy_ignore_headers          | 当处理来自于上游服务器的响应时，该指令设置哪些头可以被忽略   |
| proxy_intercept_errors        | 若启用此指令，Nginx将会显示配置的error_page错误，而不是来自上游服务器的直接响应 |
| proxy_max_temp_file_size      | 在写入内存缓冲区时，当响应与内存缓冲区不匹配时，该指令给出溢出文件的最大值 |
| proxy_pass                    | 指定请求被传递到的上游服务器，格式为URL                      |
| proxy_pass_header             | 该指令覆盖掉在proxy_hide_header指令中设置的头，允许这些头传递到客户端 |
| proxy_pass_request_body       | 若设置为off,则阻止请求体发送到上游服务器                     |
| proxy_pass_request_header     | 如设置为off,则阻止请求头发送到上游服务器                     |
| `proxy_read_timeout`          | 给出链接关闭前从上游服务器两次成功的读操作耗时。若上游服务器处理请求比较慢，则该指令应设置的高些，默认60s |
| proxy_redirect                | 该指令重写来自于上游服务器的Location和Refresh头，这对于某种应用程序框架非常有用 |
| `proxy_send_timeout`          | 该指令指定在连接关闭之前，向上游服务器两次写成功的操作完成所需要的时间长度，默认60s |
| proxy_set_body                | 发送到上游服务器的请求体可能会被该指令的设置值修改           |
| proxy_set_header              | 该指令重写发送到上游服务器头的内容，也可以通过将某种头的值设置为空字符，而不发送某种头的方法实现 |
| proxy_temp_file_write_size    | 该指令限制在同一时间内缓冲到一个临时文件的数据量，以使得Nginx不会过长地阻止单个请求 |
| proxy_temp_path               | 该指令设定临时文件的缓冲，用于缓冲从上游服务器来的文件，可以设定目录的层次 |

### 带有cookie的遗留应用程序

```shell
server{
    server_name app.example.com;
    location /legacy1{
        proxy_cookie_domain legacy1.example.com app.example.com;
        proxy_cookie_path $uri /legacy1$uri;
        proxy_redirect default;
        proxy_pass http://legacy1.example.com/;
    }
    location /legacy2{
        proxy_cookie_domain legacy2.example.com app.example.com;
        proxy_cookie_path $uri /legacy2$uri;
        proxy_redirect default;
        proxy_pass http://legacy2.example.com/;
    }
    location /{
        proxy_pass http://localhost:8080;
    }
}
```

### upstream模块

upstream模块将会启用一个新的配置区段，在该区段定义了一组上游服务器。这些服务器可能被设置了不同的权重(权重越高的上游服务器将会被Nginx传递越多的连接),也可能是不同的类型(TCP与UNIX域)，也可能出于需要对服务器进行维护，故而标记为down

| upstream模块指令 | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| ip_hash          | 该指令通过IP地址的哈希值确保客户端均匀地连接所有服务器，键值基于C类地址 |
| keepalive        | 该指令指定每一个worker进程缓存到上游服务器的连接数，在使用HTTP连接时，proxy_http_version应设置为1.1,并将proxy_set_header设置为Connection "" |
| least_conn       | 该指令激活负载均衡算法，将请求发送到活跃连接数最少的那台服务器 |
| server           | 该指令为upstream定义一个服务器地址(带TCP端口号的域名、IP地址，或者是UNIX域套接字)和可选参数。参数如下：<br>1.weight:设置一个服务器的优先级优于其他服务器<br>2.max_fails:设置在fail_timeout时间之内尝试对一个服务器连接的最大次数，若超过，则标记为down，默认为1<br>3.fail_timeout:在这个指定的时间内服务器必须提供响应，若在这个时间内未收到响应，则服务器标记为down，默认为10s<br>4.backup:一旦其他服务器宕机，那么仅有该参数标记的机器才会接收请求<br>5.down:该参数标记为一个服务器不再接受任何请求 |

保持活动连接

```shell
# http连接
upstream apache{
    server 127.0.0.1:8080;
    keepalive 32;
}
location /{
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_pass http://apache;
}

# 非http,Nginx与2个memcached实例保持64个连接
upstream memcaches{
    server 10.0.100.10:11211;
    server 10.0.100.20:11211;
    keepalive 64
}

# 切换默认的轮询负载均衡算法为least_conn
upstream apache{
    least_conn;
    server 10.0.200.10:80;
    server 10.0.200.20:80;
    keepalive 32;
}
```

### 上游服务器

> 单个服务器

```shell
# Nginx会终止所有客户端连接，然后将代理所有请求到本地主机的TCP协议的8080端口上
server {
    location / {
        proxy_pass http://localhost:8080;
    }
}

# 扩展，提供静态文件，然后把剩余的请求发送到Apache
server {
    location / {
        try_files $uri @apache;
    }
    location @apache {
        proxy_pass http://127.0.0.1:8080;
    }
}
```

> 多个上游服务器

```shell
upstream app {
    server 127.0.0.1:9000;
    server 127.0.0.1:9001;
    server 127.0.0.1:9002;
}
server {
    location / {
        proxy_pass http://app;
    }
}
```

> Memcached上游服务器

与上游服务器通过HTTP进行通信时，使用proxy_pass指令。

在保持活动链接部分，Nginx能够将请求代理到不同类型的上游服务器，有相应的*_pass指令。

在Nginx中，memcached模块(默认开启)负责与memcached守护进程通信。因此，客户端和memcached守护进程间没有直接通信，在这种情况下，Nginx不是充当反向代理。memcached模块使得Nginx使用memecached协议回话。因此，key的查询能够在请求传递到应用程序服务器之前完成

```
upstream memcaches{
	server 10.0.100.10:11211;
	server 10.0.100.20:11211;
}
server{
	location / {
		// memcached_pass使用$memcached_key实现key的查找。
		set $memcached_key "$uri?$args";
		memcached_pass memcaches;
		// 若无响应值，则将请求传递到localhost
		error_page 404 = @appserver;
	}
	location @appserver{
        proxy_pass http://127.0.0.1:8080;
	}
}
```

> FastCGI上游服务器

使用FastCGI服务器在Nginx服务器后运行PHP应用程序。fastcgi模块默认被编译，通过fastcgi_pass指令可激活该模块，之后Nginx可使用FastCGI协议同一个或者多个上游服务器进行会话

```
location /{
    fastcgi_pass fastcgis;
}

upstream fastcgis {
    server 10.0.200.10:9000;
    server 10.0.200.20:9000;
    server 10.0.200.30:9000;
}
```

> SCGI上游服务器

通过内建的scgi模块使用SCGI协议通信，通过scgi_pass指令与上游服务器通信。

> uWSGI上游服务器

Nginx通过uwsgi模块提供基于Python的上游服务器的链接，使用uwsgi_pass指令指定上游服务器。

## HTTP服务器

### 系统架构

Nginx包含一个单一的master进程和多个worker进程。所有的进程都是单线程，并且设计为同时处理成千上万个连接。worker进程是处理连接的地方。Nginx使用了操作系统事件机制来快速响应这些请求。

master进程负责读取配置文件，处理套接字、派生worker进程、打开日志文件和编译嵌入式的Per脚本。master进程是一个可以通过处理信号响应来管理请求的进程。

worker进程运行在一个忙碌的事件循环处理中，用来处理进入的连接。每个Nginx模块被构筑在worker中，因此任何请求处理、过滤、处理代理的连接和更多的操作都在worker进程中完成。由于这种worker模型，操作系统可以单独处理每一个进程，并且调度处理程序最佳地运行在每一个处理器内核上。若有任何阻塞worker进程的进程，如磁盘I/O,则需配置的worker进程要多于CPU内核数，以便处理负载。

还有少数辅助程序的Nginx的master进程用于处理专门的任务。在这些进程中有cache loader 和cache manager进程。cache loader进程负责worker进程使用缓存的元数据准备。cache manager进程负责检查缓存条目及有效期

Nginx建立在一个模块化的方式之上。master进程提供了每个模块可以执行其功能的基础，每一个协议和处理程序作为自己的模块执行，各个模块连接在一起成为一个管道来处理连接和请求。在处理完成一个请求之后，交给一系列过滤器，在这些过滤器中响应会被处理。这些过滤器有的处理子请求，与的还Nginx的强大功能之一。

子请求是Nginx根据客户端发送的不同URL返回的不同结果。这依赖于配置，可能会多重嵌套和调用其他的子请求。过滤器能够从多个子请求收集响应，并将它们组合成一个响应发送给客户端。然后，最终确定响应并将其发送到客户端。在这种方式下，可以让多个模块发挥作用。

### 核心模块

#### 查找文件

Nginx为了响应一个请求，将请求传递给一个内容处理程序，有配置文件的location指令决定处理

无条件内容处理程序首先被尝试:perl,proxy_pass,flv,mp4等,若这些处理程序不匹配，则按顺序传递给下列操作:random index,index,autoindex,gzip_static,static。

处理以斜线结束请求的是indx处理程序。若gzip没有被激活，则static模块就会处理该请求.

这些模块如何在问价那系统上找到适当的文件或目录则由某些指令组合来决定。

root指令最好定义在一个默认的server指令内，或者至少在一个特定的location指令之外定义，以便它的有效接线为整个server

```
server{
	// 任何被访问的文件都会在/home/customer/html目录中找到
    root /home/customer/html;
    location /{
    	// 若只输入了域名部分，则提供index.html,无则index.htm
        index index.html index.htm;
        // 尝试查找其他文件，无则返回通用文件
        try_files $uri/ backups$uri /generic-not-found.html;
        // 检查被投递问价你的路径，如有包含链接的文件，则返回错误
        disable_symlinks if_not_owner from=$document_root;
    }
    // 若输入了/downloads，则得到一个HTML格式的目录列表
    location /downloads{
        autoindex on;
    }
    // 若是文件不存在，则返回404
}
```

http文件路径指令

| HTTP文件路径指令 | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| disable_symlinks | 确定在将一个文件提交给客户端之前，检查其是否是一个符号链接。可设定如下参数:<br>off:禁止检查符号链接(默认)<br>on:若路径中的任何一部分是一个链接，则拒绝访问<br>if_not_owner:若路径中的任何一部分是一个链接,而且链接有不同文件宿主,则拒绝访问from=part:如指定了部分路径，则这部分之前对符号链接不做检查，而之后的部分就会按照on或if_not_owner参数检查 |
| root             | 设置文档的根目录。URI将会附加在该指令的值后，可在文件系统中找到具体的文件 |
| try_files        | 对于给定的参数测试文件的存在性，若前面的文件都没有找到，则最后的条目将作为备用，所以确保最后一个路径或者命名的location存在，或者通过`=<status code>`设置一个返回状态代码 |

#### 域名解析

如果在upstream或`*_pass`指令中使用了逻辑名字而不是IP地址，则Nginx将会默认使用操作系统的解析器来获取IP地址。这种情况只会在upstream第一次被请求时发生。若在`*_pass`指令中使用了变量，则根本不会发生解析。

域名解析指令

| 域名解析指令 | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| resolver     | 该指令配置一个或多个域名服务器，用于解析上游服务器，将上游服务器的名字解析为IP地址。有一个可选的valid参数，它将会覆盖掉域名记录中的TTL |

为了使得Nginx能够重新解析IP地址，可以将逻辑名放在变量中。当Nginx解析到这个变量后，会让DNS查找并获取该IP地址。需要配置resolver

```
server {
    resolver 192.168.100.2;
    location /{
        set $backend upstream.example.com;
        proxy_pass http://$backend;
    }
}
```

#### 客户端交互

Nginx与客户端交互的方式有多种，这些方式可以从连接本身(IP地址、超时、存活时间等)到内容协商头的属性。

| HTTP客户端交互指令     | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| Default_type           | 设置响应的默认MIME类型。若文件的MIME类型不能被types指令指定的类型正确地匹配，那么将会适应该指令指定的类型。 |
| Error_page             | 定义一个用于访问的URI，在遇到设置的错误代码时将会由该URI提供访问。使用=号参数可以改变响应代码。如果=号的参数为空，那么响应代码来自于后面的URI，在这种情况下必须由某种上游服务器提供 |
| etag                   | 对于静态资源，该指令禁止自动产生ETag响应头(默认值为on)       |
| If_modified_since      | 通过比较if-modified-since请求头的值，该指令控制如何修改响应时间<br>off:该参数忽略if-modeified-since头<br>exact:该参数精确匹配<br>before:该参数修改响应时间小于或者等于if-modified-since头的值 |
| ignore_invalid_headers | 该指令禁止忽略无效名字的头(默认on).一个有效的名字是由ASCII字母、数字、连字符号，可能还会由下划线(underscores_in_headers指令控制)组成 |
| merge_slashes          | 该指令禁止移除多个斜线。默认值为on，这意味着Nginx将会压缩两个或者多个字符为一个 |
| Recursive_error_pages  | 该指令启用error_pages指令(默认off)实现多个重定向             |
| types                  | 设置MIME类型到文件扩展名的映射。Nginx在conf/mime.types文件中包含了大多数MIME类型的映射。大多数情况下适应include载入该文件就足够了 |
| Underscores_in_heades  | 该指令在客户请求头中启用适应下划线字符。如果保留了默认值off，那么评估这样的头将服从ignore_invalid_headers指令的值 |

err_page指令是Nginx中最灵活的指令，当有任何条件的错误出现时，我们都可以提供任何页面。这个页面可以是在本机上，也可以是由应用程序服务器提供的动态页面，甚至是一个完全不同站点上的页面

```
http{
    errpage 500 501 502 503 504 share/examples/nginx/50x.html;
    server{
        server_name www.example.com;
        root /home/customer/html;
        error_page 404 /404.html;
        location / {
			error_page 500 501 502 503 504 = @error_handler;
		}
		location /microsite{
			error_page 404  http://microsite.example.com/404.html;
		}
		location @error_handler{
			default_type text/html;
			proxy_pass http://127.0.0.1:8080;
		}
    }
}
```

#### 防止滥用

防止同一个IP地址每秒到服务器请求的连接数过多。

| HTTP limit指令       | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| limit_conn           | 指定一个共享内存区域(limit_conn_zone配置),并且指定每个键-值对的最大连接数 |
| limit_conn_log_level | 由于配置了limit_conn指令，在Nginx限制连接且达到连接限制时，此时将会产生错误日志，该指令用于设置日志的错误级别 |
| limit_conn_zone      | 该指令一个key，限制在limit_conn指令中作为第一个参数。第二个参数zone，表明用于存储key的共享内存区名字、当前每个key的连接数量以及zone的大小(name:size) |
| Limit_rate           | 该指令限制客户端下载内容的速率(单位为字节/秒)。速率限制在连接级别，这意味着一个单一的客户端可以打开多个连接增加其吞吐量 |
| limit_rate_after     | 在完成设定的字节数之后，该指令启用limit_rate限制             |
| limit_req            | 在共享内存(同limit_req_zone一起配置)中，对特定key设置并发请求能力的限制。并发数量可以通过第二个参数指定。若要求在两个请求之间没有延时，那么需要配置第三个参数nodelay |
| Limit_req_log_level  | 在Nginx使用limit_req指令限制请求数量后，通过该指令指定在什么级别下报告之日记录。在河里延时(delay)记录级别要小于指示(indicated)级别 |
| limit_req_zone       | 该指令指定key，限制在limit_req指令中作为第一个参数。第二个参数zone，表明用于存储key的共享内存名字、当前每个key的请求数量，以及zone的大小(name:size)。第三个参数rate，表明配置在限制之前，每秒(r/s)请求数，或者每分钟请求数(r/m) |
| Max_ranges           | 该指令设置在byte-range请求中最大的请求数量。设置为0，禁用对byte-rang的支持 |

限制每一个唯一IP地址访问限制在10个连接。需要注意的是，在代理上网的后面可能会有多个用户，都是从同一IP地址来的，所以日志中会记录有503(service unavaliable)错误代码，表示限制已经生效

```
http{
    limit_conn_zone $binary_romote_addr zone=connections:10m;
    limit_conn_log_level notice;
    server{
        limit_conn connections 10;
    }
}
```

基于速率的访问限制原理不同，在限制每个单元时间内一个用户可以请求多个页面时，Nginx将会在第一个页面请求后插入一个延时，直到这段时间过去。Nginx提供了可以通过nodealy参数消除这种延时方法

```
http{
    limit_req_zone $binary_remote_addr zone=requests:10m rate=1r/s;
    limit_req_log_level warn;
    server{
        limit_req zone=requests burst=10 nodelay;
    }
}
```

限制每个客户端的宽带。这种方法可以确保一些客户端不会把所有可用的宽带占完。警告：尽管limit_rate指令是连接基础，一个允许打开多个连接的哭护短仍然可以绕开这个限制。

```
location /downloads{
    limit_rate 500k;
}
```

允许小文件自由下载，对于大文件则试用这种限制

```
location /downloads{
    limit_rate_after 1m;
    limit_rate 50k;
}
```

#### 约束访问

限制访问整个网站或它的某些部分。有两种形式：对一组特定的IP地址限制，或者对一组特定用户限制。

| HTTP access模块指令  | 说明                                                         |
| -------------------- | ------------------------------------------------------------ |
| allow                | 允许从这个IP地址、网络或者值为all的访问                      |
| Auth_basic           | 启用基于HTTP基本认证。以字符串作为域的名字。如果设置为off，那么表示auth_basic不再继承上级的设置 |
| auth_basic_user_file | 指定一个文件的位置，该文件的格式为username:password:comment，用于用户认证。password部分需要使用密码算法加密处理。comment部分可选 |
| deny                 | 禁止从IP地址、网络或者值为all的访问                          |
| satisfy              | 若勤勉的指令使用了all或者any，那么允许访问。默认值为all，表示用户必须来自一个特定的网络地址，并且输入正确的密码 |

约束客户端访问来自于某一个特定的IP地址

```
# 仅允许本地访问/stats URI
location /stats{
    allow 127.0.0.1;
    deny all;
}
```

为了约束认证用户访问，可对`auth_basic`和`auth_basic_user_file`进行配置

```
# 任何想哟啊访问restricted.example.com的用户都需要提供匹配conf目录下htpasswd文件中的条目，conf目录依赖于Nginx的root目录。在htpasswd文件中的条目可以使用任何有效的使用标准UNIX crypt()函数产生的工具。
server{
	server_name restricted.example.com;
	auth_basic "restricted";
	auth_basic_user_file conf/htpasswd;
}
```

若没有设置从特定IP地址来的用户，在使用用户名和密码的处理方案中仅需要输入用户名和密码，Nginx有个satisfy指令，这里使用了any参数，是一种非此即彼的方案

```
server{
    server_name intranet.example.com;
    location /{
		auth_basic "intranet: please login";
		# select a user/password combo from this file
		auth_basic-user_file conf/htpasswd-intranet;
		# unless coming from one of these networks
		allow 192.168.40.0/24;
		allow 192.168.50.0/24;
		# deny access if these conditions aren't met
		deny all;
		# if either condition is met, allow access
		satisy any;
	}
}
```

若需要为来自特定IP地址的用户配置并提供认证，那么在默认情况下使用all参数。因此，会忽略satisfy指令本身，并且仅包括allow,deny,auth_basic和auth_basic_user_file

```
server{
    server_name stage.example.com;
    location /{
		auth_basic "staging server";
		# select a user/password combo from this file
		auth_basic-user_file conf/htpasswd-stage;
		# unless coming from one of these networks
		allow 192.168.40.0/24;
		allow 192.168.50.0/24;
		# deny access if these conditions aren't met
		deny all;
	}
}
```

#### 流媒体文件

Nginx能够提供一定的视频媒体类型。flv和mp4模块包含在基本的发布中，它们能够提供伪流媒体(pseudo-streaming)。这意味着Nginx将会在特定的location中搜索视频文件，通过start请求参数来指明

为了使用伪流媒体功能，需要在编译时添加响应的模块:`--with-http_flv_module`用于Flash视频(flv)文件，`--with-http_mp4_module`用于H.264/AAC文件。

| HTTP 流指令         | 说明                            |
| ------------------- | ------------------------------- |
| flv                 | 在location中激活flv模块         |
| mp4                 | 在location中激活mp4模块         |
| Mp4_buffer_size     | 设置投递MP4文件的初始缓冲大小   |
| Mp4_max_buffer_size | 设置处理MP4元数据使用的最大缓冲 |

激活flv

```
location /videos {
	flv;
}
```

激活MP4

```
location /videos{
	mp4;
	mp4_buffer_size 1m;
	mp4_max_buffer_size 20m;
}
```

#### 预定义变量

基于变量值使得构建Nginx配置变得容易，不仅能够使用set或map指令自己设置指令，也可以使用Nginx内部使用预定义变量。为了快速优化变量，并且该缓存变量的值也将在整个请求内有效。可以在if生命中使用它们作为一个key，或者将它们传递到代理服务器。

| HTTP变量名称                                                 | 值                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| $arg_name                                                    | 指定在请求中的name参数                                       |
| $args                                                        | 指定所有请求参数                                             |
| $binary_remote_addr                                          | 指定客户端IP地址的二进制格式                                 |
| $content_length                                              | 指定请求头Content-Length的值                                 |
| $content_type                                                | 指定请求头Content-Type的值                                   |
| $cookie_name                                                 | 指定cookie标签名字                                           |
| $document_root                                               | 指定当前请求中指令root或者alias的值                          |
| $document_uri                                                | 指定$uri的别名 |                                                              |
| $host                                                        | 若当前有Host，该变来个则指定请求头host的值，若无这个头，则该值等于匹配该请求的server_name的值 |
| $hostname                                                    | 指定运行Nginx主机的主机名                                    |
| $http_name                                                   | 指定请求头name值，若这个头有破折号，会被转换为下划线，大写字母转为小写字母 |
| $https                                                       | 若连接是通过SSL的，则这个变量的值是on，否则为空字符串        |
| $is_args                                                     | 若请求有参数，则变量的值为?,否则为空字符串                   |
| $limit_rate                                                  | 指定指令limit_rate的值，若无设置，允许速率限制使用这个变量   |
| $nginx_version                                               | 指定允许的Nginx二进制版本                                    |
| $pid                                                         | 指定worker进程的ID                                           |
| $query_string                                                | 指定$args的别名 |                                                              |
| $realpath_root                                               | 指定当前请求中指令root和alias的值，用所有的符号链接解决问题  |
| $remote_addr                                                 | 指定客户端的IP                                               |
| $remote_port                                                 | 指定客户端的端口                                             |
| $remote_user                                                 | 使用http基本认证是，变量用于设置用户名                       |
| $request                                                     | 指定从客户端收到的完整请求，包括HTTP请求方法、URI、HTTP协议、头、请求体 |
| $request_body                                                | 指定请求体，在location中由`*_pass`指令处理                   |
| $request_body_file                                           | 指定临时文件的路径，在临时文件中存储请求体。对于这个被保存的文件，client_body_in_file_only指令需要被设置为on |
| $request_completion                                          | 若请求完成，该变量值为OK，否则为空字符串                     |
| $request_filename                                            | 该变量指定当前请求文件的路径及文件名，基于root或者alias指令的值加上URI |
| $request_method                                              | 指定当前请求使用的HTTP方法                                   |
| $request_uri                                                 | 指定完整请求的URI，从客户端来的请求，包括参数                |
| $scheme                                                      | 指定当前请求的协议，不是HTTP，就是HTTPS                      |
| $sent_http_name                                              | 指定响应头名字的值，若这个头有破折号，那么他们将会被转换为下划线，大写字母被转换为小写 |
| $server_addr                                                 | 指定接受请求服务器的地址值                                   |
| $server_name                                                 | 指定接受请求的虚拟主机server_name的值                        |
| $server_port                                                 | 指定接受请求的服务器端口                                     |
| $server_protocol                                             | 指定在当前请求中使用的HTTP协议                               |
| $status                                                      | 指定响应状态                                                 |
| `$tcpinfo_rtt<br>$tcpinfo_rttvar<br>$tcpinfo_snd_cwnd<br>$tcpinfo_rcv_space` | 若系统支持TCP_INFO套接字选项，这些变量将会被相关的信息填充   |
| $uri                                                         | 指定当前请求的变转化URI                                      |

