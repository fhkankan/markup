# Nginx

## 安装

mac

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

## 使用

服务启动

```
sudo nginx
```

服务重启

```
sudo nginx -s reload
```

服务关闭

```
sudo nginx -s stop
sudo nginx -s quit
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
        listen       8080;  # 端口号
        server_name  localhost;  # 可为ip地址
        location / {
            root   html;  # 根目录位置
            index  index.html index.htm;
            autoindex on;  # 显示目录
        }
}
```

修改能够上传的数据大小

```
http {
	client_max_body_size 8M;  # 客户端上传最大单文件大小
	client_body_buffer_size 128k; # 缓存大小
    include       mime.types;
    default_type  application/octet-stream;
```

反向代理

```
server{
    listen [你要监听的端口号];
    server_name [你要监听的域名/IP];
 
    location / {
        proxy_pass [代理的目标地址];
     }
}
```

## 常见问题

403

```
# 问题一：
网站目录下无index.html文件
# 问题二：
网站目录无权限,chmod 755
```

## 配置指南

### 全局配置参数

| 全局配置指令       | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| user               | 配置worker进程的用户和组，若忽略group，则group的名字等于该参数指定用户的用户组 |
| worker_processes   | 指定woeker进程启动的数量。用于处理客户的所有连接，经验法则是设置该参数的值与cpu绑定的负载处理器核心数数量相同，并用1.5~2之间的数乘以这个数作为I/O密集型负载 |
| error_log          | 所有错误写入的日志，第二个参数指定了错误的级别(debug,info,notice,warn,error,crit,alert,emerg)。debug级别的错误需在编译时配置了--with-debug |
| pid                | 设置记录主进程ID的文件，会覆盖编译时的默认配置               |
| use                | 用于指示使用什么样的连接方式。会覆盖编译时的默认配置。若配置此指令，需要一个events区段。 |
| worker_connections | 配置一个工作进程能够接受并发连接的最大数。这个连接包括客户连接和向上游服务器的连接，但并不限于此。对于反向代理服务器尤为重要，为达到这个并发性连接数量，需要在操作系统层面进行一些额外调整。 |

### Http的server部分

#### 客户端指令

| http客户端指令               | 说明                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| chunked_transfer_encodeing   | 在发送给客户端的响应中，该指令允许禁用http/1.1标准的块传输编码 |
| client_body_buffer_size      | 为了阻止临时文件写到磁盘，可以通过该指令为客户端请求体设置缓存大小，默认大小为两个内存页面 |
| client_body_in_file_only     | 用于调试或进一步处理客户端请求体。该指令设置为“on”能够将客户端请求体强制写入到磁盘文件 |
| client_body_in_single_buffer | 为了减少复制的操作，使用该指令强制Nginx将整个客户端请求保存在单个缓存中 |
| client_body_temp_path        | 定义一个命令路径用于保存客户端请求体                         |
| client_body_timeout          | 指定客户体成功读取的两个操作之间的时间间隔                   |
| client_header_buffer_size    | 为客户端请求头指定一个缓存大小，当请求头大于1kb时会用到这个设置 |
| client_header_timeout        | 该超时是读取整个客户端头的时间长度                           |
| client_max_body_size         | 定义允许最大的客户端请求头，若大于这个设置，客户端将会是413错误 |
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

### 虚拟服务器

server开始的部分被称为"虚拟服务器"部分。描述的是一组根据不同的server_name指令逻辑分割的资源。响应http请求。

一个虚拟服务器由listen和server_name指令组合定义。

```
listen address[:port];
listen port;
listen unix:path;
```

#### listen指令

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

### Location指令

用在虚拟服务器的server部分，提供来自客户端的URI或者内部的重定向访问

```
# 定义
location [modifier] uri ...
# 重命名
location @name ...
```

| location的指令 | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| alias          | 定义location的其他名字，在文件系统中能够找到。若location指定了一个正则表达式，alias将会引用正则表达式中定义的捕获。alias指令替代location中匹配的URI部分，没有匹配的部分会在文件系统中搜索。当配置改变一点，配置中使用alias指令则会有脆弱的表现，因此推荐用root。除非为了找问价而需要修改URI |
| internal       | 指定一个仅用于内部请求的location(其他指定定义的重定向、rewrite请求、error请求等) |
| limit_except   | 限定一个location可以执行的HTTP操作(GET也包括HEAD)            |

## 反向代理

Nginx能够作为一个反向代理来终结来自客户端的请求，并且向上游服务器打开一个新的请求

```
# 在请求传递到上游服务器时，/uri会被替换为/newuri
location /uri{
    proxy_pass http://localhost:8080/newuri;
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
| proxy_connect_timeout         | 指明Nginx从接受请求到连接到上游服务器的最长等待时间          |
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
| proxy_read_timeout            | 给出链接关闭前从上游服务器两次成功的读操作耗时。若上游服务器处理请求比较慢，则该指令应设置的高些 |
| proxy_redirect                | 该指令重写来自于上游服务器的Location和Refresh头，这对于某种应用程序框架非常有用 |
| proxy_send_timeout            | 该指令指定在连接关闭之前，向上游服务器两次写成功的操作完成所需要的时间长度 |
| proxy_set_body                | 发送到上游服务器的请求体可能会被该指令的设置值修改           |
| proxy_set_header              | 该指令重写发送到上游服务器头的内容，也可以通过将某种头的值设置为空字符，而不发送某种头的方法实现 |
| proxy_temp_file_write_size    | 该指令限制在同一时间内缓冲到一个临时文件的数据量，以使得Nginx不会过长地阻止单个请求 |
| proxy_temp_path               | 该指令设定临时文件的缓冲，用于缓冲从上游服务器来的文件，可以设定目录的层次 |

### 带有cookie的遗留应用程序

```
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
| server           | 该指令为upstream定义一个服务器地址(带TCP端口号的域名、IP地址，或者是UNIX域套接字)和可选参数。参数如下：<br>1.weight:设置一个服务器的优先级优于其他服务器<br>2.max_fails:设置在fail_timeout时间之内尝试对一个服务器连接的最大次数，若超过，则标记为down<br>3.fail_timeout:在这个指定的时间内服务器必须提供响应，若在这个时间内未收到响应，则服务器标记为down<br>4.backup:一旦其他服务器宕机，那么仅有该参数标记的机器才会接收请求<br>5.down:该参数标记为一个服务器不再接受任何请求 |

### 保持活动连接

```
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

```
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

```
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

#### server指令

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

#### 日志

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

| 日子格式变量名称     | 值                                                           |
| -------------------- | ------------------------------------------------------------ |
| $body_bytes_sent     | 指定发送到客户端的字节数，不包括响应头                       |
| $bytes_sent          | 指定发送到客户端的字节数                                     |
| $connection          | 指定一个串号，用于标识一个唯一的连接                         |
| $connection_requests | 指定通过一个特定连接的请求数                                 |
| $msec                | 指定以秒为单位的时间，毫秒级别                               |
| $pipe *              | 指示请求是否是管道(p)                                        |
| $request_length *    | 指定请求的长度，包括HTTP方法、URI、HTTP协议、头和请求体      |
| $request_time        | 指定请你去的处理时间，毫秒级，从客户端接收到第一个字节到客户端接收完最后一个字节 |
| $status              | 指定响应状态                                                 |
| $time_iso8601 *      | 指定本地时间，ISO8601格式                                    |
| $time_local *        | 指定本地时间普通日志格式(%d/%b/%y:%H:%M:%S %z)               |

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