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

### Http的server

> 客户端指令

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

###文件I/O指令

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

### Hash指令

控制Nginx分配给某些变量多大的静态内存。在启动和重新配置时，会计算需要的最小值。在Nginx发出警告时，只需要调整一个`*_hash_max_size`指令的参数值就可以达到效果。`*_hash_bucket_size`变量被设置了默认值，以便满足多处理器缓存行降低检索需要的检索查找。

| hash指令                      | 说明                                  |
| ----------------------------- | ------------------------------------- |
| server_names_hash_bucket_size | 指定用于保存server_name散列表大小的桶 |
| server_names_hash_max_size    | 指定server_name散列表的最大大小       |
| types_hash_bucket_size        | 指定用于存储散列表的桶的大小          |
| types_hash_max_size           | 指定散列类型比饿哦的最大大小          |
| variables_hash_bucket_size    | 指定用于存储保留变量桶大小            |
| variables_hash_max_size       | 指定存储保留变量最大散列值得大小      |

### socket指令

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

listen指令的参数

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

```

