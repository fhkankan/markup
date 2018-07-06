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

# Apache

mac

```
# mac自带apachae
# 启动
sudo apachectl start
# 

```



# Tomcat

