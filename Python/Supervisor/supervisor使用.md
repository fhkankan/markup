# supervisor

Supervisor是用Python开发的一套通用的进程管理程序，能将一个普通的命令行进程变为后台daemon，并监控进程状态，异常退出时能自动重启。它是通过fork/exec的方式把这些被管理的进程当作supervisor的子进程来启动，这样只要在supervisor的配置文件中，把要管理的进程的可执行文件的路径写进去即可。也实现当子进程挂掉的时候，父进程可以准确获取子进程挂掉的信息的，可以选择是否自己启动和报警。supervisor还提供了一个功能，可以为supervisord或者每个子进程，设置一个非root的user，这个user就可以管理它对应的进程。

## 安装

```
# pip 安装：
pip install supervisor

# Debian / Ubuntu可以直接通过apt安装：
apt-get install supervisor
```

## 配置文件

### apt-get

- 文件位置

```python
# 通过apt-get install安装后的配置文件位置
# 默认的配置文件可能不全，但是够用
etc/supervisor/supervisord.conf         
```

- 子进程配置文件

```python
# 子进程配置
etc/supervisor/conf.d/*.conf
```

示例文件

```python
# etc/supervisor/conf.d/test.conf

#项目名
[program:blog]
#脚本目录
directory=/opt/bin
#脚本执行命令
command=/usr/bin/python /opt/bin/test.py
#supervisor启动的时候是否随着同时启动，默认True
autostart=true
#当程序exit的时候，这个program不会自动重启,默认unexpected
#设置子进程挂掉后自动重启的情况，有三个选项，false,unexpected和true。如果为false的时候，无论什么情况下，都不会被重新启动，如果为unexpected，只有当进程的退出码不在下面的exitcodes里面定义的
autorestart=false
#这个选项是子进程启动多少秒之后，此时状态如果是running，则我们认为启动成功了。默认值为1
startsecs=1
#日志输出 
stderr_logfile=/tmp/blog_stderr.log 
stdout_logfile=/tmp/blog_stdout.log 
#脚本运行的用户身份 
user = zhoujy 
#把 stderr 重定向到 stdout，默认 false
redirect_stderr = true
#stdout 日志文件大小，默认 50MB
stdout_logfile_maxbytes = 20M
#stdout 日志文件备份数
stdout_logfile_backups = 20


[program:zhoujy] #说明同上
directory=/opt/bin 
command=/usr/bin/python /opt/bin/zhoujy.py 
autostart=true 
autorestart=false 
stderr_logfile=/tmp/zhoujy_stderr.log 
stdout_logfile=/tmp/zhoujy_stdout.log 
#user = zhoujy
```

### easy_install

通过easy_install安装后，配置文件不存在，需要自己导入

```python
# 运行 echo_supervisord_conf，查看配置样本：
echo_supervisord_conf

# 创建配置文件：
echo_supervisord_conf > /etc/supervisord.conf

# 在supervisor中的;[program:theprogramname]里修改配置信息
```

详细子进程配置

```
[unix_http_server]
file=/tmp/supervisor.sock   ;UNIX socket 文件，supervisorctl 会使用
;chmod=0700                 ;socket文件的mode，默认是0700
;chown=nobody:nogroup       ;socket文件的owner，格式：uid:gid
 
;[inet_http_server]         ;HTTP服务器，提供web管理界面
;port=127.0.0.1:9001        ;Web管理后台运行的IP和端口，如果开放到公网，需要注意安全性
;username=user              ;登录管理后台的用户名
;password=123               ;登录管理后台的密码
 
[supervisord]
logfile=/tmp/supervisord.log ;日志文件，默认是 $CWD/supervisord.log
logfile_maxbytes=50MB        ;日志文件大小，超出会rotate，默认 50MB，如果设成0，表示不限制大小
logfile_backups=10           ;日志文件保留备份数量默认10，设为0表示不备份
loglevel=info                ;日志级别，默认info，其它: debug,warn,trace
pidfile=/tmp/supervisord.pid ;pid 文件
nodaemon=false               ;是否在前台启动，默认是false，即以 daemon 的方式启动
minfds=1024                  ;可以打开的文件描述符的最小值，默认 1024
minprocs=200                 ;可以打开的进程数的最小值，默认 200
 
[supervisorctl]
serverurl=unix:///tmp/supervisor.sock ;通过UNIX socket连接supervisord，路径与unix_http_server部分的file一致
;serverurl=http://127.0.0.1:9001 ; 通过HTTP的方式连接supervisord
 
; [program:xx]是被管理的进程配置参数，xx是进程的名称
[program:xx]
command=/opt/apache-tomcat-8.0.35/bin/catalina.sh run  ; 程序启动命令
autostart=true       ; 在supervisord启动的时候也自动启动
startsecs=10         ; 启动10秒后没有异常退出，就表示进程正常启动了，默认为1秒
autorestart=true     ; 程序退出后自动重启,可选值：[unexpected,true,false]，默认为unexpected，表示进程意外杀死后才重启
startretries=3       ; 启动失败自动重试次数，默认是3
user=tomcat          ; 用哪个用户启动进程，默认是root
priority=999         ; 进程启动优先级，默认999，值小的优先启动
redirect_stderr=true ; 把stderr重定向到stdout，默认false
stdout_logfile_maxbytes=20MB  ; stdout 日志文件大小，默认50MB
stdout_logfile_backups = 20   ; stdout 日志文件备份数，默认是10
; stdout 日志文件，需要注意当指定目录不存在时无法正常启动，所以需要手动创建目录（supervisord 会自动创建日志文件）
stdout_logfile=/opt/apache-tomcat-8.0.35/logs/catalina.out
stopasgroup=false     ;默认为false,进程被杀死时，是否向这个进程组发送stop信号，包括子进程
killasgroup=false     ;默认为false，向进程组发送kill信号，包括子进程
 
;包含其它配置文件
[include]
files = relative/directory/*.ini    ;可以指定一个或多个以.ini结束的配置文件
```

详解

```python
[unix_http_server]            
file=/tmp/supervisor.sock   ; socket文件的路径，supervisorctl用XML_RPC和supervisord通信就是通过它进行
                              的。如果不设置的话，supervisorctl也就不能用了  
                              不设置的话，默认为none。 非必须设置        
;chmod=0700                 ; 这个简单，就是修改上面的那个socket文件的权限为0700
                              不设置的话，默认为0700。 非必须设置
;chown=nobody:nogroup       ; 这个一样，修改上面的那个socket文件的属组为user.group
                              不设置的话，默认为启动supervisord进程的用户及属组。非必须设置
;username=user              ; 使用supervisorctl连接的时候，认证的用户
                               不设置的话，默认为不需要用户。 非必须设置
;password=123               ; 和上面的用户名对应的密码，可以直接使用明码，也可以使用SHA加密
                              如：{SHA}82ab876d1387bfafe46cc1c8a2ef074eae50cb1d
                              默认不设置。。。非必须设置

;[inet_http_server]         ; 侦听在TCP上的socket，Web Server和远程的supervisorctl都要用到他
                              不设置的话，默认为不开启。非必须设置
;port=127.0.0.1:9001        ; 这个是侦听的IP和端口，侦听所有IP用 :9001或*:9001。
                              这个必须设置，只要上面的[inet_http_server]开启了，就必须设置它
;username=user              ; 这个和上面的uinx_http_server一个样。非必须设置
;password=123               ; 这个也一个样。非必须设置

[supervisord]                ;这个主要是定义supervisord这个服务端进程的一些参数的
                              这个必须设置，不设置，supervisor就不用干活了
logfile=/tmp/supervisord.log ; 这个是supervisord这个主进程的日志路径，注意和子进程的日志不搭嘎。
                               默认路径$CWD/supervisord.log，$CWD是当前目录。。非必须设置
logfile_maxbytes=50MB        ; 这个是上面那个日志文件的最大的大小，当超过50M的时候，会生成一个新的日 
                               志文件。当设置为0时，表示不限制文件大小
                               默认值是50M，非必须设置。              
logfile_backups=10           ; 日志文件保持的数量，上面的日志文件大于50M时，就会生成一个新文件。文件
                               数量大于10时，最初的老文件被新文件覆盖，文件数量将保持为10
                               当设置为0时，表示不限制文件的数量。
                               默认情况下为10。。。非必须设置
loglevel=info                ; 日志级别，有critical, error, warn, info, debug, trace, or blather等
                               默认为info。。。非必须设置项
pidfile=/tmp/supervisord.pid ; supervisord的pid文件路径。
                               默认为$CWD/supervisord.pid。。。非必须设置
nodaemon=false               ; 如果是true，supervisord进程将在前台运行
                               默认为false，也就是后台以守护进程运行。。。非必须设置
minfds=1024                  ; 这个是最少系统空闲的文件描述符，低于这个值supervisor将不会启动。
                               系统的文件描述符在这里设置cat /proc/sys/fs/file-max
                               默认情况下为1024。。。非必须设置
minprocs=200                 ; 最小可用的进程描述符，低于这个值supervisor也将不会正常启动。
                              ulimit  -u这个命令，可以查看linux下面用户的最大进程数
                              默认为200。。。非必须设置
;umask=022                   ; 进程创建文件的掩码
                               默认为022。。非必须设置项
;user=chrism                 ; 这个参数可以设置一个非root用户，当我们以root用户启动supervisord之后。
                               我这里面设置的这个用户，也可以对supervisord进行管理
                               默认情况是不设置。。。非必须设置项
;identifier=supervisor       ; 这个参数是supervisord的标识符，主要是给XML_RPC用的。当你有多个
                               supervisor的时候，而且想调用XML_RPC统一管理，就需要为每个
                               supervisor设置不同的标识符了
                               默认是supervisord。。。非必需设置
;directory=/tmp              ; 这个参数是当supervisord作为守护进程运行的时候，设置这个参数的话，启动
                               supervisord进程之前，会先切换到这个目录
                               默认不设置。。。非必须设置
;nocleanup=true              ; 这个参数当为false的时候，会在supervisord进程启动的时候，把以前子进程
                               产生的日志文件(路径为AUTO的情况下)清除掉。有时候咱们想要看历史日志，当 
                               然不想日志被清除了。所以可以设置为true
                               默认是false，有调试需求的同学可以设置为true。。。非必须设置
;childlogdir=/tmp            ; 当子进程日志路径为AUTO的时候，子进程日志文件的存放路径。
                               默认路径是这个东西，执行下面的这个命令看看就OK了，处理的东西就默认路径
                               python -c "import tempfile;print tempfile.gettempdir()"
                               非必须设置
;environment=KEY="value"     ; 这个是用来设置环境变量的，supervisord在linux中启动默认继承了linux的
                               环境变量，在这里可以设置supervisord进程特有的其他环境变量。
                               supervisord启动子进程时，子进程会拷贝父进程的内存空间内容。 所以设置的
                               这些环境变量也会被子进程继承。
                               小例子：environment=name="haha",age="hehe"
                               默认为不设置。。。非必须设置
;strip_ansi=false            ; 这个选项如果设置为true，会清除子进程日志中的所有ANSI 序列。什么是ANSI
                               序列呢？就是我们的\n,\t这些东西。
                               默认为false。。。非必须设置

; the below section must remain in the config file for RPC
; (supervisorctl/web interface) to work, additional interfaces may be
; added by defining them in separate rpcinterface: sections
[rpcinterface:supervisor]    ;这个选项是给XML_RPC用的，当然你如果想使用supervisord或者web server 这 
                              个选项必须要开启的
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface 

[supervisorctl]              ;这个主要是针对supervisorctl的一些配置  
serverurl=unix:///tmp/supervisor.sock ; 这个是supervisorctl本地连接supervisord的时候，本地UNIX socket
                                        路径，注意这个是和前面的[unix_http_server]对应的
                                        默认值就是unix:///tmp/supervisor.sock。。非必须设置
;serverurl=http://127.0.0.1:9001 ; 这个是supervisorctl远程连接supervisord的时候，用到的TCP socket路径
                                   注意这个和前面的[inet_http_server]对应
                                   默认就是http://127.0.0.1:9001。。。非必须项
                               
;username=chris              ; 用户名
                               默认空。。非必须设置
;password=123                ; 密码
                              默认空。。非必须设置
;prompt=mysupervisor         ; 输入用户名密码时候的提示符
                               默认supervisor。。非必须设置
;history_file=~/.sc_history  ; 这个参数和shell中的history类似，我们可以用上下键来查找前面执行过的命令
                               默认是no file的。。所以我们想要有这种功能，必须指定一个文件。。。非
                               必须设置

; The below sample program section shows all possible program subsection values,
; create one or more 'real' program: sections to be able to control them under
; supervisor.

;[program:theprogramname]      ;这个就是咱们要管理的子进程了，":"后面的是名字，最好别乱写和实际进程
                                有点关联最好。这样的program我们可以设置一个或多个，一个program就是
                                要被管理的一个进程
;command=/bin/cat              ; 这个就是我们的要启动进程的命令路径了，可以带参数
                                例子：/home/test.py -a 'hehe'
                                有一点需要注意的是，我们的command只能是那种在终端运行的进程，不能是
                                守护进程。这个想想也知道了，比如说command=service httpd start。
                                httpd这个进程被linux的service管理了，我们的supervisor再去启动这个命令
                                这已经不是严格意义的子进程了。
                                这个是个必须设置的项
;process_name=%(program_name)s ; 这个是进程名，如果我们下面的numprocs参数为1的话，就不用管这个参数
                                 了，它默认值%(program_name)s也就是上面的那个program冒号后面的名字，
                                 但是如果numprocs为多个的话，那就不能这么干了。想想也知道，不可能每个
                                 进程都用同一个进程名吧。

                                
;numprocs=1                    ; 启动进程的数目。当不为1时，就是进程池的概念，注意process_name的设置
                                 默认为1    。。非必须设置
;directory=/tmp                ; 进程运行前，会前切换到这个目录
                                 默认不设置。。。非必须设置
;umask=022                     ; 进程掩码，默认none，非必须
;priority=999                  ; 子进程启动关闭优先级，优先级低的，最先启动，关闭的时候最后关闭
                                 默认值为999 。。非必须设置
;autostart=true                ; 如果是true的话，子进程将在supervisord启动后被自动启动
                                 默认就是true   。。非必须设置
;autorestart=unexpected        ; 这个是设置子进程挂掉后自动重启的情况，有三个选项，false,unexpected
                                 和true。如果为false的时候，无论什么情况下，都不会被重新启动，
                                 如果为unexpected，只有当进程的退出码不在下面的exitcodes里面定义的退 
                                 出码的时候，才会被自动重启。当为true的时候，只要子进程挂掉，将会被无
                                 条件的重启
;startsecs=1                   ; 这个选项是子进程启动多少秒之后，此时状态如果是running，则我们认为启
                                 动成功了
                                 默认值为1 。。非必须设置
;startretries=3                ; 当进程启动失败后，最大尝试启动的次数。。当超过3次后，supervisor将把
                                 此进程的状态置为FAIL
                                 默认值为3 。。非必须设置
;exitcodes=0,2                 ; 注意和上面的的autorestart=unexpected对应。。exitcodes里面的定义的
                                 退出码是expected的。
;stopsignal=QUIT               ; 进程停止信号，可以为TERM, HUP, INT, QUIT, KILL, USR1, or USR2等信号
                                  默认为TERM 。。当用设定的信号去干掉进程，退出码会被认为是expected
                                  非必须设置
;stopwaitsecs=10               ; 这个是当我们向子进程发送stopsignal信号后，到系统返回信息
                                 给supervisord，所等待的最大时间。 超过这个时间，supervisord会向该
                                 子进程发送一个强制kill的信号。
                                 默认为10秒。。非必须设置
;stopasgroup=false             ; 这个东西主要用于，supervisord管理的子进程，这个子进程本身还有
                                 子进程。那么我们如果仅仅干掉supervisord的子进程的话，子进程的子进程
                                 有可能会变成孤儿进程。所以咱们可以设置可个选项，把整个该子进程的
                                 整个进程组都干掉。 设置为true的话，一般killasgroup也会被设置为true。
                                 需要注意的是，该选项发送的是stop信号
                                 默认为false。。非必须设置。。
;killasgroup=false             ; 这个和上面的stopasgroup类似，不过发送的是kill信号
;user=chrism                   ; 如果supervisord是root启动，我们在这里设置这个非root用户，可以用来
                                 管理该program
                                 默认不设置。。。非必须设置项
;redirect_stderr=true          ; 如果为true，则stderr的日志会被写入stdout日志文件中
                                 默认为false，非必须设置
;stdout_logfile=/a/path        ; 子进程的stdout的日志路径，可以指定路径，AUTO，none等三个选项。
                                 设置为none的话，将没有日志产生。设置为AUTO的话，将随机找一个地方
                                 生成日志文件，而且当supervisord重新启动的时候，以前的日志文件会被
                                 清空。当 redirect_stderr=true的时候，sterr也会写进这个日志文件
;stdout_logfile_maxbytes=1MB   ; 日志文件最大大小，和[supervisord]中定义的一样。默认为50
;stdout_logfile_backups=10     ; 和[supervisord]定义的一样。默认10
;stdout_capture_maxbytes=1MB   ; 这个东西是设定capture管道的大小，当值不为0的时候，子进程可以从stdout
                                 发送信息，而supervisor可以根据信息，发送相应的event。
                                 默认为0，为0的时候表达关闭管道。。。非必须项
;stdout_events_enabled=false   ; 当设置为ture的时候，当子进程由stdout向文件描述符中写日志的时候，将
                                 触发supervisord发送PROCESS_LOG_STDOUT类型的event
                                 默认为false。。。非必须设置
;stderr_logfile=/a/path        ; 这个东西是设置stderr写的日志路径，当redirect_stderr=true。这个就不用
                                 设置了，设置了也是白搭。因为它会被写入stdout_logfile的同一个文件中
                                 默认为AUTO，也就是随便找个地存，supervisord重启被清空。。非必须设置
;stderr_logfile_maxbytes=1MB   ; 这个出现好几次了，就不重复了
;stderr_logfile_backups=10     ; 这个也是
;stderr_capture_maxbytes=1MB   ; 这个一样，和stdout_capture一样。 默认为0，关闭状态
;stderr_events_enabled=false   ; 这个也是一样，默认为false
;environment=A="1",B="2"       ; 这个是该子进程的环境变量，和别的子进程是不共享的
;serverurl=AUTO                ; 

; The below sample eventlistener section shows all possible
; eventlistener subsection values, create one or more 'real'
; eventlistener: sections to be able to handle event notifications
; sent by supervisor.

;[eventlistener:theeventlistenername] ;这个东西其实和program的地位是一样的，也是suopervisor启动的子进
                                       程，不过它干的活是订阅supervisord发送的event。他的名字就叫
                                       listener了。我们可以在listener里面做一系列处理，比如报警等等
                                       楼主这两天干的活，就是弄的这玩意
;command=/bin/eventlistener    ; 这个和上面的program一样，表示listener的可执行文件的路径
;process_name=%(program_name)s ; 这个也一样，进程名，当下面的numprocs为多个的时候，才需要。否则默认就
                                 OK了
;numprocs=1                    ; 相同的listener启动的个数
;events=EVENT                  ; event事件的类型，也就是说，只有写在这个地方的事件类型。才会被发送
                      
                                 
;buffer_size=10                ; 这个是event队列缓存大小，单位不太清楚，楼主猜测应该是个吧。当buffer
                                 超过10的时候，最旧的event将会被清除，并把新的event放进去。
                                 默认值为10。。非必须选项
;directory=/tmp                ; 进程执行前，会切换到这个目录下执行
                                 默认为不切换。。。非必须
;umask=022                     ; 淹没，默认为none，不说了
;priority=-1                   ; 启动优先级，默认-1，也不扯了
;autostart=true                ; 是否随supervisord启动一起启动，默认true
;autorestart=unexpected        ; 是否自动重启，和program一个样，分true,false,unexpected等，注意
                                  unexpected和exitcodes的关系
;startsecs=1                   ; 也是一样，进程启动后跑了几秒钟，才被认定为成功启动，默认1
;startretries=3                ; 失败最大尝试次数，默认3
;exitcodes=0,2                 ; 期望或者说预料中的进程退出码，
;stopsignal=QUIT               ; 干掉进程的信号，默认为TERM，比如设置为QUIT，那么如果QUIT来干这个进程
                                 那么会被认为是正常维护，退出码也被认为是expected中的
;stopwaitsecs=10               ; max num secs to wait b4 SIGKILL (default 10)
;stopasgroup=false             ; send stop signal to the UNIX process group (default false)
;killasgroup=false             ; SIGKILL the UNIX process group (def false)
;user=chrism                   ;设置普通用户，可以用来管理该listener进程。
                                默认为空。。非必须设置
;redirect_stderr=true          ; 为true的话，stderr的log会并入stdout的log里面
                                默认为false。。。非必须设置
;stdout_logfile=/a/path        ; 这个不说了，好几遍了
;stdout_logfile_maxbytes=1MB   ; 这个也是
;stdout_logfile_backups=10     ; 这个也是
;stdout_events_enabled=false   ; 这个其实是错的，listener是不能发送event
;stderr_logfile=/a/path        ; 这个也是
;stderr_logfile_maxbytes=1MB   ; 这个也是
;stderr_logfile_backups        ; 这个不说了
;stderr_events_enabled=false   ; 这个也是错的，listener不能发送event
;environment=A="1",B="2"       ; 这个是该子进程的环境变量
                                 默认为空。。。非必须设置
;serverurl=AUTO                ; override serverurl computation (childutils)

; The below sample group section shows all possible group values,
; create one or more 'real' group: sections to create "heterogeneous"
; process groups.

;[group:thegroupname]  ;这个东西就是给programs分组，划分到组里面的program。我们就不用一个一个去操作了
                         我们可以对组名进行统一的操作。 注意：program被划分到组里面之后，就相当于原来
                         的配置从supervisor的配置文件里消失了。。。supervisor只会对组进行管理，而不再
                         会对组里面的单个program进行管理了
;programs=progname1,progname2  ; 组成员，用逗号分开
                                 这个是个必须的设置项
;priority=999                  ; 优先级，相对于组和组之间说的
                                 默认999。。非必须选项

; The [include] section can just contain the "files" setting.  This
; setting can list multiple files (separated by whitespace or
; newlines).  It can also contain wildcards.  The filenames are
; interpreted as relative to this file.  Included files *cannot*
; include files themselves.

;[include]                         ;这个东西挺有用的，当我们要管理的进程很多的时候，写在一个文件里面
                                    就有点大了。我们可以把配置信息写到多个文件中，然后include过来
;files = relative/directory/*.ini
```

## 组件

- supervisord

supervisord是supervisor的服务端程序。

干的活：启动supervisor程序自身，启动supervisor管理的子进程，响应来自clients的请求，重启闪退或异常退出的子进程，把子进程的stderr或stdout记录到日志文件中，生成和处理Event

- supervisorctl

这东西还是有点用的，如果说supervisord是supervisor的服务端程序，那么supervisorctl就是client端程序了。supervisorctl有一个类型shell的命令行界面，我们可以利用它来查看子进程状态，启动/停止/重启子进程，获取running子进程的列表等等。。。最牛逼的一点是，supervisorctl不仅可以连接到本机上的supervisord，还可以连接到远程的supervisord，当然在本机上面是通过UNIX socket连接的，远程是通过TCP socket连接的。supervisorctl和supervisord之间的通信，是通过xml_rpc完成的。    相应的配置在[supervisorctl]块里面

- Web Server

Web Server主要可以在界面上管理进程，Web Server其实是通过XML_RPC来实现的，可以向supervisor请求数据，也可以控制supervisor及子进程。配置在[inet_http_server]块里面

- XML_RPC接口

这个就是远程调用的，上面的supervisorctl和Web Server就是它弄的

## 运行

```python
# apt-get install安装的supervisor：
/etc/init.d/supervisor start

# 通过easy_install 安装的supervisor
supervisord
```

## web页面

需要在supervisor的配置文件里添加[inet_http_server]选项组：之后可以访问控制子线程的管理

```
[inet_http_server]
port=10.211.55.11:9001
username=user
password=123
```

## 子进程管理

```python
# 查看所有子进程的状态
supervisorctl status
# 关闭指定子进程
supervisorctl stop 进程名
# 开启指定的子进程
supervisorctl start 进程名
# 重启指定的子进程
supervisorctl restart 进程名
# 关闭所有子进程
supervisorctl stop all
# 开启所有子进程
supervisorctl start all
# 配置文件修改后可以使用该命令加载新的配置
supervisorctl update
# 重新启动配置中的所有程序
supervisorctl reload
```

## event

supervisor的event机制其实，就是一个监控/通知的框架。抛开这个机制实现的过程来说的话，event其实就是一串数据，这串数据里面有head和body两部分。咱们先弄清楚event数据结构，咱们才能做后续的处理。

- header

```python
ver:3.0 server:supervisor serial:21 pool:listener poolserial:10 eventname:PROCESS_COMMUNICATION_STDOUT len:54
```

详细说明

| key        | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| ver        | 表示event协议的版本，目前是3.0                               |
| server     | 表示supervisor的标识符，也就是咱们上一篇中[supervisord]块中的identifier选项中的东西；默认为supervisor |
| serial     | 这个东西是每个event的序列号，supervisord在运行过程中，发送的第一个event的序列号就是1，接下来的event依次类推 |
| pool       | 这个是你的listener的pool的名字，一般你的listener只启动一个进程的的话，其实也就没有         pool的概念了。名字就是[eventlistener:theeventlistenername]这个东西 |
| poolserial | 上面的serial是supervisord给每个event的编号。 而poolserial则是，eventpool给发送到我这个pool过来的event编的号 |
| eventname  | 这个是event的类型名称，这个后面说                            |
| len        | 这个长度，表示的是header后面的body部分的长度。header之后，我们会取len长度的内容作为body |

- body

body的数据结构，其实是和event的具体类型相关的，不同的event的类型，header的结构都一样，但是body的结构大多就不一样了。

以PROCESS_STATE_EXITED类型进行分析，当supervisord管理的子进程退出的时候，supervisord就会产生PROCESS_STATE_EXITED这么个event。

```python
processname:cat groupname:cat from_state:RUNNING expected:0 pid:2766
```

详细说明

| key         | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| processname | 就是进程名字，这里名字不是我们实际进程的名字，而是咱们[program:x]配置成的名字 |
| groupname   | 组名，这个一个样                                             |
| from_state  | 这个是，我们的进程退出前的状态是什么状态                     |
| expected    | 这个咱们前面也讲过，默认情况下exitcodes是0和2，也就是说0和2是expected。其它的退出码，也就是unexpected了 |
| pid         | 进程号                                                       |

​    OK，说到了这里，我们知道了event的产生，然后给我们的listener这么一种结构的数据。

现在我们有数据了，就看咱们怎么去处理这些数据了，这个过程就仁者见仁，智者见智了。我们可以利用接收的数据，加工后，进行报警，等等操作。

​    处理数据之前，咱们还得要来了解一下，listener和supervisord之间的通信过程

​    在这里我们首先要搞清楚，event的发起方和接收方。

​    event的发起方是supervisord进程，接收方是一个叫listener的东西，listener怎么配置，上一篇参数详解里面已经写的很清楚了，大伙可以去参考下，这里就不赘述了。其实listener和program一样，都是supervisord的子进程。两者的在配置上，很多选项也都一样。

​    其实，event还有另外一个过程，我们的program也就是我们要管理的进程，也可以发送event，进而和supervisord主动通信。不过program程序一般都是程序员们搞，咱们搞运维的就不管他们的事情了

OK，看看event协议。

```
1. 当supervisord启动的时候，如果我们的listener配置为autostart=true的话，listener就会作为supervisor的子进程被启动。
2. listener被启动之后，会向自己的stdout写一个"READY"的消息,此时父进程也就是supervisord读取到这条消息后，会认为listener处于就绪状态。
3. listener处于就绪状态后，当supervisord产生的event在listener的配置的可接受的events中时，supervisord就会把该event发送给该listener。  
4. listener接收到event后，我们就可以根据event的head，body里面的数据，做一些列的处理了。我们根据event的内容，判断，提取，报警等等操作。
5. 该干的活都干完之后，listener需要向自己的stdout写一个消息"RESULT\nOK"，supervisord接受到这条消息后。就知道listener处理event完毕了。
```

好，来看看例子吧

```python
#!/usr/bin/env python
#coding:utf-8

import sys
import os
import subprocess
#childutils这个模块是supervisor的一个模型，可以方便我们处理event消息。。。当然我们也可以自己按照协议，用任何语言来写listener，只不过用childutils更加简便罢了
from supervisor import childutils
from optparse import OptionParser
import socket
import fcntl
import struct

__doc__ = "\033[32m%s,捕获PROCESS_STATE_EXITED事件类型,当异常退出时触发报警\033[0m" % sys.argv[0]

def write_stdout(s):
    sys.stdout.write(s)
    sys.stdout.flush()
#定义异常，没啥大用其实
class CallError(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)
#定义处理event的类
class ProcessesMonitor():
    def __init__(self):
        self.stdin = sys.stdin
        self.stdout = sys.stdout

    def runforever(self):
        #定义一个无限循环，可以循环处理event，当然也可以不用循环，把listener的autorestart#配置为true，处理完一次event就让该listener退出，然后supervisord重启该listener，这样listen#er就可以处理新的event了
        while 1:
            #下面这个东西，是向stdout发送"READY"，然后就阻塞在这里，一直等到有event发过来
            #headers,payload分别是接收到的header和body的内容
            headers, payload = childutils.listener.wait(self.stdin, self.stdout)
            #判断event是否是咱们需要的，不是的话，向stdout写入"RESULT\NOK"，并跳过当前
            #循环的剩余部分
            if not headers['eventname'] == 'PROCESS_STATE_EXITED':
                childutils.listener.ok(self.stdout)
                continue

            pheaders,pdata = childutils.eventdata(payload+'\n')
            #判读event是否是expected是否是expected的，expected的话为1，否则为0
            #这里的判断是过滤掉expected的event
            if int(pheaders['expected']):
                childutils.listener.ok(self.stdout)
                continue

            ip = self.get_ip('eth0')
            #构造报警信息结构
            msg = "[Host:%s][Process:%s][pid:%s][exited unexpectedly fromstate:%s]" % (ip,pheaders['processname'],pheaders['pid'],pheaders['from_state'])
            #调用报警接口，这个接口是我们公司自己开发的，大伙不能用的，要换成自己的接口
            subprocess.call("/usr/local/bin/alert.py -m '%s'" % msg,shell=True)
            #stdout写入"RESULT\nOK"，并进入下一次循环
            childutils.listener.ok(self.stdout)


    '''def check_user(self):
        userName = os.environ['USER']
        if userName != 'root':
            try:
                raise MyError('must be run by root!')
            except MyError as e:
                write_stderr( "Error occurred,value:%s\n" % e.value)
                sys.exit(255)'''

    def get_ip(self,ifname):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        inet = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15]))
        ret = socket.inet_ntoa(inet[20:24])
        return ret


def main():
    parser = OptionParser()
    if len(sys.argv) == 2:
        if sys.argv[1] == '-h' or sys.argv[1] == '--help':
            print __doc__
            sys.exit(0)
    #(options, args) = parser.parse_args()
    #下面这个，表示只有supervisord才能调用该listener，否则退出
    if not 'SUPERVISOR_SERVER_URL' in os.environ:
        try:
            raise CallError("%s must be run as a supervisor event" % sys.argv[0])
        except CallError as e:
            write_stderr("Error occurred,value: %s\n" % e.value)

        return

    prog = ProcessesMonitor()
    prog.runforever()

if __name__ == '__main__':
    main()
```

其他常用的event类型，已经listener的三种状态，已经怎么转换的。可以去官网上看看

## xml_rpc

supervisor提供的两种管理方式，supervisorctl和web其实都是通过xml_rpc来实现的。

xml_rpc其实就是本地可以去调用远端的函数方法，然后函数方法经过一番处理后，把结果返回给我们。

在python里面实现xml_rpc就更加的简单，用SimpleXMLRPCServer和xmlrpclib这两个模块就可以分别实现服务端和客户端了。

调用supervisor的xml_rpc接口，其实很简单。先做好下面这两步

```python
import xmlrpclib
p = xmlrpclib.Server('http://localhost:9001/RPC2')
```

注意xmlrpclib.Server()里面的url和咱们supervisor.conf里的配置是相关的

做完上面的步骤，我们就可以得到一个叫做p的对象。p这个对象，有很多存放在服务端的方法。

supervisor默认的xml_rpc方法定义在下面这个路径里面

```bash
/usr/local/lib/python2.7/dist-packages/supervisor-3.1.0-py2.7.egg/supervisor/rpcinterface.py
```

我们可以使用system.listMethods()的方法，来查看服务端都有哪些方法可供调用？

```bash
>>>server.system.listMethods()
['supervisor.addProcessGroup', 'supervisor.clearAllProcessLogs', 'supervisor.clearLog', 'supervisor.clearProcessLog', 'supervisor.clearProcessLogs', 'supervisor.getAPIVersion', 'supervisor.getAllConfigInfo', 'supervisor.getAllProcessInfo', 'supervisor.getIdentification', 'supervisor.getPID', 'supervisor.getProcessInfo', 'supervisor.getState', 'supervisor.getSupervisorVersion', 'supervisor.getVersion', 'supervisor.readLog', 'supervisor.readMainLog', 'supervisor.readProcessLog', 'supervisor.readProcessStderrLog', 'supervisor.readProcessStdoutLog', 'supervisor.reloadConfig', 'supervisor.removeProcessGroup', 'supervisor.restart', 'supervisor.sendProcessStdin', 'supervisor.sendRemoteCommEvent', 'supervisor.shutdown', 'supervisor.startAllProcesses', 'supervisor.startProcess', 'supervisor.startProcessGroup', 'supervisor.stopAllProcesses', 'supervisor.stopProcess', 'supervisor.stopProcessGroup', 'supervisor.tailProcessLog', 'supervisor.tailProcessStderrLog', 'supervisor.tailProcessStdoutLog', 'system.listMethods', 'system.methodHelp', 'system.methodSignature', 'system.multicall']
```

我们如果想知道某一个方法怎么用，可以用system.methodHelp(name)去查看，例如：

```python
server.system.methodHelp('supervisor.startProcess')
```

这么查看其实还是有点麻烦的，直接去官网看吧，官网上列举了常用方法的用法。其实supervisor本身提供的xml_rpc的方法有很多很多，包括查看进程状态，启动/停止/重启进程，查看日志，发送event等等。

有了这些方法，我们就可以向远处执行相应的操作。或者获取想要的数据，OK，后续数据怎么处理，怎么用，就可以根据大伙的实际需求去发挥了。

还有上面的每个方法都是supervisor.x的形式，前面的supervisor其实是，我们定义在

[rpcinterface:supervisor]，rpc接口的名称。

既然有，rpc接口需要名称，那么显然名称是为了区分rpc接口。在supervisor里面，如果我们觉得supervisor自带的rpc接口函数不够用，那么我们就可以定义自己的rpc接口函数。自己定义的函数可以直接写进rpcinterface.py里面去。不过为了不污染人家原有的东西，最好别这么干。

supervisord中rpc接口函数定义的方法，除了需要在supervisord.conf中加上一块配置外，还需要一个函数签名。

先看看supervisord.conf中怎么定义吧。配置文件中找个地方，放入下面这么段东西。里面具体的接口名称，路径，及签名函数的名称，大伙可以自己去指定了。我的形式是这个样子的

```bash
[rpcinterface:myrpc]
supervisor.rpcinterface_factory = myrpc.rpc:my_rpc
args = 1
```

注意，第二行的args = 1。表示传入my_rpc这个签名函数的参数。supervisor中，签名函数的第一个参数必须为"supervisord"，后面可以没有别的参数，以key/value的形式传入。

其他参数如同args = 1的形式，放在[rpcinterface:myrpc]的块里面

OK,我们就用上面的配置，来举个小例子，来看看自定义rpc接口的完整实现。

先看看，myrpc.rpc,rpc.py这个自定义模块里面是什么？

```python
#!/usr/bin/env python

class Rpc(object):
    def __init__(self,supervisord,args):
        self.supervisord = supervisord
        self.args = args

    def walk_args(self):
        return self.walk

def my_rpc(supervisord,**args):
     return Rpc(supervisord,args)
```

启动supervisord之后，进行下面的操作

```
impot xmlrpclib
p = xmlrpclib.Server('http://localhost:9001/RPC2')
p.system.listMethods()
p.myrpc.walk_args()
```

可以看到，刚才定义的那个函数出来了，而且执行成功了