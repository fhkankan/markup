# 屏幕分辨率

```
xrandr -s 1920x1080 

# 注意：x是英文字母x，临时生效
```

# 卸载

```
# 卸载不需要的软件
sudo apt-get remove  unity-webapps-common  totem  rhythmbox simple-scan gnome-mahjongg aisleriot gnome-mines  transmission-common gnome-orca webbrowser-app gnome-sudoku onboard deja-dup firefox libreoffice-style-galaxy thunderbird
# 清理缓存
sudo apt-get autoremove
sudo apt-get autoclean 

# 使用dpkg卸载
sudo dpkg -l  # 查看已安装软件
sudo dpkg -P ...  # 删除已安装软件
```
# 更换源

- 方法一：

图形化界面中，更改设置中的软件与更新，选择中国阿里云

- 方法二：

命令行修改

```
# 打开配置文件
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
sudo vi /etc/apt/sources.list
# 添加如下163 ubuntu16.04源
deb http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse
deb http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-security main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-updates main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-proposed main restricted universe multiverse
deb-src http://mirrors.163.com/ubuntu/ xenial-backports main restricted universe multiverse
# 添加优麒麟的源
echo deb http://archive.ubuntukylin.com:10006/ubuntukylin trusty main | sudo tee /etc/apt/sources.list.d/ubuntukylin.list 
# 更新源
sudo apt-get update 
# 注:如果提示没有公钥,无法验证下列数字签名 xxx
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-　keys xxxx
sudo apt-get update
```
# 常用软件安装

| 软件名               | 安装方式 | 说明              |
| -------------------- | -------- | ----------------- |
| git                  | 常规     | 版本管理          |
| vlc                  | 常规     | 音视频播放器      |
| unrar                | 常规     | 解压缩            |
| gparted              | 常规     | 磁盘管理          |
| nginx                | 常规     | 负载均衡          |
| mongodb              | 常规     | NoSQL数据库       |
| bleachbit            | 常规     | 清理记录          |
| htop                 | 常规     | 查看资源占用      |
| openssh-server       | 常规     | ssh连接           |
| meld                 | 常规     | 代码对比          |
| vpnc                 | 常规     | vpn客户端         |
| chrome               | deb      | 浏览器            |
| vim                  | 加仓     | 编辑器            |
| typora               | 加仓     | 编辑器            |
| R                    | 加仓     | R语言             |
| mysql                | 常规     | 关系型数据库      |
| redis                | 加仓     | key-value数据库   |
| wine-de              | 加仓     | windows环境       |
| Thunderbird          | 加仓     | 邮件客户端        |
| gimp                 | 加仓     | 图片编辑器        |
| indicator-sysmonitor | 加仓     | 电脑监控客户端    |
| remarkable           | deb      | 编辑器            |
| Nodepadqq            | 加仓     | 编辑器            |
| pycharm              | deb      | pythonIDE         |
| visualStudioCode     | deb      | 编辑器            |
| sogou                | deb      | 输入法            |
| opera                | deb      | 浏览器            |
| dbeaver              | deb      | 数据库管理客户端  |
| navicat              | tar      | 数据库管理客户端  |
| postman              | tar      | 接口请求工具      |
| robo3t               | tar      | mongodb管理客户端 |
| wps                  | deb      | office            |
| polipo               |          |                   |
| ss-qt5               |          |                   |
| pac manager          |          |                   |
| gdb-dashboard        |          |                   |
| vmware               |          |                   |

## 常规安装
```
sudo apt-get install git vpnc vlc unrar gparted nginx mongodb bleachbit htop openssh-server meld filezilla
```

## 添加仓库

vim

```
sudo add-apt-repository ppa:jonathonf/vim
sudo apt update
sudo apt install vim ctags vim-doc
```

typora

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA300B7755AFCFAE
sudo add-apt-repository 'deb http://typora.io linux/'
sudo apt-get update
sudo apt-get install typora
```

R

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
sudo sh -c "echo deb http://mirror.bjtu.edu.cn/cran/bin/linux/ubuntu precise/ >>/etc/apt/sources.list"
sudo apt-get install r-base
sudo apt-get install r-base-dev
sudo apt-get update
```
mysql

```
sudo apt-get install mysql-server
sudo apt-get install libmysqld-dev 
sudo apt-get install libmysqlclient-dev 
```
redis

```
sudo apt-get install software-properties-common
sudo apt-add-repository ppa:chris-lea/redis-server
sudo apt-get update
sudo apt-get install redis-server
```
wine-de

```
sudo add-apt-repository ppa:wine/wine-builds
sudo apt-get update
sudo apt-get install wine-devel
sudo apt install winehq-devel
```

Thunderbird

```
sudo add-apt-repository ppa:ubuntu-mozilla-security/ppa
sudo apt-get update
sudo apt-get install thunderbird
```

gimp

```
# 安装
sudo add-apt-repository ppa:otto-kesselgulasch/gimp
sudo apt-get update
sudo apt-get install gimp
# 卸载(可选)
sudo apt-get install ppa-purge
sudo ppa-purge ppa:otto-kesselgulasch/gimp
```

indicator-sysmonitor

```
sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor 
sudo apt-get update 
sudo apt-get install indicator-sysmonitor
```
Nodepadqq

```
sudo add-apt-repository ppa:notepadqq-team/notepadqq
sudo apt-get update
sudo apt-get install notepadqq
```

pycharm

```
sudo add-apt-repository ppa:mystic-mirage/pycharm
sudo apt update
sudo apt install pycharm-community
```

HandBrake

```
sudo add-apt-repository ppa:stebbins/handbrake-releases
sudo apt-get update
apt-get install handbrake-gtk
apt-get install handbrake-cli   
```

## 安装deb

- 下载deb
> 官网下载

```
visualStudioCode
dbeaver
```
> 下载工具

chrome

```
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb 
sudo apt-get install libappindicator1 libindicator7 
```

remarkable

```
wget https://remarkableapp.github.io/files/remarkable_1.87_all.deb
sudo dpkg -i remarkable_1.87_all.deb
sudo apt-get -f install 
```

- 安装
```
sudo dpkg -i 安装包名字
```
- 修复依赖项
```
sudo apt-get -f install
```
## 使用tar
- 管网下载
```
pycharm
navicat
robo3t
postman
```
- 配置信息

pycharm

```sh
# 设置快捷键
# 方法一：建立命令行快捷
cd ~
vim .bashrc
alias pycharm="bash /home/application/pycharm-2019.3.2/bin/pycharm.sh"
source .bashrc
# 方法一：建立软链接(可添加启动器)
sudo ln -s bin/pycharm.sh的文件目录 /usr/bin/pycharm
# 方法二：使用快捷图标(可添加启动器)
sudo gedit /usr/share/applications/Pycharm.desktop
[Desktop Entry]
    Type=Application
    Name=Pycharm
    GenericName=Pycharm3
    Comment=Pycharm3:The Python IDE
    Exec="/home/frankguo/pycharm-community-2017.3.3/bin/pycharm.sh" %f
    Icon= /home/frankguo/pycharm-community-2017.3.3/bin/pycharm.png
    Terminal=pycharm
    Categories=Pycharm;
sudo chmod +x /usr/share/applications/pycharm.desktop
```

## 其他软件

### wps

```
# 官网下载wps..
sudo dpkg -i wps...
# 1. 下载缺失的字体文件
# 国外下载地址：https://www.dropbox.com/s/lfy4hvq95ilwyw5/wps_symbol_fonts.zip
# 国内下载地址：https://pan.baidu.com/s/1eS6xIzo
# 解压并进入目录中，继续执行：
sudo cp * /usr/share/fonts
# 2. 执行以下命令,生成字体的索引信息：
sudo mkfontscale
sudo mkfontdir
# 3. 运行fc-cache命令更新字体缓存。
sudo fc-cache
# 4. 重启wps即可，字体缺失的提示不再出现。
```

### polipo

socket5代理转换为http

```
sudo apt-get install polipo
# 停止服务
sudo service polipo stop
# 修改配置文件
sudo vi /etc/polipo/config
# 新增如下两个配置
socksParentProxy = localhost:1080
proxyPort = 8787
# 启动服务
$ sudo service polipo start
# 添加命令代理别名
$ cd
$ vi .bashrc
# 添加配置如下
 alias http_proxy='export http_proxy=http://127.0.0.1:8787/'
alias https_proxy='export https_proxy=http://127.0.0.1:8787/'
# 生效配置
$ source .bashrc 
```
### ss-qt5
```
# 安装curl
$ sudo apt install curl
# 安装ss-qt5
# github地址：https://github.com/shadowsocks/shadowsocks-qt5/releases
# 使用命令下载
$ wget https://github-production-release-asset-2e65be.s3.amazonaws.com/18427187/04086db8-f3cd-11e7-9c68-2b0d4b4dbe5b?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20180112%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20180112T150901Z&X-Amz-Expires=300&X-Amz-Signature=73c91b88196b277e49f46a9d0874d4d76384c6b35d33861b3c9b55a4396b03f7&X-Amz-SignedHeaders=host&actor_id=0&response-content-disposition=attachment%3B%20filename%3DShadowsocks-Qt5-3.0.0-x86_64.AppImage&response-content-type=application%2Foctet-stream

# 启动ss-qt5客户端，添加配置，启动服务
# 测试代理
$ http_proxy
$ https_proxy
$ curl http://ip.cn
```
### pac manager（xshell替代）
```
$ wget https://astuteinternet.dl.sourceforge.net/project/pacmanager/pac-4.0/pac-4.5.5.7-all.deb
$ sudo dpkg -i pac-4.5.5.7-all.deb 
$ sudo apt-get install -f
$ sudo apt-get install libgtk2-appindicator-perl
```
### gdb-dashboard
```
wget -P ~ git.io/.gdbinit
mv ~/.gdbinit ~/.gdb-dashboard
# 然后在使用gdb调试的时候可以在gdb界面调用gdb-dashboard
(gdb) source ~/.gdb-dashboard
# 也可以直接修改~/.gdbinit,加入source ~/.gdb-dashboard使gdb在载入时自动加载gdb-dashboard
```
### VMWare
```
# 永久许可证秘钥： 
注：如果是WinXP或32位系统请用 10.0 版本；11.0 版本之后支持Win7或更高版64位系统。

VMware 所有版本永久许可证激活密钥：

VMware Workstation v14 for Windows 
FF31K-AHZD1-H8ETZ-8WWEZ-WUUVA
CV7T2-6WY5Q-48EWP-ZXY7X-QGUWD

VMware Workstation v12 for Windows 
5A02H-AU243-TZJ49-GTC7K-3C61N 
VF5XA-FNDDJ-085GZ-4NXZ9-N20E6
UC5MR-8NE16-H81WY-R7QGV-QG2D8
ZG1WH-ATY96-H80QP-X7PEX-Y30V4
AA3E0-0VDE1-0893Z-KGZ59-QGAVF

VMware Workstation v11 for Windows 
1F04Z-6D111-7Z029-AV0Q4-3AEH8 

VMware Workstation v10 for Windows 
1Z0G9-67285-FZG78-ZL3Q2-234JG 
4C4EK-89KDL-5ZFP9-1LA5P-2A0J0 
HY086-4T01N-CZ3U0-CV0QM-13DNU 

VMware Workstation v9 for Windows 
4U434-FD00N-5ZCN0-4L0NH-820HJ 
4V0CP-82153-9Z1D0-AVCX4-1AZLV 
0A089-2Z00L-AZU40-3KCQ2-2CJ2T 

VMware Workstation v8 for Windows 
A61D-8Y0E4-QZTU0-ZR8XP-CC71Z 
MY0E0-D2L43-6ZDZ8-HA8EH-CAR30 
MA4XL-FZ116-NZ1C9-T2C5K-AAZNR 

VMware Workstation v7 for Windows 
VZ3X0-AAZ81-48D4Z-0YPGV-M3UC4 
VU10H-4HY97-488FZ-GPNQ9-MU8GA 
ZZ5NU-4LD45-48DZY-0FNGE-X6U86 

VMware Workstation v6 for Windows 
UV16D-UUC6A-49H6E-4E8DY 
C3J4N-3R22V-J0H5R-4NWPQ 
A15YE-5250L-LD24E-47E7C 

VMware Workstation v6 ACE Edition for Windows 
TK08J-ADW6W-PGH7V-4F8FP 
YJ8YH-6D4F8-9EPGV-4DZNA 
YCX8N-4MDD2-G130C-4GR4L
```
### axel

linux下轻量级下载加速工具,axel是一个多线程分段下载工具，可以从ftp或http服务器进行下载。

安装

```
sudo apt-get install axel
```

格式  

```
axel [OPTIONS] url1 url2
```

参数

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
axel -n 30 https://download.jetbrains.8686c.com/cpp/CLion-2017.3.dmg

需要指定http://或ftp://。
```

### utorrent
 安装

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



## 设置开启小键盘

```
sudo apt-get install numlockx
sudo gedit /usr/share/lightdm/lightdm.conf.d/50-unity-greeter.conf

# 在配置文件最后添加：
greeter-setup-script=/usr/bin/numlockx on
```
## 同步windows时间

```
sudo timedatectl set-local-rtc 1  
sudo apt-get install ntpdate
sudo ntpdate time.windows.com
sudo hwclock --localtime --systohc
```

# 系统镜像

cubic

```
可对官方镜像做软件的删除、增加，再生成iso
```

systemback

```
可对当前系统做备份和封装成iso安装镜像
```

# 开启ftp服务

```shell
# 给服务器创建一个目录
mkdir ~/ftp
# 创建存放用户上传的文件的目录
cd ~/ftp
mkdir anonymous
chomd 777 anonymous
# 安装ftp服务器
sudo apt-get install vsftpd
# 配置vsftpd.conf文件
sudo vi /etc/vsftpd.conf
# 重启服务器，使其重新加载配置项
sudo /etc/init.d/vsftpd restart
```

修改如下设置，允许匿名用户(可在最后直接添加)

```
anonymous_enable=YES
anon_root=/home/……/ftp
no_anon_password=YES
write_enable=YES
anon_upload_enable=YES
anon_mkdir_write_enable=YES
```

上传下载

```shell
# 1.登录，按照提示输入用户名和密码
cd testFile  # 进入上传下载文件夹
ftp 127.0.0.1  # 连接ftp服务器
# 2.上传下载文件
ls	# 显示远程服务器中文件
get filename1  # 下载远程文件至本地
put filename2  # 上传本地文件至远程
```

# 开启ssh服务

- 安装开启停用

检查是否安装开启

```
sudo ps -e | grep sshd
```

开启

```
sudo /etc/init.d/ssh start
```

安装

````
sudo apt-get update
sudo apt-get install openssh-server
````

服务相关

```
sudo service ssh status   # 查看状态
sudo service ssh stop   # 关闭服务
sudo service ssh restart  # 重启服务
```

- 配置

客户端配置(`/etc/ssh/ssh_config`)

```
# Site-wide defaults for various options
Host *
ForwardAgent no
ForwardX11 no
RhostsAuthentication no
RhostsRSAAuthentication no
RSAAuthentication yes
PasswordAuthentication yes
FallBackToRsh no
UseRsh no
BatchMode no
CheckHostIP yes
StrictHostKeyChecking no
IdentityFile ~/.ssh/identity
Port 22
Cipher blowfish
EscapeChar

逐行说明
# Site-wide defaults for various options 带“#”表示该句为注释不起作，该句不属于配置文件原文，意在说明下面选项均为系统初始默认的选项。说明一下，实际配置文件中也有很多选项前面加有“#”注释，虽然表示不起作用，其实是说明此为系统默认的初始化设置。
"Host"只对匹配后面字串的计算机有效，“*”表示所有的计算机。从该项格式前置一些可以看出，这是一个类似于全局的选项，表示下面缩进的选项都适用于该设置，可以指定某计算机替换*号使下面选项只针对该算机器生效。
"ForwardAgent"设置连接是否经过验证代理（如果存在）转发给远程计算机。
"ForwardX11"设置X11连接是否被自动重定向到安全的通道和显示集（DISPLAY set）。
"RhostsAuthentication"设置是否使用基于rhosts的安全验证。
"RhostsRSAAuthentication"设置是否使用用RSA算法的基于rhosts的安全验证
"RSAAuthentication"设置是否使用RSA算法进行安全验证。
"PasswordAuthentication"设置是否使用口令验证。
"FallBackToRsh"设置如果用ssh连接出现错误是否自动使用rsh，由于rsh并不安全，所以此选项应当设置为"no"。
"UseRsh"设置是否在这台计算机上使用"rlogin/rsh"，原因同上，设为"no"。
"BatchMode"：批处理模式，一般设为"no"；如果设为"yes"，交互式输入口令的提示将被禁止，这个选项对脚本文件和批处理任务十分有用。
"CheckHostIP"设置ssh是否查看连接到服务器的主机的IP地址以防止DNS欺骗。建议设置为"yes"。
"StrictHostKeyChecking"如果设为"yes"，ssh将不会自动把计算机的密匙加入"$HOME/.ssh/known_hosts"文件，且一旦计算机的密匙发生了变化，就拒绝连接。
"IdentityFile"设置读取用户的RSA安全验证标识。
"Port"设置连接到远程主机的端口，ssh默认端口为22。
“Cipher”设置加密用的密钥，blowfish可以自己随意设置。
“EscapeChar”设置escape字符。
```

服务端配置(`/etc/ssh/sshd_config`)

```
# This is ssh server systemwide configuration file.
Port 22
ListenAddress 192.168.1.1
HostKey /etc/ssh/ssh_host_key
ServerKeyBits 1024
LoginGraceTime 600
KeyRegenerationInterval 3600
PermitRootLogin no
IgnoreRhosts yes
IgnoreUserKnownHosts yes
StrictModes yes
X11Forwarding no
PrintMotd yes
SyslogFacility AUTH
LogLevel INFO
RhostsAuthentication no
RhostsRSAAuthentication no
RSAAuthentication yes
PasswordAuthentication yes
PermitEmptyPasswords no
AllowUsers admin

 
逐行说明
"Port"设置sshd监听的端口号。
"ListenAddress”设置sshd服务器绑定的IP地址。
"HostKey”设置包含计算机私人密匙的文件。
"ServerKeyBits”定义服务器密匙的位数。
"LoginGraceTime”设置如果用户不能成功登录，在切断连接之前服务器需要等待的时间（以秒为单位）。
"KeyRegenerationInterval”设置在多少秒之后自动重新生成服务器的密匙（如果使用密匙）。重新生成密匙是为了防止用盗用的密匙解密被截获的信息
"PermitRootLogin”设置是否允许root通过ssh登录。这个选项从安全角度来讲应设成"no"。
"IgnoreRhosts”设置验证的时候是否使用“rhosts”和“shosts”文件。
"IgnoreUserKnownHosts”设置ssh daemon是否在进行RhostsRSAAuthentication安全验证的时候忽略用户的"$HOME/.ssh/known_hosts
"StrictModes”设置ssh在接收登录请求之前是否检查用户家目录和rhosts文件的权限和所有权。这通常是必要的，因为新手经常会把自己的目录和文件设成任何人都有写权限。
"X11Forwarding”设置是否允许X11转发。
"PrintMotd”设置sshd是否在用户登录的时候显示“/etc/motd”中的信息。
"SyslogFacility”设置在记录来自sshd的消息的时候，是否给出“facility code”。
"LogLevel”设置记录sshd日志消息的层次。INFO是一个好的选择。查看sshd的man帮助页，已获取更多的信息。
"RhostsAuthentication”设置只用rhosts或“/etc/hosts.equiv”进行安全验证是否已经足够了。
"RhostsRSA”设置是否允许用rhosts或“/etc/hosts.equiv”加上RSA进行安全验证。
"RSAAuthentication”设置是否允许只有RSA安全验证。
"PasswordAuthentication”设置是否允许口令验证。
"PermitEmptyPasswords”设置是否允许用口令为空的帐号登录。
"AllowUsers”的后面可以跟任意的数量的用户名的匹配串，这些字符串用空格隔开。主机名可以是域名或IP地址。    
```

通常情况下我们在连接 OpenSSH服务器的时候假如 UseDNS选项是打开的话，服务器会先根据客户端的 IP地址进行 DNS PTR反向查询出客户端的主机名，然后根据查询出的客户端主机名进行DNS正向A记录查询，并验证是否与原始 IP地址一致，通过此种措施来防止客户端欺骗。平时我们都是动态 IP不会有PTR记录，所以打开此选项也没有太多作用。我们可以通过关闭此功能来提高连接 OpenSSH 服务器的速度。

服务端配置如下

```shell
# 1. 编辑配置文件
vim /etc/ssh/sshd_config
# 2.修改如下选项
#UseDNS yes
UseDNS no
#GSSAPIAuthentication yes
GSSAPIAuthentication no
# 3.保存配置文件
# 4. 重启 OpenSSH服务器
/etc/init.d/sshd restart

# 注意：一般远程修改ssh端口，建议22端口留着防止修改未成功。如果开启防火墙记得添加端口放行！
port 22
port 234
```

# 设置固定ip

查找网卡名称

```shell
ifconfig
```

- 16版本

配置

```shell
# 配置ip
sudo vi /etc/network/interface

auto lo
iface lo inet loopback
auto ens32  # 网卡名称
iface ens32 inet static
address 192.168.159.130  # 固定ip地址
netmask 255.255.255.0
gateway 192.168.2.1

# 设置dns
sudo vi /etc/resolvconf/resolv.conf.d/base

nameserver 8.8.8.8
nameserver 8.8.4.4
```

刷新配置文件

```shell
sudo resolvconf -u
```

重启网络服务

```shell
sudo /etc/init.d/networking restart
```

- 18及20

配置

```shell
sudo vim /etc/netplan/01-network-manager-all.yaml

network:
  version: 2
  # render: NetworkManager
  ethernets:
    enp2s0:
      dhcp4: no
      dhcp6: no
      addresses: [192.168.13.177/24]  # 固定ip
      gateway4: 192.168.13.1
      nameservers:
        addresses: [223.5.5.5, 211.138.24.66]
```

应用

```shell
sudo netplan --debug apply
```

