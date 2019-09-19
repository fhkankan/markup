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
sogou
opera
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

