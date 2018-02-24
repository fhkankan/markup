# Python安装

## Linux

已默认安装，检查安装版本

```
python --version
python3 --version
```

手动安装

```
1、使用deadsnakes
sudo add-apt-repository ppa:fkrll/deadsnakes
2、安装python
sudo apt-get update
sudo apt-get install python3.6
```

##OS x

已默认安装，检查安装版本

```
python --version
python3 --version
```

手动安装

```
1、安装Homebrew，其依赖apple的Xcode
xcode-select --install
在不断出现的确认对话框中单击ok
ruby -e "$(curl -fsSL homebrew安装网址)"
检查是否正确安装homebrew
brew doctor
2、安装python
brew install python3
```

## Windows

默认未安装，检查安装版本

```
python --version
python3 --version
```

手动安装

```
1、下载安装包
在python网站中下载需要安装的python安装程序
2、运行安装
在双击安装时，选择复选框add python to path
3、添加环境变量
若2完成后命令行验证ok即可。
若命令行无法启动python,则打开控制面板/系统与安全/系统/高级系统设置/环境变量，在path中添加python.exe的路径和同级Script目录

注意：若安装多个python版本，在各安装目录下修改python.exe名称为python2.exe/python3.exe。
```

#模块安装

## pip---包管理工具

Linux/OSX

```
# 检查
pip --version
# 安装
sudo apt-get install python-pip
sudo apt-get install python3-pip
# 更新
pip install -U pip
pip3 install --upgrade pip

# 注意：若出现pip: unsupported locale setting
export LC_ALL=C
```

Windows

```
# 检查
pip --version
# 安装
python get-pip.py
python3 get-pip.py

执行完成后，在python的安装目录下的Scripts子目录下，可以看到pip.exe、pip2.7.exe、pip2.exe等，这就表示pip安装成功了。
注意：要想能在命令行上直接运行pip程序，需要scripts这个目录加入到环境变量PATH中。
# 更新
python2 -m pip install -U pip
python3 -m pip install -U pip
```

使用

```
# 默认python版本下安装单个包
pip install somepackage

# 多种python环境下包安装
python2 －m pip install 包名
python3 －m pip install 包名
# 也可
pip2 install 包名
pip3 install 包名

安装特定版本的package，通过使用==, >=, <=, >, <来指定一个版本号。
pip install 'Markdown==2.0'
pip install 'Markdown<2.0'
pip install 'Markdown>2.0,<2.0.3

# 安装批量包
# 生成依赖包
$ pip freeze > requirements.txt
# 安装依赖包
pip install -r requirements.txt

# 卸载
pip uninstall SomePackage  

# 更新
pip install --upgrade SomePackage  

# 显示安装文件
pip show --files SomePackage 

# 显示已安装成功的包
pip list 

# 显示已过时的包
pip list --outdated


```

## ipython---交互环境

```
sudo pip install ipython

sudo pip3 install ipython
```

## virtualenv-虚环境

安装

```
# ubuntu/windows
# 默认python版本下
pip install virtualenv

# 不同python版本下安装
python2 －m pip install virtualenv
python3 －m pip install virtualenv
或
pip2 install virtualenv
pip3 install virtualenv

注意：安装一个即可，对系统里有多版本的Python，可以通过指定系统相应版本的python.exe路径来安装不同版本的Python虚拟环境
```

新建虚环境

```
# 默认python版本下
virtualenv env1

#ubuntu
创建python2虚拟环境：
virtualenv -p python2 虚拟环境名
创建python3虚拟环境：
virtualenv -p python3 虚拟环境名

# windows
# 创建python2虚环境
virtualenv -p C:\Python27\python.exe test2 
# 创建python3虚环境
virtualenv -p C:\Python36\python.exe test3 
```

基础操作

```
进入虚拟环境：首先命令行进入虚环境文件夹下的Scripts目录。

开启虚拟环境：activate/activate.bat
退出虚拟环境：deactivate/deactivate.bat

删除虚拟环境：删除文件夹即可
```

安装包

```
进入虚拟环境后，两种方法安装：
1、使用pip install XX命令来安装Python模块。有时官方安装模块时速度较慢，通过国内镜像来加速下载。

pip install -i
https://pypi.douban.com/simple [模块名] 【豆瓣源加速安装】

2、如果安装出现问题，可以进入http://www.lfd.uci.edu/~gohlke/pythonlibs下载相应的二进制文件直接进行安装。

pip install (下载的二进制安装文件路径）
```

## virtualenvwrapper-虚环境管理

```
用virtualenv创建的虚拟环境必须到指定文件夹的Scripts目录下才能利用activate激活，如果虚拟环境太多，每次启动就非常麻烦，这里可以使用virtualenvwrapper来解决这个问题。
```

安装

```
# Linux
# 默认python版本下安装
pip install virtualenvwrapper
# 不同python版本下安装
python2 －m pip install virtualenvwrapper
python3 －m pip install virtualenvwrapper
# 或
pip2 install virtualenvwrapper
pip3 install virtualenvwrapper

# Windows
# 默认python版本下安装
pip install virtualenvwrapper-win
# 不同python版本下安装
python2 －m pip install virtualenvwrapper-win
python3 －m pip install virtualenvwrapper-win
# 或
pip2 install virtualenvwrapper-win
pip3 install virtualenvwrapper-win

注意：安装一个即可，对系统里有多版本的Python，可以通过指定系统相应版本的python.exe路径来安装不同版本的Python虚拟环境
```

配置环境变量

```
# Linux
1、创建目录用来存放虚拟环境
mkdir $HOME/.virtualenvs
2、编辑主目录下面的.bashrc文件，添加下面两行。
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
3、使用以下命令使配置立即生效
source .bashrc

# Windows
进入系统属性设置系统变量，添加WORKON_HOME环境变量到你指定的文件夹，不设置的话创建的虚拟环境文件夹会放到C盘用户目录下的Envs文件夹下，这里新建的文件夹。
```

新建虚拟环境

```
# 默认python版本环境下
mkvirtualenv env1

# Linux
创建python2虚拟环境：
mkvirtualenv -p python2 虚拟环境名
创建python3虚拟环境：
mkvirtualenv -p python3 虚拟环境名

# windows:
创建python2虚拟环境
mkvirtualenv -p c:\python27\python.exe 虚拟环境名
创建python3虚拟环境
mkvirtualenv --python=c:\python36\python.exe 虚拟环境名
```

基础操作

```
查看所有通过mkvirtualenv创建的虚拟环境----- workon

进入虚拟环境----workon 文件名

退出虚拟环境----deactivate

删除虚拟环境----rmvirtualenv 文件名
```

## Django--web框架

1、安装

```
pip install django[==1.8.2]  //指定版本
```

一个Django项目（project），包含了多个应用(app)，一个应用代表一个功能模块，一个应用就是一个Python包

2、创建Django项目的两种方式：

命令行创建

```
# 进出入虚环境
workon 虚拟环境名
# 创建Django项目
django-admin startproject 项目名
# 创建应用，进入项目目录下
python manage.py startapp 应用名
# 配置应用与项目之间的关联文件
```

pycharm创建

3、运行django项目

开发调试阶段，可以使用django自带的服务器。有两种方式运行服务器

命令行运行

```
# 启动django服务器：
python manage.py runserver

默认主机和端口号为:127.0.0.1:8000，如果想指定主机和端口号，可以类似如下命令指定：
python manage.py runserver 192.168.210.137:8001

# 打开浏览器，访问服务器：http://127.0.0.1:8000

如果增加、修改、删除python文件，服务器会自动重启
可以按ctrl + C停止服务器
```

pycharm运行



## Flask---web框架

安装Flask

```
# 创建虚环境
mkvirtualenv flask_py2	
workon flask_py2

# 指定Flask版本安装
$ pip install flask==0.10.1

# Mac系统：
$ easy_install flask==0.10.1
```

安装依赖包

```
# 安装依赖包（须在虚拟环境中）
$ pip install -r requirements.txt

# 生成依赖包（须在虚拟环境中）
$ pip freeze > requirements.txt

# 在ipython中测试安装是否成功
$ from flask import Flask
```









