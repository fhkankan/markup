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

在windows下安装python某些包时出错，可在https://www.lfd.uci.edu/~gohlke/pythonlibs/下寻找windows版本

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

```python
# 多种python环境下包安装
python2 －m pip install 包名
python3 －m pip install 包名
# 也可
pip2 install 包名
pip3 install 包名

# 默认python版本下安装单个包
pip install somepackage

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

```python
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

```python
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

## pymysql

安装

```python
# 安装
pip install pymysql
```

配置

```python
# 在__init__.py中
import pymysql
pymsql.install_as_MySQLdb()
```

## mysql-python

ubuntu

```
# ubuntu下安装简单
pip install MySQL-python
```

windows

```
pip install mysql-python
```

会报异常

```
error: Microsoft Visual C++ 9.0 is required (Unable to find vcvarsall.bat).
Get it from http://aka.ms/vcpython27
```

解决方法

```
1. 在http://www.lfd.uci.edu/~gohlke/pythonlibs/#mysql-python下载对应的包版本:
若是32位2.7版本的python，则下载MySQL_python‑1.2.5‑cp27‑none‑win32.whl
若是64位2.7版本的python，就下载MySQL_python-1.2.5-cp27-none-win_amd64.whl

2.在cmd下跳转到下载MySQL_python-1.2.5-cp27-none-win_amd64.whl的目录下,然后在命令行执行pip install MySQL_python-1.2.5-cp27-none-win_amd64.whl
```

# Anaconda

## 安装

```python
1. 安装
# windows
在官方地址下载安装包
# Ubuntu
进入下载目录，打开终端，输入
bash Anaconda3-4.3.1-Linux-x86_64.sh 

2. 环境变量
# windows
在系统环境变量path处添加安装路径
C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts
# Ubuntu
在 ~/.bashrc中添加anaconda的bin目录加入PATH
echo 'export PATH="~/anaconda2/bin:$PATH"' >> ~/.bashrc
# 更新bashrc以立即生效
source ~/.bashrc

3. 检查
conda -v
```

## 包管理

```python
# 检查更新当前conda
conda update conda 

# 查看安装了哪些包
conda list 

# 对所有工具包进行升级
conda upgrade --all

# 安装一个包
conda install package_name

# 指定安装的版本
conda install package_name=x.x

# 升级 package 版本
conda update package_name

# 移除一个 package
conda remove package_name

# 模糊查询
conda  search search_term

# 将当前环境下的 package 信息存入名为 environment 的 YAML 文件中
conda env export > environment.yaml

# 用对方分享的 YAML 文件来创建一摸一样的运行环境
conda env create -f environment.yaml
```

## 虚环境

```python
# 查看当前存在哪些虚拟环境
conda env list 或 conda info -e

# 创建python虚拟环境,在Anaconda安装目录envs下
conda create -n your_env_name python=X.X 

# 检查当前python的版本
python --version

# 激活虚拟环境
# Linux 
source activate your_env_name
# Windows
activate your_env_name

# 对虚拟环境中安装额外的包
conda install -n your_env_name [package]

# 关闭虚拟环境
# Linux
source deactivate
# Windows
deactivate

# 删除虚拟环境
conda remove -n your_env_name --all

# 删除环境中的某个包
conda remove --name your_env_name package_name
```







​    

   





