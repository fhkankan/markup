[TOC]

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

修改python指向

```
# 进入python命令所在目录
cd /usr/bin
# 产看安装的python版本信息
ls -l python*
# 更改软连接指向
sudo rm python
ln -s python3 python
```

## OS x

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



## pip

- 安装

Linux

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

mac

```
brew install pip
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

- 更换国内源

pip国内的一些镜像
```
阿里云 http://mirrors.aliyun.com/pypi/simple/ 
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/ 
豆瓣(douban) http://pypi.douban.com/simple/ 
清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/ 
中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/
```
修改源方法

```shell
# 临时使用
pip install scrapy -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host
pip install scrapy -i http://mirrors.aliyun.com/pypi/simple/  --trusted-host mirrors.aliyun.com

# 永久修改： 
# linux/mac：修改 ~/.pip/pip.conf (没有就创建一个)
# windows：直接在user目录中创建一个pip目录，如：C:\Users\xx\pip，新建文件pip.ini
[global]
timeout = 6000  # 可以不写
index-url = https://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com
```

- 使用

```python
# 多种python环境下包安装
python2 －m pip install 包名
python3 －m pip install 包名
# 也可
pip2 install 包名
pip3 install 包名

# 默认python版本下安装单个包
# 安装特定版本的package，通过使用==, >=, <=, >, <来指定一个版本号。
pip install 'Markdown==2.0'
pip install 'Markdown<2.0'
pip install 'Markdown>2.0,<2.0.3

# 安装批量包
pip freeze > requirements.txt  # 生成依赖包
pip install -r requirements.txt  # 安装依赖包

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

# 解决依赖过多：This is taking longer than usual. You might need to provide the dependency
--use-deprecated=legacy-resolver
```

## virtualenv

依赖系统python环境，基于固定系统文件夹创建python环境

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

## virtualenvwrapper

虚环境管理

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
mkdir $HOME/.envs
2、编辑主目录下面的.bashrc文件，添加下面两行。
export WORKON_HOME=$HOME/.envs
source /usr/local/bin/virtualenvwrapper.sh
3、使用以下命令使配置立即生效
source .bashrc

# Windows
进入系统属性设置系统变量，添加WORKON_HOME环境变量到你指定的文件夹，不设置的话创建的虚拟环境文件夹会放到C盘用户目录下的Envs文件夹下，这里新建的文件夹。
```

新建虚拟环境

```python
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

## venv

依赖系统python环境，从python3.3开始，标准库支持创建虚拟环境

```
python3 -m venv /path/to/new/virtual/environment
```

全命令

```shell
venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear]
            [--upgrade] [--without-pip] [--prompt PROMPT]
            ENV_DIR [ENV_DIR ...]

# 在一个或多个目标目录中创建虚拟Python环境。

位置参数:
  ENV_DIR              虚拟环境安装目录

可选参数:
  -h, --help            展示帮助信息并推出
  --system-site-packages
                        授予虚拟环境访问系统的权限site-packages目录。
  --symlinks            当符号链接不是该平台的默认设置时，尝试使用符号链接而不是复制。
  --copies              即使符号链接是该平台的默认设置时，尝试使用复制而不是符号链接。
  --clear               如果在创建环境之前目录已经存在，删除环境目录的内容。
  --upgrade             升级环境目录以使用此版本的Python，假设Python已就地升级
  --without-pip        	跳过在虚拟环境中安装或升级pip环境（默认情况下，pip自举）
  --prompt PROMPT       为此环境提供备用提示前缀。

# 创建环境后，您可能希望激活它，例如通过在其bin目录中找到一个激活脚本。
```

## pyenv

不依赖于系统环境，在系统中安装多版本python环境

- 安装

> mac

安装

```
brew install pyenv
```

环境变量

```
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
```

激活

```
source ~/.zshrc
```

> ubuntu

克隆pyenv仓库

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

加入环境变量(.bashrc或.zshrc)

```
# 方式一：终端
echo 'export PATH=~/.pyenv/bin:$PATH' >> ~/.bashrc
echo 'export PYENV_ROOT=~/.pyenv' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# 方式二：编辑器
export PATH=~/.pyenv/bin:$PATH
export PYENV_ROOT=~/.pyenv
eval "$(pyenv init -)"
```

激活pyenv

```
source ~/.bashrc
```

- 删除

删除`.pyenv`文件夹和修改`.bashrc`文件中添加部分

- 常用命令

查看

```python
pyenv version	# 显示当前活动的Python版本以及有关如何设置的信息
pyenv versions  # 列出pyenv已知的所有Python版本，并在当前活动版本旁显示一个星号
pyenv which python # 显示pyenv在运行给定命令时将调用的可执行文件的完整路径
pyenv whence 2to3 # 列出安装了给定命令的所有Python版本
```

安装

```python
pyenv install --list # 列出所有可用的Python版本，包括Anaconda，Jython，pypy和stackless
pyenv install <version> # 安装对应版本
pyenv install -v <version> # 安装对应版本，若发生错误，可以显示详细的错误信息
pyenv uninstall <version> # 卸载对应版本
```

设置

```python
# global
# 通过将版本名称写入~/.pyenv/version文件来设置要在所有shell中使用的Python的全局版本。该版本可以被特定于应用程序的.python-version文件覆盖，也可以通过设置PYENV_VERSION环境变量来覆盖。
pyenv global <version> # 告诉全局环境使用某个版本，为了不破坏系统环境，不建议使用
pyenv global <version1> <version2>  # 指定多个python版本作为全局python，后者更优先
pyenv global system   #全局进行切换到系统自带python

# local
# 通过将版本名称写入.python-version当前目录中的文件来设置本地特定于应用程序的Python版本。该版本覆盖全局版本，并且可以通过设置PYENV_VERSION环境变量或pyenv shell 命令来覆盖自身。
pyenv local <version> # 当前路径创建一个.python-version, 以后进入这个目录自动切换为该版本
pyenv local <version1> <version1>  # 指定多个python版本作为本地python，后者更优先
pyenv local --unset  # 取消设置本地版本
pyenv local system	  #只针对当前目录及其子目录切换到系统自带python

# shell
# 通过PYENV_VERSION 在shell中设置环境变量来设置特定于shell的Python版本。此版本覆盖特定于应用程序的版本和全局版本。
pyenv shell <version> # 当前shell的session中启用某版本，优先级高于global 及 local
pyenv shell <version1> <version1>  # 指定多个python版本作为shell的python，后者更优先
pyenv shell --unset	# 取消设置外壳版本

# rehash
# 为pyenv（即，~/.pyenv/versions/*/bin/*）已知的所有Python二进制文件安装填充程序 。在安装新版本的Python之后运行此命令，或安装提供二进制文件
pyenv rehash          # 重建环境变量，增删Python版本或带有可执行文件的包（如pip）以后
```

> 虚环境virtualenv

安装

```
# mac
brew install pyenv-vitualenv

# ubuntu
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

使用

```python
# 虚环境的真实目录位于：~/.pyenv/versions/
# 指定的python必须是一个安装前面步骤已经安装好的python版本， 否则会出错
# 安装虚环境
pyenv virtualenv env # 从默认版本创建虚拟环境
pyenv virtualenv 3.6.4 env364 # 创建3.6.4版本的虚拟环境
# 激活
pyenv activate env364 # 激活env364这个虚拟环境
# 退出
pyenv deactivate # 停用当前的虚拟环境
# 删除
pyenv virtualenv-delete env364

# 自动激活
# 使用pyenv local 虚拟环境名
# 会把`虚拟环境名`写入当前目录的.python-version文件中
# 关闭自动激活 -> pyenv deactivate
# 启动自动激活 -> pyenv activate env364
pyenv local env364
pyenv uninstall env364 # 删除 env364 这个虚拟环境
```

## pipenv

依赖系统中python环境，基于项目文件夹创建python环境

集成pip与virtualenv，替代virtualenv和pyenv

- 安装

将pipenv安装在用户主目录下

```
pip install pipenv --user [username]
```

- 虚环境

创建虚环境

```shell
cd projectPath

pipenv --two		# 使用系统中的python2创建虚环境
pipenv --three		# 使用系统中的python3创建虚环境
pipenv --python3.6	# 创建特定版本的虚环境
pipenv install		# 若是项目中无配置文件，创建一个默认虚环境


# 介绍
pipenv install 是安装已经提供的包并将它们加入到Pipfile中，同时创建了项目的虚拟环境
Pipfile是python包依赖文件，列出了项目中所有包的依赖，这是pipenv相当大的创新，对应的是Pipfile.lock文件，两者构成虚拟环境的管理工作。

pipenv install的时候有三种逻辑：
- 如果目录下没有Pipfile和Pipfile.lock文件，表示创建一个新的虚拟环境；
- 如果有，表示使用已有的Pipfile和Pipfile.lock文件中的配置创建一个虚拟环境；
- 如果后面带诸如django这一类库名，表示为当前虚拟环境安装第三方库。
```
激活虚环境

```
pipenv shell
```

退出虚环境

```
exit
```

pipenv虚拟环境运行python命令

```
pipenv run python your_script.py
```

- 安装卸载第三方库

```
cd projectPath
pipenv install django	# 安装第三方最新库
pipenv install django==2.0.5	# 安装特定版本第三方库
pipenv uninstall django	# 卸载第三方库
```

- 开发环境管理 

Pipenv使用--dev标志区分两个环境(开发和生产)，实现在同一个虚环境中管理两种开发环境

```
pipenv install --dev django
```

项目克隆其他位置后，可使用如下命令分别安装相关依赖

```
pipenv install		# 安装生产环境依赖
pipenv install --dev	# 安装开发环境依赖
```

- 其他命令

```
$ pipenv
Usage: pipenv [OPTIONS] COMMAND [ARGS]...

Options:
  --update         更新Pipenv & pip
  --where          显示项目文件所在路径
  --venv           显示虚拟环境实际文件所在路径
  --py             显示虚拟环境Python解释器所在路径
  --envs           显示虚拟环境的选项变量
  --rm             删除虚拟环境
  --bare           最小化输出
  --completion     完整输出
  --man            显示帮助页面
  --three / --two  使用Python 3/2创建虚拟环境（注意本机已安装的Python版本）
  --python TEXT    指定某个Python版本作为虚拟环境的安装源
  --site-packages  附带安装原Python解释器中的第三方库
  --jumbotron      不知道啥玩意....
  --version        版本信息
  -h, --help       帮助信息
  
Commands:
  check      检查安全漏洞
  graph      显示当前依赖关系图信息
  install    安装虚拟环境或者第三方库
  lock       锁定并生成Pipfile.lock文件
  open       在编辑器中查看一个库
  run        在虚拟环境中运行命令
  shell      进入虚拟环境
  uninstall  卸载一个库
  update     卸载当前所有的包，并安装它们的最新版本
```

## poetry

[官方文档](https://python-poetry.org/docs/)

- 安装配置

安装

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 -
# 下载文件至本地，使用python3 文件名.py
```

配置

```
1. 拷贝
按照提示，将类似`export PATH="/Users/henry/.local/bin:$PATH"`命令拷贝至.bashrc/.zshrc中
2. 生效
source .zshrc
```

检查

```
poetry --version
```

卸载

```
python install-poetry.py --uninstall
```

- 初始化

```shell
# 创建项目及poetry相关
poetry new my-package
poetry new my-folder --name my-package

# 已有项目添加poetry管理
cd 项目名
poetry init
```

- 虚环境

创建虚拟环境

```shell
# 1.利用virtualenvs.create=true自动创建
# 当参数 virtualenvs.create=true 时，执行 poetry install 或 poetry add 时会检测当前项目是否有虚拟环境，没有就自动创建。
poetry install 
poetry add 

# 2.利用poetry env use创建
# 会在虚拟路径下创建一个envs.toml文件，存储虚拟环境指定Python解释器的版本
poetry env use <python>  # 对当前项目激活/创建一个虚环境
```

激活虚拟环境

```shell
# 显示激活的虚拟环境
poetry shell

# 执行poetry的命令并不需要激活虚拟环境，因为poetry会自动检测当前虚拟环境
poetry run <命令>

# 查看python版本
poetry run python -V
```

查看当前项目的虚环境

```shell
poetry env list  # 与当前项目有关的所有虚环境
poetry env list --full-path  # 显示绝对路径
poetry env info [-p]   # 展示当前虚环境的信息
```

删除虚环境

```shell
poetry env remove <python>  # 删除项目中特定的虚环境
```

- 配置

概述

```shell
# 可配置参数
cache-dir  # Poetry 使用的缓存目录的路径。
installer.parallel  # 使用新的安装程序时并行执行，默认true
virtualenvs.create  # 如果不存在则创建一个新的虚环境，默认true
virtualenvs.in-project  # 在项目的根目录中创建virtualenv，默认None
virtualenvs.path # 虚环境被创建的目录位置，默认`{cache-dir}/virtualenvs`
repositories.<name>  # 设置一个新的替代存储库

# 可配置方法
1.命令行
2.直接修改config.toml（不建议）
```

设置

```shell
# 本地参数配置
poetry config virtualenvs.create false --local
# 执行后，当前项目在执行poetry install/poetry add时如果没有虚拟环境的话就会直接安装到系统路径上。执行后，会在当前项目生成poetry.toml文件，不影响全局配置config.toml

# 全局配置
poetry config virtualenvs.create false
poetry config cache-dir E:\Documents\Library  
# 修改缓存目录，同时也修改了虚拟环境目录
```

删除

```shell
poetry config virtualenvs.create --unset  # 重置全局配置
poetry config virtualenvs.create --local --unset  # 重置项目配置
```

查看

```shell
poetry config --list
# 列出配置时，包括了全局和本地的配置，本地的配置会覆盖全局的参数

poetry config virtualenvs.path  # 直接展示特定配置
```

- 安装依赖

添加

```shell
# 方法一：修改pyproject.toml
[tool.poetry.dependencies]
pendulum = "^1.4"
my-package = {path = "../my/path", develop = true}

# 方法二：使用命令
poetry add pendulum  # 安装自适应最新
poetry add pendulum@^2.0.5  # 安装固定版本
poetry add "pendulum>=2.0.5" 
poetry add pendulum --dev  # 安装开发依赖

poetry add git+https://github.com/sdispater/pendulum.git
poetry add git+ssh://git@github.com/sdispater/pendulum.git
poetry add git+https://github.com/sdispater/pendulum.git#develop
poetry add git+https://github.com/sdispater/pendulum.git#2.0.5
poetry add ./my-package/
poetry add ../my-package/dist/my-package-0.1.0.tar.gz
poetry add ../my-package/dist/my_package-0.1.0.whl
```

安装`pyproject.toml/pyproject.lock`文件中依赖

```shell
poetry install  # 默认安装
poetry install --no-dev # 不安装开发依赖项
poetry install --no-root  # 不安装root包
poetry install --extras "mysql pgsql"  # 安装特定
poetry install -E mysql -E pgsql
poetry install --remove-untracked  # 删除锁定文件中不再存在的旧依赖项
```

- 删除

```shell
poetry remove pendulum  # 删除依赖
```

- 更新

```shell
poetry self update  # 更新poetry本身
poetry update  # 更新所有依赖版本和poetry.lock
poetry update requests toml  # 更新特定的包
```

- 查看

```shell
poetry show	# 展示所有包
poerty show --no-dev  # 不展示dev
poetry show --tree	# 以树形展示
poetry show --latest  # 展示最新版本
poetry show --outdated  # 展示过时包的最新版本

poetry show pedulum  # 查看包的详细信息
```

- 锁定

```shell
poetry lock  # 将所有依赖项锁定到最新的可用兼容版本。
poetry lock --no-update  # 要仅刷新锁定文件
```

- 导出

```shell
# 将锁定文件导出为其他格式
poetry export -f requirements.txt --output requirements.txt
```

- 发布

```shell
poetry build  # 打包
poetry publish  # 发布到pypi
poetry publish -r my-repository  # 发布到私有仓库
```

## ipython

```
sudo pip install ipython

sudo pip3 install ipython
```

帮助

```python
# 获取文档
help(len)
len?

# 获取源码
square??
```

魔法命令

```shell
%paste/%cpaste  # 粘贴代码块
%run 			# 执行外部脚本

%history		# 可查看历史输入
%magic			# 获取可用魔法函数的通用描述及示例
%lsmagic		# 获取所有可用魔法函数的列表
```

输出

```python
print(_)		# 更新以前的输出
print(__)		# 获得倒数第二个历史输出
print(_2)		# 等价于Out[2]
2+2;			# ;会禁止输出
```

性能

```shell
%time			# 运行一行代码执行时间
%timeit			# 重复运行计算接下来一行的语句执行时间
%%timeit		# 处理多行输入的执行时间
%prun			# 统计代码执行过程耗时细节
%load_ext line_profiler
%lprun -f func_name()  # 对代码生成逐行分析报告
%load_ext memory_profiler
%memit func_name()  # 内存消耗计算类似%timeit
%mprun func_name()  # 内存消耗计算类似%lprun
```

## jupyter

安装

```
pip install jupyter
```

使用

```
$jupyter notebook
```

# Anaconda

[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

## 安装

有anaconda和miniconda可供选择

```python
1. 安装
# windows
在官方地址下载安装包
# Ubuntu
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh 

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh

# mac
brew cask install anaconda
brew cask install miniconda

2. 环境变量
# windows
在系统环境变量path处添加安装路径
C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Scripts
# Ubuntu
会自动在.bashrc文件中生成配置项，若是没有自动生成，手动配置
export PATH="/home/Sweeneys/anaconda3/bin:$PATH"
# 更新bashrc以立即生效
source ~/.bashrc
# mac 
conda init zsh/bash

# 取消默认启用conda环境
conda config --set auto_activate_base false

3. 检查
conda -V
```

## 修改源

```python
# 编辑~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
auto_activate_base: false
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
conda search search_term

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
# mac
conda activate your_env_name

# 对虚拟环境中安装额外的包
conda install -n your_env_name [package]

# 关闭虚拟环境
# Linux
source deactivate
# Windows
deactivate
# mac
conda deactivate

# 删除虚拟环境
conda remove -n your_env_name --all

# 删除环境中的某个包
conda remove --name your_env_name package_name

# 虚环境改名字
1.修改env中的文件夹名字
2.修改env中文件夹中bin/pip和bin/pip3中的路径名字
3.修改.conda/environments.txt中的文件名映射
```







​    

   





