# 编译

## Linux

### 官方下载

```
# 卸载旧版本
sudo apt-get remove --purge vim 
sudo apt-get remove --purge vim-gtk vim-doc cscope
sudo apt-get clean


# 安装
sudo add-apt-repository ppa:jonathonf/vim
sudo apt update
sudo apt install vim
已编译，支持python3

# 卸载
sudo apt remove vim
sudo add-apt-repository ppa:jonathonf/vim
```

### 自编译

1、安装依赖项

注意，在Ubuntu16.04中Lua应该为liblua5.1-dev，而在其它版本中应为lua5.1-dev

```
sudo apt-get install libncurses5-dev libgnome2-dev libgnomeui-dev \
    libgtk2.0-dev libatk1.0-dev libbonoboui2-dev \
    libcairo2-dev libx11-dev libxpm-dev libxt-dev python-dev \
    python3-dev ruby-dev liblua5.1 lua5.1-dev libperl-dev git
```

2、从github下载vim

```
新建一个文件夹存放clone下来的vim，然后在该文件夹下打开终端执行命令

git clone https://github.com/vim/vim.git
3.查询系统中是否已经含有vim，如果有的话删除系统中vim
```

3、查询系统中是否已经含有vim，如果有的话删除系统中vim

```
dpkg -l | grep vim
sudo apt-get remove vim-common vim-tiny
```

4、编译下好的Vim,安装

```
cd vim
./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-pythoninterp=yes \
            --with-python-config-dir=/usr/lib/python2.7/config \
            --enable-python3interp=yes \
            --with-python3-config-dir=/usr/lib/python3.5/config \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
            --enable-gui=gtk2 --enable-cscope --prefix=/usr
make VIMRUNTIMEDIR=/usr/share/vim/vim80 
sudo make install
```

5、查看编译是否成功

```
vim --version
:echo has('python') || has('python3')
```

##Windows

### 官方下载

```
https://github.com/vim/vim-win32-installer
下载可用包即可（已编译）
```

### 自编译

一 、编译环境选择和安装

    1. 选择编译环境

从网上查看Win上的编译环境 Cygwin、MinGW、MSYS2等，我选了MSYS2作为编译环境，而且MSYS2用Pacman进行包管理 ，用过ArchLinux的比较方便不少。

2. 安装MSYS2编译环境

3. 下载安装[msys2](https://link.jianshu.com?t=http://sourceforge.net/projects/msys2/), 默认文件目录为：`C:\msys64`, 编译环境为以此目录为根目录。
4. 修改`/etc/pacman.d/`目录下3个文件`mirrorlist.mingw32, mirrorlist.mingw64, mirrorlist.msys`, 添加软件源提升下载速度：
   - 添加`Server = http://mirrors.ustc.edu.cn/msys2/REPOS/MINGW/i686` 到`mirrorlist.mingw32`文件首行。
   - 添加`Server = http://mirrors.ustc.edu.cn/msys2/REPOS/MINGW/x86_64` 到`mirrorlist.mingw64`文件首行。
   - 添加`Server = http://mirrors.ustc.edu.cn/msys2/REPOS/MSYS2/$arch` 到`mirrorlist.msys`文件首行。
   - 运行 `pacman -Syu` 更新
5. 安装编译器,调试器,make工具, git `pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gdb mingw-w64-x86_64-make git`
6. 提前安装 `pacman -S diffutils` ，此包包含 diff 命令，因我在编译过程碰到没有 diff 命令。

二、 编译Vim

1. 编译前准备软件安装

- 下载 [Python2.7.9(x64)](https://link.jianshu.com?t=https://www.python.org/ftp/python/2.7.9/python-2.7.9.amd64.msi) 安装到 `C:\Python27`
- 下载 [Python3.4.3(x64)](https://link.jianshu.com?t=https://www.python.org/ftp/python/3.4.3/python-3.4.3.amd64.msi) 安装到 `C:\Python34`
- 下载 [Ruby 2.2.3 (x64)](https://link.jianshu.com?t=http://dl.bintray.com/oneclick/rubyinstaller/rubyinstaller-2.2.3-x64.exe) 安装到 `C:\Ruby22-x64`
- 下载[lua-5.3_Win64_bin.zip](https://link.jianshu.com?t=http://sourceforge.net/projects/luabinaries/files/5.3/Executables/lua-5.3_Win64_bin.zip/download) 和 [lua-5.3_Win64_dllw4_lib.zip](https://link.jianshu.com?t=http://sourceforge.net/projects/luabinaries/files/5.3/Windows%20Libraries/Dynamic/lua-5.3_Win64_dllw4_lib.zip/download)，解压2个包到 `C:\Lua53`
- 编译电脑系统为Win7 64位 旗舰版
- 下载Vim源码

```
cd ~
git clone https://github.com/tracyone/vim vim-master
```

2. 设置编译参数文件

- 打开64位编译环境 `mingw64_shell`
- 进入src文件 `cd ~/vim-master/src`
- 复制编译文件 `Make_cyg_ming.mak` 为`custom.mak`, 修改`custom.mak`文件,定制自己要编译的选项.
- 修改编译64位vim

```
# FEATURES=[TINY | SMALL | NORMAL | BIG | HUGE]
# Set to TINY to make minimal version (few features).
FEATURES=HUGE
# Set to one of i386, i486, i586, i686 as the minimum target processor.
# For amd64/x64 architecture set ARCH=x86-64 .
ARCH=x86-64
```

- 修改编译Lua

```
# ifdef LUA
ifndef LUA
LUA=c:/Lua53
endif

ifndef DYNAMIC_LUA
DYNAMIC_LUA=yes
endif

ifndef LUA_VER
LUA_VER=53
endif

ifeq (no,$(DYNAMIC_LUA))
LUA_LIB = -L$(LUA)/lib -llua
endif

# endif

```

- 修改编译Python

```
# ifdef PYTHON

ifndef PYTHON
PYTHON=c:/Python27
endif

ifndef DYNAMIC_PYTHON
DYNAMIC_PYTHON=yes
endif

ifndef PYTHON_VER
PYTHON_VER=27
endif

ifeq (no,$(DYNAMIC_PYTHON))
PYTHONLIB=-L$(PYTHON)/libs -lpython$(PYTHON_VER)
endif
# my include files are in 'win32inc' on Linux, and 'include' in the standard
# NT distro (ActiveState)
ifeq ($(CROSS),no)
PYTHONINC=-I $(PYTHON)/include
else
PYTHONINC=-I $(PYTHON)/win32inc
endif
# endif
```

- 修改编译Python3

```
# ifdef PYTHON3
ifndef PYTHON3
PYTHON3=c:/Python34
endif

ifndef DYNAMIC_PYTHON3
DYNAMIC_PYTHON3=yes
endif

ifndef PYTHON3_VER
PYTHON3_VER=34
endif

ifeq (no,$(DYNAMIC_PYTHON3))
PYTHON3LIB=-L$(PYTHON3)/libs -lPYTHON$(PYTHON3_VER)
endif

ifeq ($(CROSS),no)
PYTHON3INC=-I $(PYTHON3)/include
else
PYTHON3INC=-I $(PYTHON3)/win32inc
endif
# endif
```

- 修改编译Ruby

```
# ifdef RUBY
ifndef RUBY
RUBY=c:/Ruby22-x64
endif
ifndef DYNAMIC_RUBY
DYNAMIC_RUBY=yes
endif
#  Set default value
ifndef RUBY_VER
RUBY_VER = 22
endif
ifndef RUBY_VER_LONG
RUBY_VER_LONG = 2.2.0
endif
ifndef RUBY_API_VER
RUBY_API_VER = $(subst .,,$(RUBY_VER_LONG))
endif

... 中间一大段默认不用设置

# endif # RUBY
```

- 这里编译和Linux下稍有不同,不需要 configue, 直接make 就可以了

  3. 开始编译

- 开始编译运行 `mingw32-make.exe -f custom.mak`
- 若先前有编译失败,或要重新编译,需要先运行 `mingw32-make.exe -f custom.mak clean`
- 成功后在 src 文件夹下就有 Gvim.exe 文件了,可以运行看看正常不.
- 需要把 `C:\Lua53\lua53.dll` 文件复制到和Gvim同一文件夹下

4. 整理打包编译好的Vim

我查了不少资料，但没有查到相关的，就是编译好后怎么整理编译好的文件到vim包，最后我直接参考 [tracyone](https://link.jianshu.com?t=http://www.cnblogs.com/tracyone/) 的编译脚本，打包了一个vim包，可以正常运行。

```
cd ~/vim-master
mkdir -p vim74-x64/vim74
cp -a runtime/* vim74-x64/vim74
cp -a src/*.exe vim74-x64/vim74
cp -a src/GvimExt/gvimext.dll vim74-x64/vim74
cp -a src/xxd/xxd.exe vim74-x64/vim74
cp -a vimtutor.bat vim74-x64/vim74
cp -a /c/Lua53/lua53.dll vim74-x64/vim74

```

三 安装Vim74-x64

运行管理员模式命令行，进行 Vim74-x64 文件夹，运行 install.exe
选择 d 进行默认安装。

参考资料

- [win7上编译安装64位VIM](https://www.jianshu.com/p/85739296bdc5)

- 编译成功的[Vim74 64位包](https://link.jianshu.com?t=http://pan.baidu.com/s/1nusRjfV)
- [Windows下编译YouCompleteMe流程](https://link.jianshu.com?t=http://www.cnblogs.com/tracyone/p/4735411.html)



# 配置

## 帮助文档

1. 下载帮助文档
2. 安装到vim的安装位置同级目录下的vimfiles文件中，不会覆盖原英文文档
3. 更换语言版本，在末行模式下，输入set helplang=en（或cn）

## ctags

1. 下载

   [源码](https://github.com/universal-ctags/ctags)

   [windows安装包](https://github.com/universal-ctags/ctags-win32/releases)

2. 安装

```
Linux：
sudo apt-get install ctags

windows:
直接解压放于安装位置，在path中设置环境变量
```

3. 使用

```
在源码目录下执行 　　ctags –R * 　　
“-R”表示递归创建，也就包括源代码根目录（当前目录）下的所有子目录。
“*”表示所有文 件。这条命令会在当前目录下产生一个“tags”文件，当用户在当前目录中运行vim时，会自动载入此tags文件。
Tags文件中包括这些对象的列表： 用#define定义的宏枚举型变量的值函数的定义、原型和声明名字空间（namespace）类型定义（typedefs）变量（包括定义和声明）类 （class）、结构（struct）、枚举类型（enum）和联合（union）类、结构和联合中成员变量或函数VIM用这个“tags”文件来定位上 面这些做了标记的对象。
:ts			标签列表
:tp			上一个标签
:tn			下一个标签
ctr+]		跳进定义源
ctr+t		返回跳转	
```

## 配置文件

位置

```
~/.vimrc 	Linux中配置文件
~/_vimrc		windows中配置文件
```

加载配置文件

```
重启vim
在normal执行:source %
```

编辑配置文件

```
vim ~/.vimrc	Linux中编辑配置文件
```

# 插件

## Vundle

插件管理工具

windows:

1、安装vundle前需要先安装git

安装完成，打开cmd 命令提示符，运行命令( git –version )检查git 版本号

2、配置curl

只需要在git的cmd目录创建文件curl.cmd即可

```
@rem Do not use "echo off" to not affect any child calls.
@setlocal
 
@rem Get the abolute path to the parent directory, which is assumed to be the
@rem Git installation root.
@for /F "delims=" %%I in ("%~dp0..") do @set git_install_root=%%~fI
@set PATH=%git_install_root%\bin;%git_install_root%\mingw\bin;%git_install_root%\mingw64\bin;%PATH%
@rem !!!!!!! For 64bit msysgit, replace 'mingw' above with 'mingw64' !!!!!!!
 
@if not exist "%HOME%" @set HOME=%HOMEDRIVE%%HOMEPATH%
@if not exist "%HOME%" @set HOME=%USERPROFILE%
 
@curl.exe %*
```

打开cmd 命令提示符，运行命令（ curl –version ）检查curl 版本号

3、安装vundle

```
# 在Vim/vimfiles路径下新建文件夹bundle，然后在此文件夹下克隆github上的vundel项目：
git clone https://github.com/VundleVim/Vundle.vim.git Vundle.vim

# 完成后会在bundle文件夹下看到Vundle.vim文件夹下的内容。
# 在VIM的配置文件_vimrc中开始配置vundle；
set nocompatible	" be iMproved, required
filetype off		" required
"Vundle的路径
set rtp+=$VIM/vimfiles/bundle/Vundle.vim
"插件的安装路径
call vundle#begin('$VIM/vimfiles/bundle/')
" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
"my bundle plugi

call vundle#end()		" required
filetype plugin indent on	" required

# 保存后，在VIM的标准命令模式normal下进行vundle插件安装
:PluginInstall

Vundle常用命令：
:PluginList 		列出已经安装的插件
:PluginInstall 		安装所有配置文件中的插件
:PluginInstall! 	更新所有插件
:PluginUpdate		更新所有插件
:PluginSearch 		搜索插件
:PluginSearch! 		更新本地缓存
:PluginClean 		确认删除不用的插件
:PluginClean! 		根据配置文件删除插件
```
Linux:

```
1、直接在命令行输入：
git clone https://github.com/gmarik/vundle.git ~/.vim/bundle/Vundle.vim
2、在.vimrc文件下输入如下代码：
set nocompatible 
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin('~/.vim/bundle/')
"此处为插件引用
Plugin 'VundleVim/Vundle.vim'

call vundle#end()
filetype plugin indent on
```

插件类别不同，在vundel中写入的路径也不同

```
" 在GitHub上托管的，需写出"用户名/插件名"
Plugin 'tpope/vim-fugitive'
" 不在GitHub上托管的，需要写出git全路径
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}
```

## vim-virtualenv

识别python虚环境

配置

```
Plugin 'plytophogy/vim-virtualenv'
```

用法

```
:VirtualEnvDeactivate		# 取消当前的虚环境
:VirtualEnvList				# 列出所有的虚环境
:VirtualEnvActivate spam	# 激活虚环境spam
:VirtualEnvActivate <tab>	# 如果您不确定要激活哪一个，可以使用制表符完成
:help virtualenv			# 更多细节
```

## indentLine

代码缩进对齐提示

```
Plugin 'Yggdroot/indentLine'

" 自定义配置
" 关闭默认颜色(灰色)
let g:indentLine_setColors = 0
" 自定义颜色
" Vim
let g:indentLine_color_term = 239
" GVim
let g:indentLine_color_gui = '#A4E57E'
" none X terminal
let g:indentLine_color_tty_light = 7 " (default: 4)
let g:indentLine_color_dark = 1 " (default: 2)
" Background (Vim, GVim)
let g:indentLine_bgcolor_term = 202
let g:indentLine_bgcolor_gui = '#FF5F00'
" 自定义标识符
let g:indentLine_char = 'ASCII character'或¦, ┆, │, ⎸, or ▏
" 自定义隐藏额行为
let g:indentLine_concealcursor = 'inc'
" level非1或2时插件停止运行
let g:indentLine_conceallevel = 2
" 保留自己的隐藏设定
let g:indentLine_setConceal = 0

" 关闭默认行为
let g:indentLine_enabled = 0
```

## Autoformat

一键格式化代码

需要安装有代码格式化工具

```
pip install autopep8
```

配置

```
Plugin 'Chiel92/vim-autoformat'
nnoremap <F6> :Autoformat<CR>
let g:autoformat_autoindent = 0
let g:autoformat_retab = 0
let g:autoformat_remove_trailing_spaces = 0
```

## nerdtree

文件树，在vim中浏览文件夹

配置

```
Plugin 'scrooloose/nerdtree'

nnoremap <F3> :NERDTreeToggle<CR>
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" 显示行号
let NERDTreeShowLineNumbers=1
let NERDTreeAutoCenter=1
" 是否显示隐藏文件
let NERDTreeShowHidden=1
" 设置宽度
let NERDTreeWinSize=21
" 忽略一下文件的显示
let NERDTreeIgnore=['\.pyc','\~$','\.swp']
" 显示书签列表
let NERDTreeShowBookmarks=1

```

用法

```
?: 快速帮助文档
o: 打开一个目录或者打开文件，创建的是buffer，也可以用来打开书签
go: 打开一个文件，但是光标仍然留在NERDTree，创建的是buffer
t: 打开一个文件，创建的是Tab，对书签同样生效
T: 打开一个文件，但是光标仍然留在NERDTree，创建的是Tab，对书签同样生效
i: 水平分割创建文件的窗口，创建的是buffer
gi: 水平分割创建文件的窗口，但是光标仍然留在NERDTree
s: 垂直分割创建文件的窗口，创建的是buffer
gs: 和gi，go类似
x: 收起当前打开的目录
X: 收起所有打开的目录
e: 以文件管理的方式打开选中的目录
D: 删除书签
P: 大写，跳转到当前根路径
p: 小写，跳转到光标所在的上一级路径
K: 跳转到第一个子路径
J: 跳转到最后一个子路径
<C-j>和<C-k>: 在同级目录和文件间移动，忽略子目录和子文件
C: 将根路径设置为光标所在的目录
u: 设置上级目录为根路径
U: 设置上级目录为跟路径，但是维持原来目录打开的状态
r: 刷新光标所在的目录
R: 刷新当前根路径
I: 显示或者不显示隐藏文件
f: 打开和关闭文件过滤器
q: 关闭NERDTree
A: 全屏显示NERDTree，或者关闭全屏
```

## tarbar

标签预览

配置

```
# 前提配置好ctags

"ctags标签预览
Plugin 'majutsushi/tagbar'

"Tagbar标签预览
"文件侦查启动,用以检测文件的后缀  
filetyp on  
"设置tagbar的窗口宽度  
let g:tagbar_width=30  
"设置tagbar的窗口显示的位置,为左边  
let g:tagbar_right=1  
"打开文件自动 打开tagbar  
autocmd BufReadPost *.cpp,*.c,*.h,*.hpp,*.cc,*.cxx call tagbar#autoopen()  
"映射tagbar的快捷键  
nmap <F7> :TagbarToggle<CR>
```

## ctrlp

vim中的模糊文件，缓冲区，标记的发现者

配置

```
Plugin 'ctrlpvim/ctrlp.vim'

# 更改默认映射和默认命令来调用CtrlP：
let g：ctrlp_map  =  '<c-p>'
let g：ctrlp_cmd  =  'CtrlP'

# 当没有明确的启动目录时，CtrlP会根据这个变量设置它的本地工作目录：
let g:ctrlp_working_path_mode = 'ra'
'c' 	- 当前文件的目录。
'a'		- 当前文件的目录，除非它是cwd的子目录
'r'		- 当前文件最近的父级目录，当前目录包含这些目录或文件：.git .hg .svn .bzr _darcs
'w'		- 修饰符“r”：从cwd开始搜索，而不是当前文件目录
0或''    - 禁用此功能。

# 如果项目中不存在默认标记（.git .hg .svn .bzr _darcs），则可以使用以下内容定义其他标记
let g:ctrlp_root_markers = ['pom.xml', '.p4ignore']
如果指定了多个模式，则会按顺序尝试，直到找到目录。

# 如果文件已经打开，请在新窗格中再次打开，而不是切换到现有窗格
let g:ctrlp_switch_buffer = 'et'

#排除文件和目录。使用Vim的命令wildignore和CtrlP的命令g:ctrlp_custom_ignore创建自定义列表：
set wildignore+=*/tmp/*,*.so,*.swp,*.zip     " MacOSX/Linux
set wildignore+=*\\tmp\\*,*.swp,*.zip,*.exe  " Windows

let g:ctrlp_custom_ignore = '\v[\/]\.(git|hg|svn)$'
let g:ctrlp_custom_ignore = {
  \ 'dir':  '\v[\/]\.(git|hg|svn)$',
  \ 'file': '\v\.(exe|so|dll)$',
  \ 'link': 'some_bad_symbolic_links',
  \ }

# 使用自定义文件列表命令：
let g:ctrlp_user_command = 'find %s -type f'        " MacOSX/Linux
let g:ctrlp_user_command = 'dir %s /-n /b /s /a-d'  " Windows
  
# 忽略文件 .gitignore 
let g:ctrlp_user_command = ['.git', 'cd %s && git ls-files -co --exclude-standard']

:help ctrlp-options 	# 查找其他指令
```

使用

```
# 检查
:help ctrlp-commands
:help ctrlp-extensions

# 运行
:CtrlP或:CtrlP [starting-directory]	# 在查找文件模式下调用CtrlP。
:CtrlPBuffer或:CtrlPMRU				# 调用查找缓冲区中的CtrlP或查找MRU文件模式。
:CtrlPMixed						     # 同时搜索文件，缓冲区和MRU文件。

# 一旦CtrlP打开：
<F5>					# 清除当前目录的缓存以获取新文件，删除已删除的文件并应用新的忽略选项。
<c-f>和<c-b>			   # 在模式之间循环。
<c-d>					# 切换到仅文件名而不是全路径搜索。
<c-r>					# 切换到正则表达式模式。
<c-j>，<c-k>或箭头键		# 浏览结果列表。
<c-t>或<c-v>，<c-x>	   # 在新选项卡或新的拆分中打开选定的条目。
<c-n>，<c-p>			    # 在提示的历史记录中选择下一个/上一个字符串。
<c-y>					# 创建一个新的文件，它的父目录。
<c-z>					# 标记/取消标记多个文件，并<c-o>打开它们。

:help ctrlp-mappings或提交?以获得更多映射帮助。
..				# 跳转上级目录。
:25				# 在打开的文件中执行命令,用于跳到第25行	
:diffthis。		# 打开多个文件时,前4个文件与当前文件对比。
```



## rainbow_parentheses

使用不同的颜色高亮匹配的括号

配置

```
git clone https://github.com/kien/rainbow_parentheses.vim rainbow_parentheses.vim

Plugin 'kien/rainbow_parentheses.vim'
let g:rbpt_colorpairs = [
                        \ ['brown',       'RoyalBlue3'],
                        \ ['Darkblue',    'SeaGreen3'],
                        \ ['darkgray',    'DarkOrchid3'],
                        \ ['darkgreen',   'firebrick3'],
                        \ ['darkcyan',    'RoyalBlue3'],
                        \ ['darkred',     'SeaGreen3'],
                        \ ['darkmagenta', 'DarkOrchid3'],
                        \ ['brown',       'firebrick3'],
                        \ ['gray',        'RoyalBlue3'],
                        \ ['darkmagenta', 'DarkOrchid3'],
                        \ ['Darkblue',    'firebrick3'],
                        \ ['darkgreen',   'RoyalBlue3'],
                        \ ['darkcyan',    'SeaGreen3'],
                        \ ['darkred',     'DarkOrchid3'],
                        \ ['red',         'firebrick3'],
                        \ ]
let g:rbpt_max = 16
let g:rbpt_loadcmd_toggle = 0
au VimEnter * RainbowParenthesesToggle
au Syntax * RainbowParenthesesLoadRound
au Syntax * RainbowParenthesesLoadSquare
au Syntax * RainbowParenthesesLoadBraces
```

##vim-airline

状态栏增强

配置

```
Plugin 'vim-airline/vim-airline'
```

## ale(替代syntastic)

代码检查，利用了vim8的异步处理功能，用起来不会有syntastic的卡顿现象

需要代码检测工具支持

```
pip install flake8
```

配置

```
git clone https://github.com/w0rp/ale ale

Plugin 'w0rp/ale'
 let g:ale_fix_on_save = 1
 let g:ale_completion_enabled = 1
 let g:ale_sign_column_always = 1
 let g:airline#extensions#ale#enabled = 1
```

syntastic配置

```
Plugin 'scrooloose/syntastic'
set statusline+=%#warningmsg#
set statusline+=%{SyntasticStatuslineFlag()}
set statusline+=%*
let g:syntastic_always_populate_loc_list = 1
let g:syntastic_auto_loc_list = 1
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

```

## 自动补全

```
YouCompleteMe		# 需要python编译，全支持
neocomplete 		# 需要lua编译，全支持
Python-mode			# 需要python编译，支持python
jedi-vim			# 需要python编译，需要jedi包
pydiction			# 不需编译，支持python
```

## xptemplate

自动补全代码块

配置

```
Plugin 'drmingdrmer/xptemplate'
```

用法

```
>vim xpt.c
for<C-\>
generates:

for (i = 0; i < len; ++i){
    /* cursor */
}
Press <tab>,<tab>.. to navigate through "i", "0", "len" and finally stop at "/* cursor */"
```



## vim-fugitive

执行基本的git指令

```
Plugin 'tpope/vim-fugitive'

```

用法

```
：Gstatus	# git status的输出
Press - to add/reset a file's changes, or p to add/reset --patch
:Gcommit	# git commit
:Gblame		# git blame
:Gedit		# go back to the work tree version.
:Gmove		# git mv on a file and simultaneously renames the buffer.
:Gdelete	# git rm on a file and simultaneously deletes the buffer.
:Ggrep		# earch the work tree (or any arbitrary commit) with git grep
:Glog 		# loads all previous revisions of a file into the quickfix list 
:Gread		# a variant of git checkout -- filename that operates on the buffer rather than the filename. 
:Gwrite 	# writes to both the work tree and index versions of a file, making it like git add when called from a work tree file and like git checkout when called from the index or a blob in history.
:Gbrowse 	# to open the current file on the web front-end of your favorite hosting provider, with optional line range (try it in visual mode!).
:Git		#  running any arbitrary command
Git!		# open the output of a command in a temp file.
```

##vim-commentary

快速注释

配置

```
Plugin 'tpope/vim-commentary'

autocmd FileType apache setlocal commentstring=#\ %s
```

用法

```
gcc			# 普通模式下，注释/取消注释一行
gc			# 普通模式下，注释/取消注释两行，可视，注释/取消注释一行
n gcc				# 注释n行
gcap 				# 普通模式下，注释整块
:7,17Commentary		# 注释所指定的行
:g/TODO/Commentary	# 作为全局调用的一部分

gcgc 				# 对相邻的注释行取消注释
```

## markdown-preview.vim

依赖与安装

```
Plugin 'iamcco/markdown-preview.vim'
Plugin 'iamcco/mathjax-support-for-mkdp'
```

配置

```
"实时浏览markdown文件
" 设置 chrome 浏览器的路径（或是启动 chrome（或其他现代浏览器）的命令）
" 如果设置了该参数, g:mkdp_browserfunc 将被忽略
let g:mkdp_path_to_chrome = ""
" vim 回调函数, 参数为要打开的 url
let g:mkdp_browserfunc = 'MKDP_browserfunc_default'
" 设置为 1 可以在打开markdown文件的时候自动打开浏览器预览，只在打开markdown文件的时候打开一次    
let g:mkdp_auto_start = 0
" 设置为 1 在编辑 markdown 的时候检查预览窗口是否已经打开，否则自动打开预览窗口 
let g:mkdp_auto_open = 0
" 在切换 buffer 的时候自动关闭预览窗口，设置为 0 则在切换 buffer 的时候不自动关闭预览窗口    "    
let g:mkdp_auto_close = 1
" 设置为 1 则只有在保存文件，或退出插入模式的时候更新预览，默认为 0，实时更新预览
let g:mkdp_refresh_slow = 0
" 设置为 1 则所有文件都可以使用 MarkdownPreview 进行预览，默认只有markdown文件可以使用改命令    
let g:mkdp_command_for_global = 0

nmap <silent> <F8> <Plug>MarkdownPreview        
imap <silent> <F8> <Plug>MarkdownPreview       
nmap <silent> <F9> <Plug>StopMarkdownPreview   
imap <silent> <F9> <Plug>StopMarkdownPreview  
```

