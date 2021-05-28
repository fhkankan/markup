# mac

## brew

>类似ubuntu中的apt-get包管理器，主要装非图形化界面，需下载源码，编译，安装

[官网](https://brew.sh)

官方命令安装卸载

```shell
# 安装
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# 卸载
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall.sh)"
```

国内安装卸载

```shell
# 方法一：安装配置
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
# 方法二：安装dmg，配置启动

# 卸载
cd `brew --prefix`
rm -rf Cellar
brew prune
rm -rf Library .git .gitignore bin/brew README.md share/man/man1/brew
rm -rf ~/Library/Caches/Homebrew
```

使用

```shell
brew update                # brew自身更新
brew cleanup             	#清除下载的缓存

brew search [pack]		# 搜索brew支持的软件
brew info [pack]		# 显示软件的各种信息
brew list               # 列出通过brew安装的所有软件

brew install [pack]     # 安装源码, 多软件时空格区分
brew uninstall [pack]	# 卸载软件

brew outdated 			# 查看可用的更新
brew outdated --greedy  # 查看所有可用的更新，包括标记了 auto_updates或者latest版本号的软件包；
brew upgrade 			# 更新所有软件包，不包括标记了auto_updates或者latest版本号的软件包；
brew upgrade [pack]     # 更新安装过的软件
brew upgrade --greedy 	# 更新所有可用软件包


(PS:详见man brew)
```

cask

```
brew [command] --cask  # 对于图形化界面应用，在常用brew后面加--cask
```

- 使用异常

报错信息

```
invalid 'depends_on macos' value: ">= :lion"
```

解决方法

```shell
/usr/bin/find "$(brew --prefix)/Caskroom/"*'/.metadata' -type f -name '*.rb' -print0 | /usr/bin/xargs -0 /usr/bin/perl -i -pe 's/depends_on macos: \[.*?\]//gsm;s/depends_on macos: .*//g'
```

- 离线安装

方法步骤：

1. 输入 `brew cask upgrade`，获取 calibre 最新安装包的下载链接（此时在默认下载位置会生成一个.incomplete 的中间文件）
2. 通过 [Aria2GUI](https://github.com/yangshun1029/aria2gui)，下载该安装包
3. 将下载的安装包移动至 Homebrew Cask 的默认下载位置，即 `/Users/ouyang/Library/Caches/Homebrew/Cask`
4. 将安装包更名，主要是在软件名与版本号之间增加一个 `-`
5. 再次运行 `brew cask upgrade` 命令，Homebrew 在检测到该软件安装包后，会跳过下载步骤，校验文件信息是否与 Homebrew 预存的一致，如一致则执行后续安装步骤。

## 系统文件

```
# 显示
defaults write com.apple.finder AppleShowAllFiles TRUE
killall Finder
# 关闭
defaults write com.apple.finder AppleShowAllFiles FALSE
killall Finder
```

## zsh/bash

切换zsh

```
# 查看支持的shell
cat /etc/shells
# 切换为zsh
chsh -s /bin/zsh
# 复原
chsh -s /bin/bash
```

oh my zsh

```
# 安装
curl -L https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh | sh
# 卸载
uninstall oh-my-zsh
```

更改终端提示符

```shell
vim /etc/bashrc
vim /etc/zshrc

#更改PS1
# bash
\u			# 用户名
\h			# 主机名
\W			# 当前工作目录
\$			# 显示$提示符

# zsh
%n			# 用户名
%m			# 主机名
%c			# 目录base
%/			# 完整目录
%%		  # 显示%
%1			# 当前父文件夹名
```

## 环境变量

[参考](https://blog.csdn.net/yilovexing/article/details/78750946)

mac 一般使用bash作为默认shell，其环境变量的配置文件为
```
/etc/profile 
/etc/paths 
/etc/bashrc
~/.bash_profile
~/.bash_login 
~/.bashrc
```
> 执行顺序

其中`/etc/profile,/etc/paths,/etc/bashrc`是系统级别的，变量生效是全局的。系统启动就会依次加载。

后面3个是当前用户级的环境变量，按照从前往后的顺序读取，如果`~/.bash_profile`文件存在，则后面的几个文件就会被忽略不读了，如果`~/.bash_profile`文件不存在，才会以此类推读取后面的文件。`~/.bashrc`没有上述规则，它是bash shell打开的时候载入的。

> 修改原则

前3个中，需要ROOT权限，建议修改`/etc/paths`文件

后3个中，建议修改`~/.bashrc`

- 全局设置

下面的几个文件设置是全局的，修改时需要root权限

> /etc/paths （全局建议修改这个文件 ）

编辑 paths，将环境变量添加到 paths文件中 ，一行一个路径

注意：输入环境变量时，不用一个一个地输入，只要拖动文件夹到 Terminal 里就可以了。

建议：并不修改此文件，而是添加新的文件
```
1.创建一个文件：
sudo touch /etc/paths.d/mysql
2.用 vim 打开这个文件：
sudo vim /etc/paths.d/mysql
3.编辑该文件，键入路径并保存（关闭该 Terminal 窗口并重新打开一个，就能使用 mysql 命令了）
/usr/local/mysql/bin
据说，这样可以自己生成新的文件，不用把变量全都放到 paths 一个文件里，方便管理。
```
>/etc/profile （建议不修改这个文件 ）

全局（公有）配置，不管是哪个用户，登录时都会读取该文件。

>/etc/bashrc （一般在这个文件中添加系统级环境变量）

全局（公有）配置，bash shell执行时，不管是何种方式，都会读取此文件。

- 单个用户设置

> ~/.bash_profile 或~/.bashrc 

注：Linux 里面是 .bashrc 而 Mac 是 .bash_profile

若bash shell是以login方式执行时，才会读取此文件。该文件仅仅执行一次！

设置命令别名

```
alias ll="ls -la"
```
设置环境变量

```
export PATH=/opt/local/bin:/opt/local/sbin:$PATH
```

- 生效

如果想立刻生效，则可执行下面的语句：
```shell
source 相应的文件
```
一般环境变量更改后，重启后生效。

## 安装破解软件

显示 显示"任何来源"选项在控制台中执行：

```
sudo spctl --master-disable
```

不显示"任何来源"选项（macOS 10.12默认为不显示）在控制台中执行：

```
sudo spctl --master-enable
```

## 相关服务

安装

```
brew install mysql
brew install nginx
brew install redis
brew install mosquitto
```

相关指令

```shell
# 服务列表
brew services list
# 运行服务(无注册登陆)
brew services run (服务名|--all)
# 启动服务
brew services start (服务名|--all)
# 停止服务
brew services stop (服务名|--all)
# 重启服务
brew services restart (服务名|--all)
# 删除所有不使用的服务
brew services cleanup
```

## macVIM

配置文件

```shell
filetype on
filetype plugin indent on

"快捷键绑定================================
"<space>    --------------------折叠
"<C-J>      --------------------下个窗口
"<C-K>      --------------------上个窗口
"<C-L>      --------------------右边窗口
"<C-H>      --------------------左边窗口
"<C-f>      --------------------Autoformat
"<C-t>      --------------------NERDTreeToggle
"<C-b>      --------------------TagbarToggle

"设置外观==================================
set number                      "显示行号 
"set guioptions-=m              "隐藏菜单栏
set guioptions-=T               "隐藏工具栏
set guioptions-=L               "隐藏左侧滚动条
set guioptions-=r               "隐藏右侧滚动条 
set guioptions-=b               "隐藏底部滚动条
set showtabline=0               "隐藏顶部标签栏
"set showtabline=2 	            "显示顶部标签栏
set cursorline                  "突出显示当前行
"set cursorcolumn               "突出显示当前列
set guicursor+=a:blinkon0       "设置光标不闪烁
set langmenu=zh_CN.UTF-8        "显示中文菜单
set helplang=cn		            "设置中文帮助
set guifont=Monaco:h12	        "设置字体为Monaco，大小10
colorscheme monokai             "开启颜色
"set list 		                "显示制表符
set scrolloff=5 	            "在光标接近底端或顶端时，自动下滚或上滚5行
set laststatus=2                "命令行为两行"

"编程辅助==================================
syntax enable                   "打开语法高亮
syntax on                       "开启语法高亮
set fileformat=unix             "设置以unix的格式保存文件"
set cindent                     "设置C样式的缩进格式"
set tabstop=4                   "一个tab 显示出来是多少个空格，默认 8
set shiftwidth=4                "每一级缩进是多少个空格
set backspace+=indent,eol,start "set backspace&可以对其重置
set backspace=2 	            "设置退格键可用
set wrap 		                "设置自动换行
"set nowrap                     "设置代码不折行
set linebreak 		            "整词换行，与自动换行搭配使用
set showmatch                   "设置匹配模式，相当于括号匹配
set smartindent                 "智能对齐
"set shiftwidth=4               "换行时，交错使用4个空格
set autoindent                  "设置自动对齐
set ai!                         "设置自动缩进
set fdm=indent "
set foldmethod=indent           "根据每行缩进开启折叠
set foldlevel=99                "折叠层级

"空格折叠收起
nnoremap <space> za    

"设置文件的代码形式 utf8====================
set encoding=utf-8
set termencoding=utf-8
set fileencoding=utf-8
set fileencodings=ucs-bom,utf-8,chinese,cp936
"其他杂项==================================
set mouse=a 		            "设置在任何模式下鼠标都可用
set selection=exclusive
set selectmode=mouse,key
set matchtime=5
set ignorecase                  "忽略大小写"
set hlsearch                    "高亮显示查找结果
set incsearch                   "增量查找
set noexpandtab                 "不允许扩展table"
set whichwrap+=<,>,h,l
set history=500		            "保留历史记录
set autochdir 		            "自动设置当前目录为正在编辑的目录
set hidden 		                "自动隐藏没有保存的缓冲区，切换buffer时不给出保存当前buffer的提示
set autoread 		            "设置当文件在外部被修改，自动更新该文件
set nobackup 		            "设置不生成备份文件

"===========================================
"窗口设置
"===========================================
" 指定屏幕上可以进行分割布局的区域
set splitbelow               " 允许在下部分割布局
set splitright               " 允许在右侧分隔布局

" 组合快捷键：
nnoremap <C-J> <C-W><C-J>    
nnoremap <C-K> <C-W><C-K>    
nnoremap <C-L> <C-W><C-L>    
nnoremap <C-H> <C-W><C-H> 

"===========================================
"插件安装
"===========================================
set nocompatible 
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin('~/.vim/bundle/')
"此处为插件引用
Plugin 'VundleVim/Vundle.vim'
Plugin 'plytophogy/vim-virtualenv'
Plugin 'Yggdroot/indentLine'
Plugin 'Chiel92/vim-autoformat'
Plugin 'scrooloose/nerdtree'
Plugin 'Xuyuanp/nerdtree-git-plugin'
Plugin 'majutsushi/tagbar'
Plugin 'kien/rainbow_parentheses.vim'
Plugin 'vim-airline/vim-airline'
Plugin 'jiangmiao/auto-pairs'
Plugin 'w0rp/ale'
Plugin 'drmingdrmer/xptemplate'
Plugin 'tpope/vim-commentary'
Plugin 'mattn/emmet-vim'
Plugin 'kien/ctrlp.vim'
Plugin 'Valloric/YouCompleteMe'
Plugin 'tpope/vim-fugitive'

call vundle#end()
filetype plugin indent on

"===========================================
"插件配置
"===========================================
"==========indentline=================
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
let g:indentLine_char = '¦'
" 自定义隐藏额行为
let g:indentLine_concealcursor = 'inc'
" level非1或2时插件停止运行
let g:indentLine_conceallevel = 2
" 保留自己的隐藏设定
let g:indentLine_setConceal = 0

" 关闭默认行为
let g:indentLine_enabled = 1

"==========autoformat=================
nnoremap <C-f> :Autoformat<CR>
let g:autoformat_autoindent = 0
let g:autoformat_retab = 0
let g:autoformat_remove_trailing_spaces = 0

"==========nerdtreetoggle=================
nnoremap <C-t> :NERDTreeToggle<CR>
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" 是否显示隐藏文件
let NERDTreeShowHidden=0
" 设置忽略文件类型
let NERDTreeIgnore=['\.pyc','\~$','\.swp','\.idea','\.DS_Store']
" 显示书签列表
let NERDTreeShowBookmarks=1
let NERDTreeChDirMode=1
" 窗口大小"
let NERDTreeWinSize=25
" 显示行号
let NERDTreeShowLineNumbers=1
let NERDTreeAutoCenter=1
" 修改默认箭头
let g:NERDTreeDirArrowExpandable = '▸'
let g:NERDTreeDirArrowCollapsible = '▾'
" 在终端启动vim时，共享NERDTree
let g:nerdtree_tabs_open_on_console_startup=1
"How can I open a NERDTree automatically when vim starts up if no files were specified?
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 0 && !exists("s:std_in") | NERDTree | endif
" 打开vim时自动打开NERDTree
autocmd vimenter * NERDTree           
"How can I open NERDTree automatically when vim starts up on opening a directory?
autocmd StdinReadPre * let s:std_in=1
autocmd VimEnter * if argc() == 1 && isdirectory(argv()[0]) && !exists("s:std_in") | exe 'NERDTree' argv()[0] | wincmd p | ene | endif
" 关闭vim时，如果打开的文件除了NERDTree没有其他文件时，它自动关闭，减少多次按:q!
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif
" 开发的过程中，我们希望git信息直接在NERDTree中显示出来， 和Eclipse一样，修改的文件和增加的文件都给出相应的标注， 这时需要安装的插件就是 nerdtree-git-plugin,配置信息如下
let g:NERDTreeIndicatorMapCustom = {
    \ "Modified"  : "✹",
    \ "Staged"    : "✚",
    \ "Untracked" : "✭",
    \ "Renamed"   : "➜",
    \ "Unmerged"  : "═",
    \ "Deleted"   : "✖",
    \ "Dirty"     : "✗",
    \ "Clean"     : "✔︎",
    \ "Unknown"   : "?"
    \ }

"==========tarbar=================
"Tagbar标签预览
"映射tagbar的快捷键  
nmap <C-b> :TagbarToggle<CR>
"文件侦查启动,用以检测文件的后缀  
filetyp on  
"设置tagbar的窗口宽度  
let g:tagbar_width=30  
"设置tagbar的窗口显示的位置,为左边  
let g:tagbar_right=1  
"打开文件自动 打开tagbar  
autocmd BufReadPost *.cpp,*.c,*.h,*.hpp,*.cc,*.cxx,*py, call tagbar#autoopen()  

"==========rainbow=================
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

"==========vim-commentary=================
autocmd FileType apache setlocal commentstring=#\ %s

"==========YCM=================
" 补全菜单的开启与关闭
set completeopt=longest,menu                    " 让Vim的补全菜单行为与一般IDE一致(参考VimTip1228)
let g:ycm_min_num_of_chars_for_completion=2             " 从第2个键入字符就开始罗列匹配项
let g:ycm_cache_omnifunc=0                      " 禁止缓存匹配项,每次都重新生成匹配项
let g:ycm_autoclose_preview_window_after_completion=1       " 智能关闭自动补全窗口
autocmd InsertLeave * if pumvisible() == 0|pclose|endif         " 离开插入模式后自动关闭预览窗口

" 补全菜单中各项之间进行切换和选取：默认使用tab  s-tab进行上下切换，使用空格选取。可进行自定义设置：
"let g:ycm_key_list_select_completion=['<c-n>']
"let g:ycm_key_list_select_completion = ['<Down>']      " 通过上下键在补全菜单中进行切换
"let g:ycm_key_list_previous_completion=['<c-p>']
"let g:ycm_key_list_previous_completion = ['<Up>']
inoremap <expr> <CR>       pumvisible() ? "\<C-y>" : "\<CR>"    " 回车即选中补全菜单中的当前项

" 开启各种补全引擎
let g:ycm_collect_identifiers_from_tags_files=1         " 开启 YCM 基于标签引擎
let g:ycm_auto_trigger = 1                  " 开启 YCM 基于标识符补全，默认为1
let g:ycm_seed_identifiers_with_syntax=1                " 开启 YCM 基于语法关键字补全
let g:ycm_complete_in_comments = 1              " 在注释输入中也能补全
let g:ycm_complete_in_strings = 1               " 在字符串输入中也能补全
let g:ycm_collect_identifiers_from_comments_and_strings = 0 " 注释和字符串中的文字也会被收入补全

" 重映射快捷键
"上下左右键的行为 会显示其他信息,inoremap由i 插入模式和noremap不重映射组成，只映射一层，不会映射到映射的映射
inoremap <expr> <Down>     pumvisible() ? "\<C-n>" : "\<Down>"
inoremap <expr> <Up>       pumvisible() ? "\<C-p>" : "\<Up>"
inoremap <expr> <PageDown> pumvisible() ? "\<PageDown>\<C-p>\<C-n>" : "\<PageDown>"
inoremap <expr> <PageUp>   pumvisible() ? "\<PageUp>\<C-p>\<C-n>" : "\<PageUp>"

"nnoremap <F5> :YcmForceCompileAndDiagnostics<CR>           " force recomile with syntastic
"nnoremap <leader>lo :lopen<CR>    "open locationlist
"nnoremap <leader>lc :lclose<CR>    "close locationlist
"inoremap <leader><leader> <C-x><C-o>

nnoremap <leader>jd :YcmCompleter GoToDefinitionElseDeclaration<CR> " 跳转到定义处
let g:ycm_confirm_extra_conf=0                  " 关闭加载.ycm_extra_conf.py确认提示
```



​    