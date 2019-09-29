Git版本库（本地仓库）

```
- Directory：根目录，由Git管理的一个目录，包含我们的工作区和Git仓库信息。
- Workspace：工作区，即项目的根目录，但不包括.git目录。
- .git： Git版本库目录，保存了所有的版本信息。该目录会由git初始化仓库的时自动生成。
- Index/Stage(阶段；舞台)： 暂存区，工作区变更，先提交到暂存区，再从暂存区提交到本地仓库。
- Local Repo： 本地仓库，保存了项目所有历史变更版本。
  	- HEAD指针： 表示工作区当前版本，HEAD指向哪个版本，当前工作区就是哪个版本；通过HEAD指针，可以实现版本回退。
- Stash(存放；贮藏)： 工作状态保存栈，用于保存和恢复工作区的临时工作状态（代码状态）。
```

## 安装和配置

```
# 安装git
sudo apt-get update
sudo apt-get install git

# 设置git用户名和邮箱(全局)
git config --global user.name "Your Name"
git config --global user.email "youremail@example.com"
```

## 基本使用

### 创建版本库

```
# 进入需要提交的文件夹及文件所在目录中
cd 目标文件夹下
# 初始化仓库
git init
```

### 提交代码

```
# 提交代码（代码修改）到 暂存区
git add 文件名		# 一个文件作了修改
git add .		  # 多个文件作了修改

# 提交暂存区的代码到本地仓库
git commit -m "代码修改说明"
```

> 注意事项

使用`commit`命令，漏写参数 `-m` 选项，则git会默认使用GNU Nano的编缉器打开 `.git/COMMIT_EDITMSG ` 文件。可做两种处理方式：

```
- 第1种： 可以按 ctrl+x 退出即可， 然后再重新执行 git commit并指定 -m 选项提交代码
- 第2种： 在打开的Nano编缉器中输入提交的注释说明， 再按 ctrl + o 保存， 接着按回车确认文件名， 最后再按ctrl + x退出， 回去之后，git就会提交之前的代码了。
```

### 查看历史版本

```
# 产看历史版本的所有信息（包括用户名和日期）
git log

# 用带参数的git log，输出的信息会短一些
git log --graph --pretty=oneline
```

## 管理修改

### 查看和提交修改

```
# 查看工作区修改
git status

# 了解修改后提交代码
git add .
git commit -m "注释" 
```

### 对比文件

```
# 对比工作区和暂存区的某个文件，了解作了哪些修改
git diff 文件名

说明：
减号表示： 本地仓库的代码
加号表示： 工作区的代码
```

### 版本回退

```
# 查看历史提交版本,只能看到当前版本之前的版本
git log
# 查看所有的历史版本
git reflog

# 工作区回退到某个版本
git reset --hard <commit版本号>
选项说明：
hard  重置： 本地仓库HEAD指针、暂存区、工作区
mixed 重置： 本地仓库HEAD指针、暂存区         【默认值】
soft  重置： 本地仓库HEAD指针


# 回退到上个版本
git reset --hard HEAD^

# 回退到上上个版本
git reset --hard HEAD^^

# 往上10个版本
git reset --hard HEAD~10
```

### 撤销修改

```
场景1： 改乱了工作区的代码，想撤销工作区的代码修改
git checkout -- <file>  # 撤销指定文件的修改
git checkout -- .		# 撤销当前目录下所有修改

说明：git checkout会用本地仓库中的版本替换工作区的版本，无论工作区是修改还是删除，都可以“一键还原”。 

场景2： 改乱了工作区的代码，提交到了暂存区，同时撤销工作区和暂存区修改：
方法一： 
# 撤销暂存区修改
git reset HEAD <file>     # 撤销暂存区指定文件的修改
git reset HEAD            # 撤销暂存区所有修改
# 撤销工作区的代码修改
git checkout -- <file>    # 撤销指定文件的修改
git checkout -- .		  # 撤销当前目录下所有修改
方法二：
# 同时撤销暂存区和工作区的修改：
git reset --hard HEAD
```

## 删除文件

```
场景1： 如果该文件是属于误删，那么要恢复回来：
git checkout -- 文件名

场景2： 确实要删除该文件：
# 提交代码修改到暂存区：	
git add <文件名>	    # 把文件修改操作，提交到暂存区
# 工作区删除后，提交代码修改到本地仓库
git commit -m "注释"
```

# 分支管理

在实际开发中，我们会创建多个分支，按照几个基本原则进行分支管理：

```
- master 主分支： 该分支是非常稳定的，仅用来发布新版本，不会提交开发代码到这里
- dev 开发分支： 不稳定的，团队成员都把各自代码提交到这里。当需要发布一个新版本，经测试通过后，把dev分支合并到master分支上, 作为一个稳定版本，比如v1.0
- featrue 功能分支： 团队每个人负责不同的功能，分别有各的分支，在各自的分支中干活，适当的时候往 dev开发分支合并代码。
```

## 基本使用

HEAD指针： 表示工作区当前版本，HEAD指向哪个版本，当前工作区就是哪个版本；

每个分支都有一个指针，指向当前分支的最新版本

每次提交一个版本，分支指针就会向前移动一步，HEAD指针也会往前移动一步； 

```shell
git branch			# 列出所有分支,当前分支前面会标一个*号

git branch dev	   	# 创建分支 (开发分支)
git checkout dev   	# 切换分支
# 以上两个操作，可以使用以下一个命令代替
git checkout -b dev # 创建+切换分支

git add readme.md	# 增加内容
git commit -m "branch test"	# 提交本地

git checkout master	# 切回master分支
git merge dev		# 合并指定分支到当前分支
```

## 合并冲突解决

|      | master分支       | feature1分支     | feature1合并到master的结果 |
| ---- | -------------- | -------------- | -------------------- |
| 场景1  | 无修改            | `readme文件` 有修改 | 自动合并成功               |
| 场景2  | `readme文件` 有修改 | `readme文件` 有修改 | 合并后有冲突，需要解决冲突，再提交代码  |
| 场景3  | 无修改            | 有新增文件          | ??                   |

**场景二 **：

```
# 查看冲突信息
git status

# 通过手动修改冲突文件，解决冲突

# 提交代码
git add .
git commit -m "冲突解决"

# 产看分支的合并情况
git log --graph --pretty=oneline
```

**场景三 ** 

```
合并后，会自动弹出一个窗口, 要求输入提交的注释，输入完注释后，按ctrl+x返回退出

分支的新文件，在master分支中不存在，所以合并到master分支后，要把该新文件提交到主分支上，就需要指定commit命令的提交注释。
```

## 分支删除

```
# 删除一个已合并的分支，注意，无法删除当前所在的分支，需要切换到其它分支，才能删除
git branch -d 分支名

# 如果分支还没有被合并，删除分支将会丢失修改。如果要强行删除，需要使用如下命令：
git branch -D 分支名

# 删除之后通过以下命令，就查看不到了
 git branch
```

## BUG分支(保存工作现场)

- **bug分支：**当你接到修复一个bug任务的时候，可以创建一个bug临时分支（例如：issue01）来修复它，修改完成再把该分支删除掉
- **问题：**当前正在dev上进行的工作还没有提交, 工作只进行到一半，还没法提交，但如果不提交就切换到其它分支工作，代码会丢失，怎么办呢？
- **使用工作状态保存栈：**Git版本库中有一个 **Stash 临时状态保存栈**, 可以使用它来保存当前工作现场，把当前工作现场保存起来，等完成了其它紧急工作，再回来恢复工作现场，继续接着工作
- **工作状态保存栈使用：**

```
- 保存当前工作现场
   git stash  
- 查看有哪些临时现场
  git stash list
  输出结果：
  stash@{0}: WIP on dev: 6224937 add merge
- 恢复某个临时现场
  git stash apply stash@{0} 
- 恢复最近保存的工作现场  	
  git stash pop
- 清空工作状态保存栈
  git stash clear
```

- 案例演示

  假设当前在dev分支上工作, master分支有bug，需要紧急修复

  	git checkout dev		    # 当前在dev分支上工作
  	git stash					# 工作到一半，需要保存工作现场
  	
  	# 要修复到哪个分支的bug(假定是master分支)，就切换到哪个分支，并创建临时bug分支
  	git checkout master			# 切换到master分支 
  	git checkout -b issue01		# 创建并切换到bug临时分支
  	
  	git add readme.txt 			
  	git commit -m "fix bug"	 	 # bug修改完，提交修改
  	
  	git checkout master			 # 改完bug切换回master主分支
  	git merge issue01			 # 合并bug分支到主分支
  	
  	git branch -d issue01		 # 删除bug分支
  	
  	git checkout dev			 # 切换回开发分支
  	
  	git stash pop                # 恢复到之前的工作现场

# 使用Github（远程仓库）

## Git托管平台

```
- GitHub
  - 官网地址： http://www.github.com
- 码云
  - 官网地址： http://www.gitee.com
```


## 配置SSH密钥对

Git通信协议

```
Git支持多种协议，包括SSH, https协议

使用ssh协议速度快，但是在某些只开放http端口的公司，内部就无法使用ssh协议， 而只能用https了

与ssh相比，使用https速度较慢，而且每次通过终端上传代码到远程仓库时，都必须输入账号密码
```
配置SSH密钥对

Git服务器会使用SSH密钥对来确认代码提交者的合法身份。

> 注册github账号
>
> 创建秘钥对
>
> 配置github后台

```shell
# 查看秘钥对(.ssh)
cd ~/.ssh/
ls 

# 创建
# 三次回车，生成.ssh目录，id_rsa （私钥）和id_rsa.pub （公钥）
ssh-keygen -t rsa -C youremail@example.com
       
# 查看id_rsa.pub公钥
cat ~/.ssh/id_rsa.pub
    
# 配置到 GitHub 的后台
登陆GitHub，点击 头像 -> settings -> SSH And GPG keys -> New SSH Keys: 
   
# 验证是否配置成功
ssh -T git@github.com
# 第一次，需要输入“yes”, 若返回类似 “Hi islet1010! You've successfully authenticated”，则配置成功。

```
## 上传本地项目

```
1、创建远程仓库: 登陆GitHub创建一个新的远程仓库：
在右上角找到“Create a new repo”按钮，Repository name填入项目名，再点击创建按钮：
仓库创建出来后，目前为空，要把本地的项目上传上来。

2、获取刚创建的Github远程仓库的地址
git@github.com:islet1010/PlaneGame.git

3、添加Git远程仓库地址
git remote add origin git@服务名:路径/仓库名.git
eg: git remote add origin git@github.com:islet1010/PlaneGame.git

4、推送代码到服务器
# -u参数把本地的master分支和远程的master分支关联起来
git push -u origin master
```

## 克隆项目

```
1. 获取要clone项目的地址，假设是上面刚上传的项目，它的地址：
   git@github.com:islet1010/PlaneGame.git
   
2. 进入ubuntu的某一个目录，例如 Workpace03目录，目前该目录为空：

3. 执行clone命令
   git clone git@github.com:islet1010/PlaneGame.git
```

## 推送分支 

把本地仓库 该分支所有的修改，推送到远程仓库对应的分支上,以便团队中其中人看到：

```
1. 创建并切换到分支f1
git checkout -b f1
       
2. 修改代码并提交
git add .
git commit -m "注释"

3. 推送分支到服务器
# 推送的分支远程服务器不存在
git push --set-upstream origin f1(方法一，同时建立了push和pull)
git push origin f1(方法二,只建立了push，未建立pull)
# 推送的分支远程服务器存在
git push
   
4. 到github上查看，会看到有新增了分支
```

**注意：并不需要把本地所有分支，都推送到服务器**

- master分支是主分支，因此要时刻与远程同步；
- dev分支是开发分支，团队所有成员都需要在上面工作，所以也需要与远程同步；
- bug分支只用于在本地修复bug，就没必要推到远程了，修复完bug一般会删除掉；
- feature分支是否推到远程，取决于你是否和你的小伙伴合作开发。

## 拉取分支

团队开发时，更新服务器上别人提交的代码到本地：

```
场景一： 要拉取的分支本地不存在:
# 查看远程分支情况：
git remote show origin

# 已跟踪的远程分支，可以通过以下命令直接拉取下来（本地和远程分支的名称最好一致）：
git checkout -b 本地分支名 origin/远程分支名

# 新的远程分支，需要先获取更新，让新分支变为已跟踪状态，然后才能拉取。
git remote update
git checkout -b f1 origin/f1

场景二： 要拉取的分支本地已存在：
# 本地分支与远程分支已建立关联： 
git pull   # 拉取服务器分支的最新代码，与当前本地分支合并

# 本地分支与远程分支未建立关联，需要先建立本地与远程分支的链接关系再拉取
git branch --set-upstream-to=origin/远程分支名 本地分支名
git pull
```

## 忽略特殊文件

- 概述

某些文件需要放到Git工作目录中，但又不能提交它们，以下文件应该忽略：

```
- 操作系统自动生成的文件，比如缩略图Thumbs.db等
- 忽略编译生成的中间文件、可执行文件等，比如Java编译产生的.class文件
- 有敏感信息的配置文件，比如存放口令的配置文件
```

不需要从头写.gitignore文件，GitHub已经为我们准备了各种配置文件，只需要组合一下就可以使用了。所有配置文件可以直接在线浏览：[https://github.com/github/gitignore](https://github.com/github/gitignore)：

- 忽略规则：

在git中如果想忽略掉某个文件，不让这个文件提交到版本库中，可以使用修改根目录中 .gitignore 文件的方法（如果没有这个文件，则需自己手工建立此文件）。这个文件每一行保存了一个匹配的规则例如：

```
# 此为注释 – 将被 Git 忽略
*.sample 　　 # 忽略所有 .sample 结尾的文件
!lib.sample 　　 # 但 lib.sample 除外
/TODO 　　 # 仅仅忽略项目根目录下的 TODO 文件，不包括 subdir/TODO
build/ 　　 # 忽略 build/ 目录下的所有文件
doc/*.txt 　　# 会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
```

- 规则不生效的解决办法

把某些目录或文件加入忽略规则，按照上述方法定义后发现并未生效，原因是.gitignore只能忽略那些原来没有被追踪的文件，如果某些文件已经被纳入了版本管理中，则修改.gitignore是无效的。那么解决方法就是先把本地缓存删除（改变成未被追踪状态），然后再提交：

```
git rm -r --cached .
git add .
git commit -m 'update .gitignore'
```

- 示例： `.gitignore`文件

```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# pycharm
.idea/

# vscode
.vscode/

# mac
.DS_Store
```
# 命令行操作

## 设置

- 单平台公私钥

密钥

```shell
# 查看秘钥对(.ssh)
cd ~/.ssh/

# 创建
# 三次回车，生成.ssh目录，id_rsa （私钥）和id_rsa.pub （公钥）
ssh-keygen -t rsa -C youremail@example.com
       
# 查看id_rsa.pub公钥
cat ~/.ssh/id_rsa.pub

# 配置到远程仓库的后台ssh

# 验证是否配置成功
ssh -T git@github.com
# 第一次，需要输入“yes”, 若返回类似 “Hi islet1010! You've successfully authenticated”，则配置成功。
```

签名

```shell
# 系统级别签名
git config --globaluser.name [AAA]
git config --global user.email [邮箱地址]
cat .gitconfig
# 项目级别签名
cd 项目文件夹
git config user.name [AAA]
git config user.email [邮箱地址]
cat .git/config
```

- 多平台公私钥

生成不同名称

```shell
# gitlab
ssh-keygen -t rsa -C "fu.hang.2009@163.com" -f ~/.ssh/gitlab_id_rsa
# github
ssh-keygen -t rsa -C "fu.hang.2008@163.com" -f ~/.ssh/github_id_rsa
```

添加config文件

```python
# 添加config配置文件
# vi ~/.ssh/config

# 文件内容如下：
# gitlab
Host gitlab.com
    HostName gitlab.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/gitlab_id_rsa
# github
Host github.com
    HostName github.com
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/github_id_rsa

# 配置文件参数
# Host : Host可以看作是一个你要识别的模式，对识别的模式，进行配置对应的的主机名和ssh文件
# HostName : 要登录主机的主机名
# User : 登录名
# IdentityFile : 指明上面User对应的identityFile路径
```

签名

```shell
# 查看全局设置
git config --global --list 				
# 取消全局配置
git config --global --unset user.name	
git config --global --unset user.email
# 进入项目文件夹，设置局部配置
git config user.name "yourname"
git config user.email "your@email.com"
```

命令行进入项目目录，重建 origin

```shell
git remote rm origin
git remote add origin git@ieit.github.com
```

## 初始

- 新建仓库

```shell 
git clone xxx  # xxx表示地址
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

- 已存在的文件夹

```shell
cd existing_folder
git init
git remote add origin xxx  # xxx表示地址
git add .
git commit -m "Initial commit"
git push -u origin master
```

- 已存在仓库

```shell
cd existing_repo
git remote rename origin old-origin
git remote add origin xxx  # xxx表示地址
git push -u origin --all
git push -u origin --tags
```

## 查看

查看分支

```shell
git branch  			# 本地分支
git branch -r  		# 远程分支
git branch -a			# 所有分支
git remote show origin  # 查看远程分支详细情况
```

查看状态

```shell
git status  # 显示工作区状态
```

产看其他

```
git bisect	# 通过二分法查找定位引入bug的提交
git grep		# 输出和模式匹配的行
git show		# 显示各种类型的对象
```

查看日志

```shell
git log 
git log --pretty=oneline
git log --oneline
git reflog (HEAD@{移动到当前版本需要多少步})
```

查看差异

```shell
git diff [文件名]  # 将工作区中的文件和暂存区的进行比较
git diff [本地库历史版本] [文件名]  # 将工作区中的文件和本地库历史记录比较，不带文件名的话，会比较多个文件
```

## 切换

```shell
git checkout f1			# 切换本地已存在的分支f1
git checkout -b f1  # 创建并切换到新分支f1
git checkout -b f1 origin/f1  # 跟踪拉去远程分支f1，在本地起名为f1，并切换至分支f1
```

## 变更

- 添加

```shell
git add path1  						# 添加文件内容至索引
git add .  								# 添加所有变动的文件至索引
```

- 移动

```shell
git mv		# 移动或重命名一个文件、目录或符号链接
```

- 删除

```
git rm 		# 从工作区和索引中删除文件
```

- 回退

```
git reset --hard <commit-id>  # 撤销上次commit的内容
git reset --hard HEAD^		# 后退1步(一个^表示后退一步)
git reset --hard HEAD~2		# 后退2步(~后的数字n摆哦是后退n步)
```

- 记录

```shell
git commit -m '修改注释'    # 记录变更到仓库
```

- 合并

```shell
git checkout dev	# 切换至要合并到的分支dev
git merge f1			# 合并需要合并的分支f1
```

- 转移

```
git rebase		# 本地提交转移至更新后的上游分支中
```

- 标记

```
git tag			# 	创建、列出、删除或校验一个GPG签名的标签对象
```

## 协同

```shell
git fetch			# 从另外一个仓库下载对象和引用

git pull			# 获取并整合另外的仓库或一个本地分支

git push  								# 更新远程引用和相关对象
git push --set-upstream origin f1  # 若远程不存在此分支,创建新的远程分支，并追踪push和pull
git puh origin f1  # 方法二，只追踪push

git remote update  # 更新本地追踪显示的远程分支
git remote prune origin --dry-run  # 查看远程哪些分支需要清理
git remote prune origin		# 清除无效的远程追踪分支
```

## 高级

压缩合并多次记录

```shell
# 查看提交历史
git log
# 1.对commit进行rebase操作
git rebase -i HEAD~4  # 最近的4个进行合并
git rebase -i commitId  # 指明需要合并的commit位于哪个commitId后
# 2.对弹窗commands处理
pick：正常选中
reword：选中，并且修改提交信息；
edit：选中，rebase时会暂停，允许你修改这个commit
squash：选中，会将当前commit与上一个commit合并
fixup：与squash相同，但不会保存当前commit的提交信息
exec：执行其他shell命令
# 3.编辑后保存退出，git 会自动压缩提交历史，
esc
:
wq
# 4.如果有冲突，记得解决冲突后，使用 
git rebase --continue # 重新回到当前的 git 压缩过程；
git rebase --abort		# 放弃压缩命令
# 5. 推送到远程仓库
git push -f
```

删除某次记录

```shell
# 查看提交历史
git log
# 1.对commit进行rebase操作
git rebase -i HEAD~1  # 对最近1次commit进行处理
git rebase -i 9fbf10  # 对某次id为9fbf10之后的commit进行rebase
# 2.对弹窗commands处理
将需要删除的commit设置操作指令drop
# 3.保存退出

```

