# Git版本控制 

## 安装与概述
安装
```shell
# ubuntu
sudo apt-get update
sudo apt-get install git

# mac
brew install git
```


Git本地仓库

```
- Directory：根目录，由Git管理的一个目录，包含我们的工作区和Git仓库信息。
- Workspace：工作区，即项目的根目录，但不包括.git目录。
- .git： Git版本库目录，保存了所有的版本信息。该目录会由git初始化仓库的时自动生成。
- Index/Stage(阶段；舞台)： 暂存区，工作区变更，先提交到暂存区，再从暂存区提交到本地仓库。
- Local Repo： 本地仓库，保存了项目所有历史变更版本。
  	- HEAD指针： 表示工作区当前版本，HEAD指向哪个版本，当前工作区就是哪个版本；通过HEAD指针，可以实现版本回退。
- Stash(存放；贮藏)： 工作状态保存栈，用于保存和恢复工作区的临时工作状态（代码状态）。
```

## 创建查看

### 初始

```shell
cd existing_folder
git init
```

### 查看

查看状态

```python
git status  # 显示工作区状态
```

查看日志

```python
git log  # 产看历史版本的所有信息（包括用户名和日期）
git log --graph --pretty=oneline  # 用带参数的git log，输出的信息会短一些

git log --oneline  # 美化后的日志输出

git reflog (HEAD@{移动到当前版本需要多少步})

# 搜索
git log -S ZLIB_BUF_MAX --oneline  # 想找到ZLIB_BUF_MAX常量是什么时候引入的，-S选项来显示新增和删除该字符串的提交
git log -L :git_deflate_bound:zlib.c  # 想查看zlib.c文件中git_deflate_bound函数的每一次变更        
```

查看差异

```python
git diff [文件名]  # 将工作区中的文件和暂存区的进行比较
git diff [本地库历史版本] [文件名]  # 将工作区中的文件和本地库历史记录比较，不带文件名的话，会比较多个文件

# 说明：
# 减号表示： 本地仓库的代码
# 加号表示： 工作区的代码
```

查看其他

```python
git bisect	# 通过二分法查找定位引入bug的提交
git show		# 显示各种类型的对象

git blame filename   # 查看每一行的修改人以及commit-id
git show commit-id   # 查看详细的修改提交记录
```

搜索

```shell
# 很方便地从提交历史、工作目录、甚至索引中查找一个字符串或者正则表达式。

git grep -n gmtime_r  # 查找工作目录的文件，-n选项数来输出 Git 找到的匹配行的行号
git grep -c gmtime_r  # 不想打印所有匹配的项，-c或--count选项输出概述的信息， 其中仅包括那些包含匹配字符串的文件，以及每个文件中包含了多少个匹配
git grep -p gmtime_r *.c  # 关心搜索字符串的上下文，-p或--show-function选项来显示每一个匹配的字符串所在的方法或函数
```

## 管理修改

### 修改

- 添加

```shell
git add path1  # 添加文件内容至索引
git add .  	# 添加所有变动的文件至索引
```

- 移动

```shell
git mv  # 移动或重命名一个文件、目录或符号链接
```

- 删除

```python
git rm 		# 从工作区和索引中删除文件
git branch -d 分支名  # 删除一个已合并的分支(非当前分支)
git branch -D 分支名  # 强行删除分支(未合并则丢失修改)
```

- 记录

```shell
git commit -m '修改注释'    # 记录变更到仓库
git commit --amend  修改注释
git reset --soft HEAD^  # 撤销上次commit，不撤销add
```

> 注意事项

使用`commit`命令，漏写参数 `-m` 选项，则git会默认使用GNU Nano的编缉器打开 `.git/COMMIT_EDITMSG ` 文件。可做两种处理方式：

```
- 第1种： 可以按 ctrl+x 退出即可， 然后再重新执行 git commit并指定 -m 选项提交代码
- 第2种： 在打开的Nano编缉器中输入提交的注释说明， 再按 ctrl + o 保存， 接着按回车确认文件名， 最后再按ctrl + x退出， 回去之后，git就会提交之前的代码了。
```

- 标记

```shell
git tag						# 查看标签
git tag tag-name  			# 创建新标签/查看标签内容
git tag tag-name commit-id  # 对以往的记录进行标签 

git pull --tags  			# 拉取所有标签

git push origin tag-name 	# 推送某个标签
git push --tags			 	# 推送所有标签

git tag -d tage-name		# 删除本地标签
git push origin :refs/tags/tag-name  # 删除远程标签
```

### 回退

```shell
git log  # 查看历史提交版本,只能看到当前版本之前的版本
git reflog  # 查看所有的历史版本

# 工作区回退到某个版本
git reset --hard <commit版本号>
选项说明：
hard  重置： 本地仓库HEAD指针、暂存区、工作区
mixed 重置： 本地仓库HEAD指针、暂存区  【默认值】
soft  重置： 本地仓库HEAD指针

# 例子
git reset --soft HEAD^  # 撤销上次commit，不撤销add
git reset --hard <commit-id>  # 撤销上次commit的内容
git reset --hard HEAD^		# 后退1步(一个^表示后退一步)
git reset --hard HEAD^^		# 回退到上上个版本
git reset --hard HEAD~2		# 后退2步(~后的数字n是后退n步)
git push -f  # 强制提交
```

### 撤销

```python
# 场景1： 改乱了工作区的代码，想撤销工作区的代码修改
git checkout -- <file>  # 撤销指定文件的修改
git checkout -- .		# 撤销当前目录下所有修改

# 说明：git checkout会用本地仓库中的版本替换工作区的版本，无论工作区是修改还是删除，都可以“一键还原”。 

# 场景2： 改乱了工作区的代码，提交到了暂存区，同时撤销工作区和暂存区修改：
# 方法一： 
# 撤销暂存区修改
git reset HEAD <file>     # 撤销暂存区指定文件的修改
git reset HEAD            # 撤销暂存区所有修改
# 撤销工作区的代码修改
git checkout -- <file>    # 撤销指定文件的修改
git checkout -- .		  # 撤销当前目录下所有修改
# 方法二：
# 同时撤销暂存区和工作区的修改：
git reset --hard HEAD
```

## 分支管理

常用分支

```
- master 主分支： 该分支是非常稳定的，仅用来发布新版本，不会提交开发代码到这里
- develop 开发分支： 不稳定的，团队成员都把各自代码提交到这里。当需要发布一个新版本，经测试通过后，把dev分支合并到master分支上, 作为一个稳定版本，比如v1.0
- featrue 功能分支： 团队每个人负责不同的功能，分别有各的分支，在各自的分支中干活，适当的时候往 dev开发分支合并代码。
```

### 查看

```shell
git branch  			# 本地分支， 当前分支有*
git branch -r  		# 远程分支
git branch -a			# 所有分支
git remote show origin  # 查看远程分支详细情况
```
### 切换

```shell
git branch dev	   	# 创建分支 (开发分支)
git checkout dev   	# 切换本地已存在的分支f1
# 创建+切换
git checkout -b dev # 创建+切换分支
git branch --set-upstream-to=origin/develop dev  # 追踪分支
# 创建+切换+远程
git checkout -b f1 origin/f1  # 跟踪拉去远程分支f1，在本地起名为f1，并切换至分支f1
```
### 合并

- merge

指令

```shell
git checkout master  # 切换至要合并到的分支master
git merge develop  # 合并需要合并的分支到当前分支
# ---有冲突时，解决冲突
git status  # 查看冲突文件
git add .
git commit -m 'merge ...'
# ---完成合并
```
说明
```shell
# 步骤
1.找到要合并的分支develop和当前分支master最近的共同祖先commit点
2.若是这个点不是当前分支master最新的点，则会把当前分支masteer的最新的commit和要合并分支develop的最新的commit合并为一个新的commit，若有冲突，则需要解决冲突，提交解决冲突
3.将以上共同祖先commit点后的所有提交点，按照提交时间（不是push时间）的先后顺序依次放到master分支上

# 优缺点
优点：操作简单，按照提交时间排序，提交点完整。
缺点：节点过多，在主分支上有无明确意义节点(自动生成的merge记录)，呈现非整条线性直线的关系。

# 应用场景
对公共仓库代码合并处理时使用
```
- rebase

指令

```shell
git checkout develop
git rebase master 
# ---有冲突，则解决冲突，提交记录
git add .
git commit -m 'merge...' 
git rebase --continue
# -----
git checkout master
git merge develop
```

说明

```shell
# 步骤
1.找到这两个分支（即当前分支develop和变基操作的目标基底分支master） 的最近共同祖先commitId
2.对比当前分支相对于该祖先的历次提交，提取相应的修改并存为临时文件(patch)(这些补丁放到".git/rebase"目录中)
3.将当前分支指向目标master分支最新commit, 最后以此将之前另存为临时文件的修改依序应用。 
4.回到master分支，进行一次快进合并。

# 优缺点
优点：可以对某一段线性提交历史进行编辑、删除、复制、粘贴，提交历史干净简洁，形成线性提交历史。
缺点：操作不当，可能误改他人的提交commit，如对公共库进行rebase

# 应用场景
合并未推送的本地修改分支至公共分支时
```

更详细指令

```shell
git rebase -i  [startpoint]  [endpoint]

# 参数
-i：是--interactive，即弹出交互式的界面让用户编辑完成合并操作
[startpoint] [endpoint]则指定了一个编辑区间，如果不指定[endpoint]，则该区间的终点默认是当前分支HEAD所指向的commit(注：该区间指定的是一个前开后闭的区间)。
# 交互指令
pick：正常选中，保留该commit（缩写:p）
reword：选中，保留该commit，但要修改该commit的注释（缩写:r）
edit：选中，保留该commit, rebase时会暂停，允许修改这个commit（缩写:e）
squash：选中，会将当前commit与上一个commit合并，（缩写:s）
fixup：与squash相同，但不会保存当前commit的提交信息（缩写:f）
exec：执行其他shell命令（缩写:x）
drop：丢弃该commit（缩写:d）
```

合并多个commit为一个完整commit

```shell
# 查看提交历史
git log
# 对commit进行rebase操作
git rebase -i HEAD~4  # 最近的4个进行处理
git rebase -i commitId  # 对commitId后的所有commit进行处理
# 对弹窗commands处理
p/r/e/s/f/x/d
# 编辑后保存退出，git 会自动压缩提交历史，
wq
# 如果有冲突，记得解决冲突后，使用 
git rebase --continue # 重新回到当前的 git 压缩过程；
git rebase --abort		# 放弃压缩命令
# 推送到远程仓库
git push -f
```

将某一段commit粘贴到另一个分支

```shell
git rebase [startpoint] [endpoint]  --onto [branchName]

# 对要合并分支处理rebase
git checkout develop
git rebase  90bc0045b^ 5de0da9f2 --onto master
git log  # 最新commit为0c72e64...
git status  # 当前HEAD处于游离状态，指向内容正确但是master分支未改变，需要将master指向需最新commit
# 对master处理
git checkout master
git reset --hard  0c72e64
```

### 删除

```python
# 删除一个已合并的分支，注意，无法删除当前所在的分支，需要切换到其它分支，才能删除
git branch -d 分支名

# 如果分支还没有被合并，删除分支将会丢失修改。如果要强行删除，需要使用如下命令：
git branch -D 分支名
```

### 暂存

```python
# 保存
git stash [save "save message"]  # 保存当前工作现场,添加备注，方便查找，也可不添加备注

# 查看
git stash list  # 查看stash了哪些存储
git stash show # 显示做了哪些改动，默认show第一个存储stash@{0},显示其他存贮，后面加stash@{$num}
git stash show -p # 显示第一个存储的改动，如果想显示其他存存储，命令：git stash show  stash@{$num}  -p 

# 恢复
git stash apply   # 应用某个存储,但不会把存储从存储列表中删除，默认使用第一个存储,即stash@{0}，如果要使用其他个，git stash apply stash@{$num} 
git stash pop  # 恢复之前缓存的工作目录，将缓存堆栈中的对应stash删除，并将对应修改应用到当前的工作目录下,默认为第一个stash,即stash@{0}，如果要应用并删除其他stash，git stash pop stash@{$num} 

# 删除
git stash drop stash@{$num} # 丢弃stash@{$num}存储，从列表中删除这个存储
git stash clear # 删除所有缓存的stash
```

## 远程仓库


### SSH

Git通信协议

```
Git支持多种协议，包括SSH, https协议

使用ssh协议速度快，一次配置无扰使用
使用https速度较慢，而且每次通过终端上传代码到远程仓库时，都必须输入账号密码
```
配置SSH密钥对

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
# 设置系统级别签名
git config --global user.name [AAA]
git config --global user.email [邮箱地址]
cat .gitconfig

# 查看全局设置
git config --global --list 				

# 取消全局配置
git config --global --unset user.name	
git config --global --unset user.email

# 项目级别签名
cd 项目文件夹
git config user.name [AAA]
git config user.email [邮箱地址]
cat .git/config
```

### 关联

新建仓库

```shell 
git clone xxx  # xxx表示地址
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

已存在的文件

```shell
cd existing_folder
git init
git remote add origin xxx  # xxx表示地址
git add .
git commit -m "Initial commit"
git push -u origin master
```

已存在仓库

```shell
cd existing_repo
git remote rename origin old-origin
git remote add origin xxx  # xxx表示地址
git push -u origin --all
git push -u origin --tags
```

一份代码多个仓库

```shell
# 方法一：需要push两次，但是优点是可以pull两次
git remote add origin2 地址2  # 在gitA项目中添加另一个gitB远程的地址，origin2可以自定义
git pull origin2 master --allow-unrelated-histories   # 先拉取gitB地址上的数据，allow-unrelated-histories是为了解决冲突
git push origin2 master  # 在gitA项目中把项目内容同步到gitB地址中
# --实现推送--
git push origin  master 
git push origin2 master
# --删除--
git remote -v  // 查看此时的包括两个远程地址
git remote rm origin2  // 删除gitB的远程地址
git remote -v  //此时应该只有gitA的远程地址

# 方法二：push一次
git remote set-url --add origin 地址   # 给origin添加一个远程push地址，这样一次push就能同时push到两个地址上面
git remote -v # 查看是否多了一条push地址（这个可不执行）
git push origin master -f    # 一份代码就可以提交到两个git仓库上了，如果第一次推不上去代码，可以使用强推的方式
# --实现推送--
git push origin master
# --删除--
git remote set-url --delete origin 地址

# 方法三：直接修改.git/config
# --单仓库--
[remote "origin"]
	url = 地址
	fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
	remote = origin
	merge = refs/heads/master
# --多仓库--
[remote "origin"]
	url = 地址
	fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
	url = 地址2
	remote = origin
	merge = refs/heads/master
```

### 协同

操作远程

```shell
git remote show origin  # 查看远程分支情况
git remote update  # 更新本地追踪显示的远程分支

git remote prune origin --dry-run  # 查看远程哪些分支需要清理
git remote prune origin		# 清除无效的远程追踪分支

git remote rename pb paul  # 修改远程仓库pb为paul
git remote remove paul  # 删除远程仓库paul
```

远程->本地

```shell
git fetch origin master  # 更新本地仓库中origin/master对远程仓库origin/master的下载引用
git log -p master..origin/master  # 比对本地master分支和origin/master分支区别
git merge origin/master  # 对本地仓库的master和origin/master进行合并


git pull origin master # 从远程获取最新版本并合并远程分支到当前分支
git pull --rebase  # 把你的本地当前分支里的每个提交(commit)取消掉，并且把它们临时 保存为补丁(patch)(这些补丁放到".git/rebase"目录中),然后把本地当前分支更新 为最新的"origin"分支，最后把保存的这些补丁应用到本地当前分支上。
```

本地->远程

```python
# 远程有此分支
git checkout -b f1 origin/f1  # 跟踪拉去远程分支f1，在本地起名为f1，并切换至分支f1
git branch --set-upstream-to=origin/develop dev  # 本地分支与远程分支未建立关联，需要先建立本地与远程分支的链接关系再拉取

# 远程无此分支
git push --set-upstream origin f1  # 若远程不存在此分支,创建新的远程分支，方法一：追踪push和pull
git puh origin f1  # 方法二，只追踪push

git push  		# 推送本地代码，更新远程引用和相关对象

git push --force  # 强制本地覆盖远程

git push origin --delete serverfix  # 本地推送删除远程分支(服务器会暂存一段儿时间待垃圾回收，故异常删除时可恢复)
```

### 忽略

- 概述

`.gitignore`文件提供了可以忽略提交的文件配置。

GitHub的示例配置文件：[https://github.com/github/gitignore](https://github.com/github/gitignore)

- 忽略规则：

在git中如果想忽略掉某个文件，不让这个文件提交到版本库中，可以使用修改根目录中` .gitignore` 文件的方法。

```python
# 此为注释 – 将被 Git 忽略
*.sample 　　 # 忽略所有.sample 结尾的文件
!lib.sample 　　 # 但 lib.sample 除外
/TODO 　　 # 仅仅忽略项目根目录下的 TODO 文件，不包括 subdir/TODO
build/ 　　 # 忽略 build/ 目录下的所有文件
doc/*.txt 　　# 会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
```

- 规则不生效的解决办法

把某些目录或文件加入忽略规则，按照上述方法定义后发现并未生效，原因是.gitignore只能忽略那些原来没有被追踪的文件，如果某些文件已经被纳入了版本管理中，则修改.gitignore是无效的。那么解决方法就是先把本地缓存删除（改变成未被追踪状态），然后再提交：

```shell
git rm -r --cached .
git add .
git commit -m 'update .gitignore'
```

