# Git基础

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
git status -s # 显示缩略信息
```

查看日志

```python
git log  # 产看历史版本的所有信息（包括用户名和日期）
git log --graph --pretty=oneline  # 用带参数的git log，输出的信息会短一些

git log --oneline  # 美化后的日志输出
git log --stat  # 产看提交的简略统计信息

git log -p -2  # 查看最近2次提交的引入差异
git log --since="2021-01-01" --until="2021-06-06" # 查看一段儿时间的日志

git reflog (HEAD@{移动到当前版本需要多少步})

# 搜索
git log -S ZLIB_BUF_MAX --oneline  # 想找到ZLIB_BUF_MAX常量是什么时候引入的，-S选项来显示新增和删除该字符串的提交
git log -L :git_deflate_bound:zlib.c  # 想查看zlib.c文件中git_deflate_bound函数的每一次变更        
```

查看差异

```python
git diff [文件名]  # 将工作区中的文件和暂存区的进行比较
git diff [本地库历史版本] [文件名]  # 将工作区中的文件和本地库历史记录比较，不带文件名的话，会比较多个文件
git diff  # 查看当前与暂存区之间的文件差异
git diff --staged  # 查看已暂存与最后一次提交的文件差异

# 说明：
# 减号表示： 本地仓库的代码
# 加号表示： 工作区的代码
```

查看其他

```python
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

定位bug提交

```shell
 # 启动二分法查找
git bisect start 

# 告诉git当前commit是good/bad，git会自动调整到二分处理范围的那个commit
git bisect bad  # 告诉系统当前所在的提交是有问题的
git bisect good <good_commit>	# 告诉系统已知的最后一次正常状态是哪次提交
# git会发现在good和bad之间检出中间的提交，此时执行测试，查看问题是否还存在
# 若不存在问题，说明问题在这个提交之后，告诉git，然后git会继续寻找
git bisect good
# 若存在问题，说明问题这个提交及之前，git会继续缩小范围
git bisect bad
# 直到缩减范围至一个good与bad紧挨时可以确定产生bad的commit

# 重置HEAD指针到最开始的位置
git bisect reset
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
git mv file_from file_to # 移动或重命名一个文件、目录或符号链接
```

- 删除

```python
git rm 	文件	# 从工作区和索引中删除文件
git branch -d 分支名  # 删除一个已合并的分支(非当前分支)
git branch -D 分支名  # 强行删除分支(未合并则丢失修改)

git clean  # 去除冗余文件或清理工作目录
git clean -f -d  # 移除工作目录中所有未追踪的文件已经空的子目录
```

- 记录

```shell
git commit -m '修改注释'    # 记录变更到仓库
git commit --amend  # 若是上次commit后未push，直接使用则修改注释重新提交，添加其他文件使用则合并上次的commit使用这次的注释
git reset --soft HEAD^  # 撤销上次commit，不撤销add，再次commit时作用等同amend
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

git tag -a tag-name -m tag-info  # 创建附注标签
git tag -a tag_name commit-id  # 对以往的记录进行标签

git tag tag-name  			# 创建轻量标签/查看标签内容
git tag tag-name commit-id  # 对以往的记录进行标签 

git pull --tags  			# 拉取所有标签

git push origin tag-name 	# 推送某个标签
git push --tags			 	# 推送所有标签

git tag -d tage-name		# 删除本地标签
git push origin :refs/tags/tag-name  # 删除远程标签

git checkout tag-name  # 签出标签指向的版本
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
git branch -a			# 所有分支

git branch  			# 本地分支， 当前分支有*
git branch -v 		# 查看每个分支的最后一次提交
git branch --merged # 查看已经合并到当前分支的分支
git branch --no-merged # 查看尚未合并到当前分支的分支

git branch -r  		# 远程分支
git remote show origin  # 查看远程分支详细情况
```
### 重命名

```shell
# 重命名远程分支对应的本地分支
git branch -m old-local-branch-name new-local-branch-name

# 删除远程分支
git push origin  :old-local-branch-name

# 上传新命名的本地分支
git push origin  new-local-branch-name
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

先合并后解决冲突

<img src="/Users/henry/Markup/Git/images/分叉的提交历史.png" alt="分叉的提交历史" style="zoom:50%;" />

<img src="/Users/henry/Markup/Git/images/通过合并操作来整合分叉的历史.png" alt="通过合并操作来整合分叉的历史" style="zoom:50%;" />

指令

```shell
git checkout master  # 切换至要合并到的分支master
git merge develop  # 合并需要合并的分支到当前分支
# ---有冲突时，解决冲突
git status  # 查看冲突文件
git add .
git commit -m 'merge ...'
# ---完成合并
git push master
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
在主分支进行冲突解决
```
### 撤销合并
假设现在在一个主题分支上工作，不小心将其合并到 master 中，有两种方法解决
![撤销合并-意外的合并提交](/Users/henry/Markup/Git/images/撤销合并-意外的合并提交.png)

- 修复引用

如果这个不想要的合并提交至存在于本地仓库，最简单且最好的方法就是移动分支到想要它指向的地方。

```shell
git reset --hard HEAD~
```

<img src="/Users/henry/Markup/Git/images/撤销合并-reset之后的历史.png" alt="撤销合并-reset之后的历史" style="zoom:50%;" />

> 注意

这个方法的缺点是它会重写历史，在一个共享的仓库中这会造成问题的。如果其他人已经有你将要重写的提交，你应当避免使用 reset。 如果有任何其他提交在合并之后创建了，那么这个方法也会无效；移动引用实际上会丢失那些改动。

- 还原提交

如果移动分支指针并不适合你，Git 给你一个生成一个新提交的选项，提交将会撤消一个已存在提交的所有修改。 Git 称这个操作为“还原”。

```shell
git revert -m 1 HEAD

# -m 1 标记指出 “mainline” 需要被保留下来的父结点。 当你引入一个合并到 HEAD（git merge topic），新提交有两个父结点：第一个是 HEAD（C6），第二个是将要合并入分支的最新提交（C4）。 在本例中，我们想要撤消所有由父结点 #2（C4）合并引入的修改，同时保留从父结点 #1（C6）开始的所有内容。
```

有还原提交的历史看起来像这样

<img src="/Users/henry/Markup/Git/images/撤销合并-revert后的历史.png" alt="撤销合并-revert后的历史" style="zoom:50%;" />

新的提交 ^M 与 C6 有完全一样的内容，所以从这儿开始就像合并从未发生过，除了“现在还没合并”的提交依然在 HEAD 的历史中。 如果你尝试再次合并 topic 到 master Git 会感到困惑

```shell
git merge topic
# Already up-to-date
```

topic 中并没有东西不能从 master 中追踪到达。 更糟的是，如果你在 topic 中增加工作然后再次合并，Git 只会引入被还原的合并 之后 的修改。

<img src="/Users/henry/Markup/Git/images/撤销合并-含有坏掉合并的历史.png" alt="撤销合并-含有坏掉合并的历史" style="zoom:50%;" />

解决这个最好的方式是撤消还原原始的合并，因为现在你想要引入被还原出去的修改，然后 创建一个新的合并提交

```shell
git revert ^M
git merge topic
```

<img src="/Users/henry/Markup/Git/images/撤销合并-在重新合并一个还原合并后的历史.png" alt="撤销合并-在重新合并一个还原合并后的历史" style="zoom:50%;" />

在本例中，M 与 ^M 抵消了。 ^^M 事实上合并入了 C3 与 C4 的修改，C8 合并了 C7 的修改，所以现在 topic 已经完全被合并了。

### 变基

- 基础指令

<img src="/Users/henry/Markup/Git/images/变基-将C4中的修改变基到C3.png" alt="将C4中的修改变基到C3" style="zoom:50%;" />

<img src="/Users/henry/Markup/Git/images/变基-master分支的快进合并.png" alt="master分支的快进合并" style="zoom:50%;" />

命令

```shell
# ---变基
git checkout develop
git rebase master 
git rebase master develop  # 包含了上面两条
# ---有冲突，则解决冲突，提交记录
git add .
git commit -m 'merge...' 
git rebase --continue
# -----
git push develop  # 更新远程仓库引用位置
git checkout master
git merge develop
git push master
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
缺点：如果变基中要打散的提交被其他人正在开发，则会在此次变基提交后，其他人提送代码时需要合并此次变基操作，同时自己在变基后推送代码也需要合并他人的处理，造成工作繁琐且记录中有多条重复记录，造成混乱。

摘录来自: Scott Chacon. “Pro Git。” Apple Books. 

摘录来自: Scott Chacon. “Pro Git。” Apple Books. 

# 应用场景
合并未推送的本地修改分支至公共分支时
在子分支进行冲突解决，注意适用于其他人没有使用变基中要改变的提交进行开发
```

- 更详细指令

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

- 合并多个commit为一个完整commit

```shell
# 1.查看提交历史
git log
# 2.对commit进行rebase操作
git rebase -i HEAD~3  # 最近的3个进行处理
git rebase -i commitId  # 对commitId后的所有commit进行处理
# 3.对弹窗commands进行处理，按照从旧到新的顺序排列
p	xxx
s	xxx
s	xxx
# 4.编辑后保存退出，git 会自动压缩提交历史,弹窗git messge
# 5.编辑需要的commit信息，保存退出
# 6.如果有冲突，记得解决冲突后，使用 
git rebase --continue # 重新回到当前的 git 压缩过程；
git rebase --abort		# 放弃压缩命令
# 7.查看日志
git log
# 8.推送到远程仓库
git push -f
```

- 将某一段commit粘贴到另一个分支

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

- 多层分支挑拣合并

<img src="/Users/henry/Markup/Git/images/变基-从一个主题分支里再分出一个主题分支的提交历史.png" alt="变基-从一个主题分支里再分出一个主题分支的提交历史" style="zoom:50%;" />
<img src="/Users/henry/Markup/Git/images/变基-截取主题分子上的另一个主题分支然后变基到其他分支.png" alt="变基-截取主题分子上的另一个主题分支然后变基到其他分支" style="zoom:50%;" />
<img src="/Users/henry/Markup/Git/images/变基-快进合并master分支使其包含来自client分支的修改.png" alt="变基-快进合并master分支使其包含来自client分支的修改" style="zoom:50%;" />
<img src="/Users/henry/Markup/Git/images/变基-将server中的修改变基到master上.png" alt="变基-将server中的修改变基到master上" style="zoom:50%;" />
<img src="/Users/henry/Markup/Git/images/变基-最终提交的历史.png" alt="变基-最终提交的历史" style="zoom:50%;" />

命令
```shell
# Figure 32
git checkout client
git rebase --onto master server client  # 取出 client 分支，找出它从 server 分支分歧之后的补丁， 然后把这些补丁在 master 分支上重放一遍，让 client 看起来像直接基于 master 修改一样
# Figure 33
git checkout master
git merge client  
# Figure 34
git rebase master server  # 省去git checkout server,git rebase master
# Figure 35
git checkout master
git merge server
```

### 拣选

`git cherry-pick`命令用来获得在单个提交中引入的变更，然后尝试将作为一个新的提交引入到你当前分支上。 从一个分支单独一个或者两个提交而不是合并整个分支的所有变更是非常有用的。

<img src="/Users/henry/Markup/Git/images/拣选-拣选之前的示例历史.png" alt="拣选-拣选之前的示例历史" style="zoom:50%;" />
<img src="/Users/henry/Markup/Git/images/拣选-拣选主题分支中的一个提交后的历史.png" alt="拣选-拣选主题分支中的一个提交后的历史" style="zoom:50%;" />

命令

```shell
git checkout master
git cherry-pick e43a6
```

其他命令

```shell
git checkout master
git cherry-pick  希望合并的commitId   # 合并单一提交

git cherry-pick (commitId1..commitId100)  # 合并多个，commitId1为最老提交，commitId100为最新提交，左开右闭
git cherry-pick (commitId1^..commitId100)  # 合并多个，commitId1为最老提交，commitId100为最新提交，左闭右闭
```

### 删除

```python
# 删除一个已合并的分支，注意，无法删除当前所在的分支，需要切换到其它分支，才能删除
git branch -d 分支名

# 如果分支还没有被合并，删除分支将会丢失修改。如果要强行删除，需要使用如下命令：
git branch -D 分支名

# 本地推送删除远程分支(服务器会暂存一段儿时间待垃圾回收，故异常删除时可恢复)
git push origin --delete serverfix  
```

### 暂存

```python
# 保存
git stash   		# 不添加备注
git stash save demo # 添加备注，仅为了区分显示
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

# 从贮藏创建一个分支
git stash branch testchanges  # 以指定的分支名创建一个新分支，检出贮藏工作时所在的提交，重新在那应用工作，然后在应用成功后丢弃贮藏
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
git clone xxx [文件夹名字] # xxx表示地址，有文件夹名字时将代码写入，无时自动命名
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

- 一份代码多个仓库

多关联，即拉取也推送

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
```
多推送，一拉取
```shell
# 方法二：push一次，注意备用库中不能改主库中相同代码，避免冲突无法推送
git remote set-url --add origin 地址   # 给origin添加一个远程push地址，这样一次push就能同时push到两个地址上面
git remote -v # 查看是否多了一条push地址（这个可不执行）
git push origin master -f    # 一份代码就可以提交到两个git仓库上了，如果第一次推不上去代码，可以使用强推的方式
# --实现推送--
git push
# --删除--
git remote set-url --delete origin 地址
```
随意配置

```shell
# 方法三：直接修改.git/config

# 默认
[remote "origin"]
	url = 地址
	fetch = +refs/heads/*:refs/remotes/origin/*
[branch "master"]
	remote = origin
	merge = refs/heads/master

# 多推送，多拉取
[remote "origin"]
	url = 地址
	fetch = +refs/heads/*:refs/remotes/origin/*
[remote "origin2"]
    url = 地址
    fetch = +refs/heads/*:refs/remotes/origin2/*
[branch "master"]  
	remote = origin  # 默认的推拉，若要多拉推，需手动置顶origin/origin2
	merge = refs/heads/master

# 多推送，一拉取
[remote "origin"]
	url = 地址
	fetch = +refs/heads/*:refs/remotes/origin/*
	url = 地址2
[branch "master"]  
	remote = origin  # 默认的推拉
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
git fetch origin master  # 从远程获取最新版本，不会自动合并或修改当前的工作
git log -p master..origin/master  # 比对本地master分支和origin/master分支区别
git merge origin/master  # 对本地仓库的master和origin/master进行合并

git pull origin master # 从远程获取最新版本，自动合并远程分支到到当前的工作
git pull --rebase  # 把你的本地当前分支里的每个提交(commit)取消掉，并且把它们临时 保存为补丁(patch)(这些补丁放到".git/rebase"目录中),然后把本地当前分支更新为最新的"origin"分支，最后把保存的这些补丁应用到本地当前分支上。
```

指定文件/文件夹拉取(适用于1.7.0后版本)

```
1.设置core.sparsecheckout为true
git config core.sparsechekout true
2.在.git/info/sparse-checkout文件中添加指定文件/夹
3.拉取想要的分支即可实现checkout指定文件/夹
git pull origin master
```

本地->远程

```python
# 查看本地记录的远程信息
git remote -v  # 查看远程配置

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

## 忽略

### 本地远程

- 概述

`.gitignore`文件提供了可以忽略提交的文件配置，同时此文件会提交到git仓库中，会影响其他人。

GitHub的示例配置文件：[https://github.com/github/gitignore](https://github.com/github/gitignore)

```
# ide
.idea/
.vscode
.DS_Store
```

- 忽略规则：

在git中如果想忽略掉某个文件，不让这个文件提交到版本库中，可以使用修改根目录中` .gitignore` 文件的方法。

```python
# 此为注释 – 将被 Git 忽略
*.sample 　　 # 忽略所有.sample 结尾的文件
/TODO 　　 # 仅仅忽略项目根目录下的 TODO 文件，不包括 subdir/TODO
build/ 　　 # 忽略 build/ 目录下的所有文件
doc/*.txt 　　# 会忽略 doc/notes.txt 但不包括 doc/server/arch.txt

!lib.sample 　　 # 但 lib.sample 除外
```

- 规则不生效的解决办法

把某些目录或文件加入忽略规则，按照上述方法定义后发现并未生效，原因是.gitignore只能忽略那些原来没有被追踪的文件，如果某些文件已经被纳入了版本管理中，则修改.gitignore是无效的。那么解决方法就是先把本地缓存删除（改变成未被追踪状态），然后再提交：

```shell
git rm -r --cached .
git add .
git commit -m 'update .gitignore'
```

### 只本地

`.git/info/exclude` 设置你自己本地需要排除的文件。不会提交到版本库中去，所以不会影响他人。

```shell
# 进入/创建文件进行编辑
vim .git/info/exclude
# 输入忽略内容
*.sample 　　 # 忽略所有.sample 结尾的文件
!lib.sample 　　 # 但 lib.sample 除外
/TODO 　　 # 仅仅忽略项目根目录下的 TODO 文件，不包括 subdir/TODO
build/ 　　 # 忽略 build/ 目录下的所有文件
doc/*.txt 　　# 会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
```



