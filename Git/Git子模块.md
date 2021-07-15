# git管理子模块

当项目越来越庞大之后，不可避免的要拆分成多个子模块，我们希望各个子模块有独立的版本管理，并且由专门的人去维护，这时候我们就要用到git的submodule/subtree功能。

sumodule是引用，subtree是复制

| /                | submodule                                                    | subtree                          | 结果                               |
| :--------------- | :----------------------------------------------------------- | :------------------------------- | :--------------------------------- |
| 远程仓库空间占用 | submodule只是引用，基本不占用额外空间                        | 子模块copy，会占用较大的额外空间 | submodule占用空间较小，略优        |
| 本地空间占用     | 可根据需要下载                                               | 会下载整个项目                   | 所有模块基本都要下载，二者差异不大 |
| 仓库克隆         | 克降后所有子模块为空，需要注册及更新，同时更新后还需切换分支 | 克隆之后即可使用                 | submodule步骤略多，subtree占优     |
| 更新本地仓库     | 更新后所有子模块后指向最后一次提交，更新后需要重新切回分支，所有子模块只需一条更新语句即可 | 所有子模块需要单独更新           | 各有优劣，相对subtree更好用一些    |
| 提交本地修改     | 只需关心子模块即可，子模块的所有操作与普通git项目相同        | 提交执行命令相对复杂一些         | submodule操作更简单，submodule占优 |

## submodule

### 常用命令

```shell
git clone <repository> --recursive   	# 递归的方式克隆整个项目
git submodule add <repository> <path>   # 添加子模块
git submodule init 						# 初始化子模块
git submodule update 					# 更新子模块
git submodule foreach git pull 			# 拉取所有子模块
```

### 创建带子模块的项目

例如我们要创建如下结构的项目

```
project
  |--moduleA
  |--readme.txt
```

创建project版本库，并提交readme.txt文件

```shell
git init --bare project.git
git clone project.git project
cd project
echo "This is a project." > readme.txt
git add .
git commit -m "add readme.txt"
git push origin master
cd ..
```

创建moduleA版本库，并提交a.txt文件

```shell
git init --bare moduleA.git
git clone moduleA.git moduleA
cd moduleA
echo "This is a submodule." > a.txt
git add .
git commit -m "add a.txt"
git push origin master
cd ..
```

在project项目中引入子模块moduleA，并提交子模块信息

```shell
cd project
git submodule add moduleA.git moduleA
git status
git diff
git add .
git commit -m "add submodule"
git push origin master
cd ..
```

使用`git status`可以看到多了两个需要提交的文件：`.gitmodules`和`moduleA`。使用`git diff`可以看到这两项的内容。这里需要指出父项目的git并不会记录submodule的文件变动，它是按照commit id指定submodule的git header，所以`.gitmodules`和`moduleA`这两项是需要提交到父项目的远程仓库的。

```shell
On branch master
Your branch is up-to-date with 'origin/master'.
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)
    new file:   .gitmodules  # 指定submodule的主要信息，包括子模块的路径和地址信息
    new file:   moduleA  # 指定了子模块的commit id
```

### 克隆带子模块的项目

方法一，先clone父项目，再初始化submodule，最后更新submodule，初始化只需要做一次，之后每次只需要直接update就可以了，需要注意submodule默认是不在任何分支上的，它指向父项目存储的submodule commit id。

```shell
git clone project.git project
cd project
git submodule init
git submodule update
cd ..
```

方法二，采用递归参数`--recursive`，需要注意同样submodule默认是不在任何分支上的，它指向父项目存储的submodule commit id。

```shell
git clone project.git project --recursive
```

### 项目中修改子模块

修改子模块之后只对子模块的版本库产生影响，对父项目的版本库不会产生任何影响，如果父项目需要用到最新的子模块代码，我们需要更新父项目中submodule commit id，默认的我们使用`git status`就可以看到父项目中submodule commit id已经改变了，我们只需要再次提交就可以了。

```shell
# 进入子模块进行修改提交
cd project/moduleA
git branch
echo "This is a submodule." > b.txt
git add .
git commit -m "add b.txt"
git push origin master
# 进入项目进行提交
cd ..
git status
git diff
git add .
git commit -m "update submodule add b.txt"
git push origin master
cd ..
```

### 项目中获取子模块

更新子模块的时候要注意子模块的分支默认不是master。

```shell
cd project/moduleA
git checkout master
```

方法一：子模块更新 

```shell
cd project/moduleA
git pull
```

方法二：主目录更新

```shell
cd project
git submodule foreach git pull
```

### 项目中删除子模块

删除缓存和文件夹

```shell
git rm --cached moduleA
rm -rf moduleA
rm .gitmodules
vim .git/config
```

删除配置信息

```shell
[submodule "moduleA"]
      url = /Users/nick/dev/nick-doc/testGitSubmodule/moduleA.git
```

然后提交到远程服务器

```shell
git add .
git commit -m "remove submodule"
```

## subtree

### 常用命令

```shell
git subtree add -P <prefix> <commit>
git subtree add -P <prefix> <repository> <ref>
git subtree pull -P <prefix> <repository> <ref>
git subtree push -P <prefix> <repository> <ref>
git subtree merge -P <prefix> <commit>
git subtree split -P <prefix> [OPTIONS] [<commit>]
```

参数

```shell
-q | --quiet
-d | --debug
-P <prefix> | --prefix=<prefix>      # 引用库对应的本地目录,prefix可以为空
-m <message> | --message=<message>   # 适用于add/pull/merge子命令。设置产生的合并提交的说明文本
```
squash
```shell
--squash                             
#. 适用于add/pull/merge子命令。先合并引用库的更新记录，将合并结果并到主项目中。
# 使用此选项时，把 subtree 子项目的更新记录进行合并，再合并到主项目中：subtree add/pull 操作的结果对应两个commit:一个是squash了子项目的历史记录，一个是 Merge 到主项目中。优点是可以让主项目历史记录很规整，缺点是在子项目需要 subtree pull 的时候，经常需要处理冲突，甚至每次subtree pull的时候都需要重复处理同样的冲突。
# 如果不加 --squash 参数，主项目会合并子项目本身所有的 commit 历史记录。优点是子项目更新的时候，subtree pull 很顺利， 能够自动处理已解决过的冲突。缺点是子项目会污染主项目。
# 一个更好的解决方案是：单独建一个分支进行--no-squash的subtree更新，然后再--squash合并到主分支。每次在此分支做操作前都需要先把主分支合并进来。参考：http://www.fwolf.com/blog/post/246.
```

split

```shell
# split子命令选项：
--annotate=<annotation>              # 创建合成历史时有可能形成内容不同但提交信息完全相同的提交版本，使用这个选项在每个提交消息前加上此前缀用来区分。
-b <branch> | --branch=<branch>      # 创建合成的提交历史时，创建此参数指定的新分支包含生成的合成历史。<branch>必须是还不存在的。
--onto=<onto>
--rejoin
--ignore-joins
```

### 创建带子模块的项目

例如我们要创建如下结构的项目

```
project
  |--moduleA
  |--readme.txt
```

创建project版本库，并提交readme.txt文件

```shell
git init --bare project.git
git clone project.git project
cd project
echo "This is a project." > readme.txt
git add .
git commit -m "add readme.txt"
git push origin master
cd ..
```

创建moduleA版本库，并提交a.txt文件

```shell
git init --bare moduleA.git
git clone moduleA.git moduleA
cd moduleA1
echo "This is a submodule." > a.txt
git add .
git commit -m "add a.txt"
git push origin master
cd ..
```

在project项目中引入子模块moduleA，并提交子模块信息

```shell
git remote add moduleA http://xx.git  # 非必须，便于简化下步操作
git subtree add --prefix=moduleA moduleA master --squash # git subtree add --prefix=<S项目的相对路径> <S项目远程库仓库地址 | S项目远程库别名> <分支> --squash
git push  # 提交
```

### 克隆带子模块的项目

```
git clone project.git project
```

### 项目中修改子模块

```shell
# git subtree push --prefix=<S项目的相对路径> <S项目远程库仓库地址 | S项目远程库别名> <S项目分支>
git subtree push --prefix=moduleA moduleA master
```

每次push命令都会遍历全部的commit,当你的项目越来越大,commit的数上来的时候,等待时间就会很长。--rejoin 避免了遍历全部commit的问题.

```shell
# git subtree split --rejoin --prefix=<S项目的相对路径> --branch <临时branch>
git subtree split --rejoin --prefix=moduleA --branch srcTemp  # 提取与引用库子目录相关的变更并生成一个新的合成历史到新分支
# git push <S项目远程库仓库地址 | S项目远程库别名> srcTemp:master
git push share srcTemp:master
```

### 项目中获取子模块

```shell
# git subtree pull --prefix=<S项目的相对路径> <S项目远程库仓库地址 | S项目远程库别名> <分支> --squash
git subtree pull --prefix=src/share share master --squash
```

### 项目中删除子模块

删除缓存和文件夹

```shell
git rm --cached moduleA
rm -rf moduleA
vim .git/config
```

删除配置信息

```shell
[remote "module"]
      url = git@github.com:fhkankan/moduleA.git
	  fetch = +refs/heads/*:refs/remotes/module/*
```

然后提交到远程服务器

```shell
git add .
git commit -m "remove submodule"
```

## 文件软链接

在项目中创建其他仓库的项目的软连接，也可以实现多模块管理，需要注意，部署时，需要将所有代码模块进行同位置下载与维护

```
cd project
ln -s ../project2 lib1
ln -s ../project2 lib2
```







