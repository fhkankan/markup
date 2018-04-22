# mac

## brew

>类似ubuntu中的apt-get包管理器

命令行安装

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

使用

```
# 安装软件
brew install xxx
# 卸载软件
brew uninstall xxx
# 升级
brew upgrade xxxs
# 更新
brew update
# 直接安装软件
brew search /xxx*/
# 展示
brew list xxx
# 帮助
brew help
```

## 系统文件

```
# 显示
defaults write com.apple.finder AppleShowAllFiles TRUE
killall Finder
# 关闭
defaults write com.apple.finder AppleShowAllFiles FALSE
killall Finder
```

