# adb命令

## 安装

ubuntu

```shell
sudo apt install adb
```

mac

```shell
brew install android-platform-tools --cask
```

## 使用

```shell
# 查看设备
adb devices

# 重启服务
adb kill-server

# 从安卓推送文件到PC
adb pull <安卓设备文件路径> <PC文件路径>

# 从pc推送文件到安卓
adb push <pc文件路径> <安卓设备文件路径>

# 进入安卓终端
adb shell

# 安装软件
adb install -r <apk文件名>
```

