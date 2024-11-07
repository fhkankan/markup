# windows命令行

## 查看

端口和进程

```shell
# 查看所有端口映射
netsh interface portproxy show all

# 查看123端口占用情况
netstat -ano | findstr "123"

# 查看进程相关信息
tasklist | findstr "PID值"

# 关闭进程
taskkill /f /pid 123
```

ip

```
ipconfig
```
