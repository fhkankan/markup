import socket

"""
构造函数
"""
# 套接字构造函数 socket(family,type[,protocal])
# family：套接字家族，可使用AF_UNIX或AF_INET或AF_INET6
# type：套接字类型，可以根据是面向连接的还是非连接分为SOCK_STREAM或SOCK_DGRAM
# protocal:协议编号，一般不填，默认为0

# 参数取值含义
# socket.AF_UNIX ---> 只能够用于单一的Unix系统进程间通
# socket.AF_INET ---> 服务器之间网络通信
# socket.AF_INET6---> IPv6
# socket.SOCK_STREAM ---> 流式socket，针对TCP
# socket.SOCK_DGRAM  ---> 数据报式socket，针对UDP
# socket.SOCK_RAW    ---> 原始套接字，普通的套接字无法处理ICMP/IGMP等网络报文，而socket.SOCK_RAW可以，其次socket.SOCK_RAW也可以处理特殊的IPv4报文，此外，利用原始套接字，可以通过IP_HDRINCL套接字选项由用户构造IP头
# socket.SOCK_SEQPACKET ---> 可靠的连续数据包服务

"""
服务器端
"""
s.bind(host,port)
# 绑定地址到套接字，在AF_INET下以元组(host,port)的形式表示地址
s.listen(backlog)
# 开始TCP监听。backlog指定在拒绝连接之前，可以最大连接数量。最少是1，大部分程序设为5
s.accept()
# 被动接受TCP客户端连接，(阻塞式)等待连接的到来

"""
客户端
"""
s.connect(address)
# 主动与TCP服务器连接。一般address的格式为元组(hostname,port),若连接出错，返回socket.erro错误
s.connect_ex()
# connect函数的扩展版本，出错时返回出错代码，而不是抛出异常

"""
公共用途
"""
s.recv(bufsize,[,flag])
# 接收TCP数据，数据以字节串形式返回，bufsize指定要接收的最大数据量。flag提供有关消息的其他信息，通常可以忽略
s.send(data)
# 发送TCP数据，将data中的数据发送到连接的套接字。返回值是要发送的字节数量，该数量可能小于data的字节大小
s.sendall(data)
# 完整发送TCP数据，将data中的数据发送到连接套接字，但在返回之前会尝试发送所有数据。成功返回None,失败则抛出异常
s.recvform(bufsize,[,flag])
# 接收UDP数据，与recv()类似，但返回值是(data,address)。其中data是包含接收数据的字节串，address是发送数据的套接字地址
s.sendto(data,address)
# 发送UDP数据，将数据数据发送到套接字，address是形式为(ip,port)的元组，指定远程地址。返回值是发送的字节数
s.close()
# 关闭套接字
s.getpeername()
# 返回连接套接字的远程地址。返回值通常是元组(ipaddr,port)
s.getsockname()
# 返回套接字自己的地址。通常是元组(ipaddr,port)
s.setsockopt(level,optname,value)
# 设置给定套接字选项的值
s.getsockopt(level,optname)
# 返回套接字选项的值
s.settimeout(timeout)
# 设定套接字操作的超时时间，timeout是一个浮点数，单位是秒。值为None表示没有超时时间。一般超时时间应该在刚创建套接字时设置，因为它们可能用于连接的操作
s.gettimeout()
# 返回当时超时时间的值，单位是秒，若没有设置超时时间，返回None
s.fileno()
# 返回套接字的文件描述符
s.setblocking(flag)
# 如果flag为0，则将套接字设置为非阻塞模式，否则将套接字设置为阻塞模式(默认)，非阻塞模式下，若调用recv()没有发现任何数据，或send()调用无法立即发送数据，将引起socket.error错误
s.makefile()
# 创建一个与该套接字相关联的文件