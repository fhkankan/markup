"""
客户端
"""
# 导入socket模块
import socket

# 创建socket
tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 服务器信息
server_ip = input('enter the ip of the server:')
server_port = int(input('enter the port of the server:'))
# 连接服务器
tcp_client_socket.connect((server_ip, server_port))
# 发送的数据
send_content = input('enter the data to send to the server:')
# 对数据编码
send_data = send_content.encode('gbk')
# 发送
tcp_client_socket.send(send_data)
# 接收对方的数据
recv_data = tcp_client_socket.recv(1024)
# 对数据解码
recv_content = recv_data.decode('gbk')
# 打印接收信息
print(recv_content)
# 关闭套接字
tcp_client_socket.close()


"""
服务端
"""
import socket


# 创建用户连接客户端的socket
tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 立即释放端口
tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
# 服务端绑定端口
tcp_server_socket.bind(('', 6789))
# 监听客户端信息，使服务端创建的socket由主动变被动
# 128:等待accept处理的最大链接数
tcp_server_socket.listen(128)
# 若有新的客户端来链接服务器，产生一个新套接字专为这个客户服务
# tcp_server_socket就可以专门等待其他新客户端的链接
client_socket, clientAddr = tcp_server_socket.accept()
# 接收客户端发送的数据
recv_data = client_socket.recv(1024)
# 转码
recv_content = recv_data.decode('gbk')
# 打印接收的数据
print(recv_content)
# 要发送的数据
send_content = 'the message has been received!'
send_data = send_content.encode('gbk')
client_socket.send(send_data)
# 关闭客户端服务的套接字
client_socket.close()
# 关闭与客户端的的链接
tcp_server_socket.close()














