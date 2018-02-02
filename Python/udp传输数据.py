# UDP
# 面向无连接的通信协议，不保证数据包的顺利到达，不可靠传输，但是效率高。
"""
用UPD协议发送接收数据
"""

# 导入socket模块
import socke#判断程序入口


if __name__ == '__main__':
    # 创建socket
    # socket.AF_INET表示ipv4
    # socket.SOCK_DGRAM表示UDP传输协议
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 绑定接收数据端口号
    recv_iport = ('',6789)
    s.bind(recv_iport)
    # 发送信息
    send_content = '春风绿了江南岸'
    # 对发送信息进行编码,
    # gbk为国标汉字编码
    # utf-8为国际字体编码
    send_data = send_content.encode('gbk')
    # 接收方地址,元组
    dest_iport = ('192.168.42.93',8888)
    # 发送
    s.sendto(send_data,dest_iport)
    # 接收对方发送内容
    dest_data,dest_iport = s.recvfrom(1024)
    # 对接收内容解码
    dest_content = dest_data.decode('gbk')
    print(dest_content)
    # 关闭传输
    s.close()
