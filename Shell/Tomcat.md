# Tomcat

## 安装

windows

```
1. 下载：http://tomcat.apache.org/
2. 安装：解压压缩包即可。
	注意：安装目录建议不要有中文和空格
3. 卸载：删除目录就行了
```

## 使用

启动

```shell
# windows
<Tomcat-installation-directory>/bin/startup.bat
# linux
<Tomcat-installation-directory>/bin/startup.sh
		
# 可能遇到的问题：
1. 黑窗口一闪而过：
	* 原因： 没有正确配置JAVA_HOME环境变量
	* 解决方案：正确配置JAVA_HOME环境变量

2. 启动报错：
	1. 暴力：找到占用的端口号，并且找到对应的进程，杀死该进程
		* netstat -ano
	2. 温柔：修改自身的端口号
		* conf/server.xml
		* <Connector port="8888" protocol="HTTP/1.1"
	             connectionTimeout="20000"
	             redirectPort="8445" />
		* 一般会将tomcat的默认端口号修改为80。80端口号是http协议的默认端口号。
			* 好处：在访问时，就不用输入端口号
```

关闭

```shell
# 正常关闭：
bin/shutdown.bat
ctrl+c
# 强制关闭：
点击启动窗口的×
```

## 配置

部署项目的方式

```
1. 直接将项目放到webapps目录下即可。
	* /hello：项目的访问路径-->虚拟目录
	* 简化部署：将项目打成一个war包，再将war包放置到webapps目录下。
		* war包会自动解压缩

2. 配置conf/server.xml文件
	在<Host>标签体中配置
	<Context docBase="D:\hello" path="/hehe" />
	* docBase:项目存放的路径
	* path：虚拟目录

3. 在conf\Catalina\localhost创建任意名称的xml文件。在文件中编写
	<Context docBase="D:\hello" />
	* 虚拟目录：xml文件的名称
```

动态项目目录结构

```
-- 项目的根目录
	-- WEB-INF目录：
		-- web.xml：web项目的核心配置文件
		-- classes目录：放置字节码文件的目录
		-- lib目录：放置依赖的jar包
```

