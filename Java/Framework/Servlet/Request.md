# Request

## 原理

`request,response`对象是由服务器创建的。我们来使用它们

`request`对象是来获取请求消息，`response`对象是来设置响应消息

## 体系结构

```
ServletRequest		--	接口
	|	继承
HttpServletRequest	-- 接口
	|	实现
org.apache.catalina.connector.RequestFacade 类(tomcat)
```

## 常用函数

获取请求行

```java
// GET /day14/demo1?name=zhangsan HTTP/1.1
String getMethod() // 获取请求方式:GEET
String getContextPath() // 获取虚拟目录:/day14
String getServletPath() // 获取Servlet路径:/demo1
String getQueryString() // 获取get方式请求参数：name=zhangsan
String getRequestURI()	// 获取请求URI：/day14/demo1
StringBuffer getRequestURL()  // 获取请求URL:http://localhost/day14/demo1
String getProtocol() // 获取协议及版本：HTTP/1.1
String getRemoteAddr() // 获取客户机的IP地址：
```

获取请求头

```java
String getHeader(String name) // 通过请求头的名称获取请求头的值
Enumeration<String> getHeaderNames() // 获取所有的请求头名称
```

获取请求体

```java
// 步骤：
// 1. 获取流对象
BufferedReader getReader() // 获取字符输入流，只能操作字符数据
ServletInputStream getInputStream() // 获取字节输入流，可以操作所有类型数据
// 2. 再从流对象中拿数据
  
String getContentType() // 返回请求主体的 MIME 类型，如果不知道类型则返回 null。
```

获取请求参数通用方式

```java
String getParameter(String name)	// 根据参数名称获取参数值    username=zs&password=123
String[] getParameterValues(String name) // 根据参数名称获取参数值的数组  hobby=xx&hobby=game
Enumeration<String> getParameterNames():获取所有请求的参数名称
Map<String,String[]> getParameterMap():获取所有参数的map集合

// 中文乱码问题：
get方式：tomcat 8 已经将get方式乱码问题解决了
post方式：会乱码
		* 解决：在获取参数前，设置request的编码request.setCharacterEncoding("utf-8");
```

请求转发

```java
// 一种在服务器内部的资源跳转方式
// 特点
// 1. 浏览器地址栏路径不发生变化
// 2. 只能转发到当前服务器内部资源中。
// 3. 转发是一次请求

// 步骤
// 1. 通过request对象获取请求转发器对象：
RequestDispatcher getRequestDispatcher(String path)
// 2. 使用RequestDispatcher对象来进行转发：
forward(ServletRequest request, ServletResponse response) 
```

共享数据

```java
// 域对象：一个有作用范围的对象，可以在范围内共享数据
// request域：代表一次请求的范围，一般用于请求转发的多个资源中共享数据
void setAttribute(String name,Object obj)  // 存储数据
Object getAttitude(String name)  // 通过键获取值
void removeAttribute(String name)  // 通过键移除键值对
```

获取ServletContext

```java
ServletContext getServletContext()
```

## 使用

### 请求行

```java
package cn.itcast.web.request;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * 演示Request对象获取请求行数据
 */

@WebServlet("/requestDemo1")
public class RequestDemo1 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //1. 获取请求方式 ：GET
        String method = request.getMethod();
        System.out.println(method);
        //2.(*)获取虚拟目录：/day14
        String contextPath = request.getContextPath();
        System.out.println(contextPath);
        //3. 获取Servlet路径: /demo1
        String servletPath = request.getServletPath();
        System.out.println(servletPath);
        //4. 获取get方式请求参数：name=zhangsan
        String queryString = request.getQueryString();
        System.out.println(queryString);
        //5.(*)获取请求URI：/day14/demo1
        String requestURI = request.getRequestURI();
        StringBuffer requestURL = request.getRequestURL();
        System.out.println(requestURI);
        System.out.println(requestURL);
        //6. 获取协议及版本：HTTP/1.1
        String protocol = request.getProtocol();
        System.out.println(protocol);
        //7. 获取客户机的IP地址：
        String remoteAddr = request.getRemoteAddr();
        System.out.println(remoteAddr);
    }
}
```

### 请求头

```java
package cn.itcast.web.request;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Enumeration;

@WebServlet("/requestDemo2")
public class RequestDemo2 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //1.获取所有请求头名称
        Enumeration<String> headerNames = request.getHeaderNames();
        //2.遍历
        while(headerNames.hasMoreElements()){
            String name = headerNames.nextElement();
            //根据名称获取请求头的值
            String value = request.getHeader(name);
            System.out.println(name+"---"+value);
        }
      
      	String agent = request.getHeader("user-agent");
        //判断agent的浏览器版本
        if(agent.contains("Chrome")){
            //谷歌
            System.out.println("谷歌来了...");
        }else if(agent.contains("Firefox")){
            //火狐
            System.out.println("火狐来了...");
        }
      
      	String referer = request.getHeader("referer");
        System.out.println(referer);//http://localhost/day14/login.html

        //防盗链
        if(referer != null ){
            if(referer.contains("/day14")){
                //正常访问
               // System.out.println("播放电影....");
                response.setContentType("text/html;charset=utf-8");
                response.getWriter().write("播放电影....");
            }else{
                //盗链
                //System.out.println("想看电影吗？来优酷吧...");
                response.setContentType("text/html;charset=utf-8");
                response.getWriter().write("想看电影吗？来优酷吧...");
            }
        }

    }
}

```

### 请求体

```java
package cn.itcast.web.request;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;

@WebServlet("/requestDemo5")
public class RequestDemo5 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //获取请求消息体--请求参数

        //1.获取字符流
        BufferedReader br = request.getReader();
        //2.读取数据
        String line = null;
        while((line = br.readLine()) != null){
            System.out.println(line);
        }
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    }
}
```

### 请求参数

```java
package cn.itcast.web.request;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Map;
import java.util.Set;

@WebServlet("/requestDemo6")
public class RequestDemo6 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //post 获取请求参数
				// 设置流的编码
        request.setCharacterEncoding("utf-8");
        
        //根据参数名称获取参数值
        String username = request.getParameter("username");
       /* System.out.println("post");
        System.out.println(username);*/

       //根据参数名称获取参数值的数组
        String[] hobbies = request.getParameterValues("hobby");
        /*for (String hobby : hobbies) {
            System.out.println(hobby);
        }*/

        //获取所有请求的参数名称

        Enumeration<String> parameterNames = request.getParameterNames();
        /*while(parameterNames.hasMoreElements()){
            String name = parameterNames.nextElement();
            System.out.println(name);
            String value = request.getParameter(name);
            System.out.println(value);
            System.out.println("----------------");
        }*/

        // 获取所有参数的map集合
        Map<String, String[]> parameterMap = request.getParameterMap();
        //遍历
        Set<String> keyset = parameterMap.keySet();
        for (String name : keyset) {
            
            //获取键获取值
            String[] values = parameterMap.get(name);
            System.out.println(name);
            for (String value : values) {
                System.out.println(value);
            }
            System.out.println("-----------------");
        }
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //get 获取请求参数
				/*
        //根据参数名称获取参数值
        String username = request.getParameter("username");
        System.out.println("get");
        System.out.println(username);*/

        this.doPost(request,response);
    }
}
```

###  转发共享

`RequestDemo8.java`

```java
package cn.itcast.web.request;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/requestDemo8")
public class RequestDemo8 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        System.out.println("demo8888被访问了。。。");
        //转发到demo9资源
				/*
        RequestDispatcher requestDispatcher = request.getRequestDispatcher("/requestDemo9");
        requestDispatcher.forward(request,response);
        */
        //存储数据到request域中
        request.setAttribute("msg","hello");

        request.getRequestDispatcher("/requestDemo9").forward(request,response);
        //request.getRequestDispatcher("http://www.itcast.cn").forward(request,response);
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

        this.doPost(request,response);
    }
}
```

`RequestDemo9.java`

```java
package cn.itcast.web.request;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/requestDemo9")
public class RequestDemo9 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //获取数据
        Object msg = request.getAttribute("msg");
        System.out.println(msg);
        System.out.println("demo9999被访问了。。。");
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doPost(request,response);
    }
}
```

### ServletContext

```java
package cn.itcast.web.request;

import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/requestDemo10")
public class RequestDemo10 extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        ServletContext servletContext = request.getServletContext();
        System.out.println(servletContext);
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doPost(request,response);
    }
}
```

## 案例

### 需求

```
1.编写login.html登录页面
	username & password 两个输入框
2.使用Druid数据库连接池技术,操作mysql，day14数据库中user表
3.使用JdbcTemplate技术封装JDBC
4.登录成功跳转到SuccessServlet展示：登录成功！用户名,欢迎您
5.登录失败跳转到FailServlet展示：登录失败，用户名或密码错误
```

### 项目准备

```shell
# 1.创建项目，导入html，配置文件，jar包
- Test
	- src
		- itcast
			- dao
				UserDao.java
			- domain
				User.java
			- test
				BeanUtilsTest.java
				UserDaoTest.java
			- util
				JDBCUtils.java
			- web
				- servlet
					FailServlet.java
					LoginServlet.java
					SuccessServlet.java
	- web
		- WEB-INF
			- lib
				...
		index.jsp
    login.html

# 2.创建数据库和表
CREATE DATABASE day14;
USE day14;
CREATE TABLE USER(

	id INT PRIMARY KEY AUTO_INCREMENT,
	username VARCHAR(32) UNIQUE NOT NULL,
	PASSWORD VARCHAR(32) NOT NULL
);
```

### 代码实现

创建包`cn.itcast.domain`,创建类User

```java
package cn.itcast.domain;

public class User {

    private int id;
    private String username;
    private String password;

    private String gender;

    public void setHehe(String gender){
        this.gender = gender;
    }

    public String getHehe(){
        return gender;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", password='" + password + '\'' +
                ", gender='" + gender + '\'' +
                '}';
    }
}

```

创建包`cn.itcast.util`,编写工具类JDBCUtils

```java
package cn.itcast.util;

import com.alibaba.druid.pool.DruidDataSourceFactory;

import javax.sql.DataSource;
import javax.xml.crypto.Data;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.SQLException;
import java.util.Properties;

/**
 * JDBC工具类 使用Durid连接池
 */
public class JDBCUtils {

    private static DataSource ds ;

    static {

        try {
            //1.加载配置文件
            Properties pro = new Properties();
            //使用ClassLoader加载配置文件，获取字节输入流
            InputStream is = JDBCUtils.class.getClassLoader().getResourceAsStream("druid.properties");
            pro.load(is);

            //2.初始化连接池对象
            ds = DruidDataSourceFactory.createDataSource(pro);

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 获取连接池对象
     */
    public static DataSource getDataSource(){
        return ds;
    }


    /**
     * 获取连接Connection对象
     */
    public static Connection getConnection() throws SQLException {
        return  ds.getConnection();
    }
}

```

创建包`cn.itcast.dao`,创建类UserDao,提供login方法

```java
package cn.itcast.dao;

import cn.itcast.domain.User;
import cn.itcast.util.JDBCUtils;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import org.springframework.jdbc.core.JdbcTemplate;

/**
 * 操作数据库中User表的类
 */
public class UserDao {

    //声明JDBCTemplate对象共用
    private JdbcTemplate template = new JdbcTemplate(JDBCUtils.getDataSource());

    /**
     * 登录方法
     * @param loginUser 只有用户名和密码
     * @return user包含用户全部数据,没有查询到，返回null
     */
    public User login(User loginUser){
        try {
            //1.编写sql
            String sql = "select * from user where username = ? and password = ?";
            //2.调用query方法
            User user = template.queryForObject(sql,
                    new BeanPropertyRowMapper<User>(User.class),
                    loginUser.getUsername(), loginUser.getPassword());
            return user;
        } catch (DataAccessException e) {
            e.printStackTrace();//记录日志
            return null;
        }
    }
}
```

编写`cn.itcast.web.servlet.LoginServlet`类

```java
package cn.itcast.web.servlet;

import cn.itcast.dao.UserDao;
import cn.itcast.domain.User;
import org.apache.commons.beanutils.BeanUtils;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Map;


@WebServlet("/loginServlet")
public class LoginServlet extends HttpServlet {


    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        //1.设置编码
        req.setCharacterEncoding("utf-8");
       /* //2.获取请求参数
        String username = req.getParameter("username");
        String password = req.getParameter("password");
        //3.封装user对象
        User loginUser = new User();
        loginUser.setUsername(username);
        loginUser.setPassword(password);*/
        //2.获取所有请求参数
        Map<String, String[]> map = req.getParameterMap();
        //3.创建User对象
        User loginUser = new User();
        //3.2使用BeanUtils封装
        try {
            BeanUtils.populate(loginUser,map);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }


        //4.调用UserDao的login方法
        UserDao dao = new UserDao();
        User user = dao.login(loginUser);

        //5.判断user
        if(user == null){
            //登录失败
            req.getRequestDispatcher("/failServlet").forward(req,resp);
        }else{
            //登录成功
            //存储数据
            req.setAttribute("user",user);
            //转发
            req.getRequestDispatcher("/successServlet").forward(req,resp);
        }

    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        this.doGet(req,resp);
    }
}

```

编写FailServlet和SuccessServlet类

```java
package cn.itcast.web.servlet;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

// FailServlet
@WebServlet("/failServlet")
public class FailServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //设置编码
        response.setContentType("text/html;charset=utf-8");
        //输出
        response.getWriter().write("登录失败，用户名或密码错误");
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doPost(request,response);
    }
}


// SuccessServlet
@WebServlet("/successServlet")
public class SuccessServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //获取request域中共享的user对象
        User user = (User) request.getAttribute("user");

        if(user != null){
            //给页面写一句话

            //设置编码
            response.setContentType("text/html;charset=utf-8");
            //输出
            response.getWriter().write("登录成功！"+user.getUsername()+",欢迎您");
        }
    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doPost(request,response);
    }
}

```

`index.jsp`

```jsp
<%--
  Created by IntelliJ IDEA.
  User: fqy
  Date: 2018/6/4
  Time: 16:43
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
  <head>
    <title>$Title$</title>
  </head>
  <body>
  $END$
  </body>
</html>

```

`login.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <form action="/test/loginServlet" method="post">
        用户名:<input type="text" name="username"> <br>
        密码:<input type="password" name="password"><br>

        <input type="submit" value="登录">

    </form>
</body>
</html>
```

### BeanUtils

用于封装封装数据，标准的Java类

要求
```
1. 类必须被public修饰
2. 必须提供空参的构造器
3. 成员变量必须使用private修饰
4. 提供公共setter和getter方法
```

成员变量/属性

```
属性：setter和getter方法截取后的产物
例如：getUsername() --> Username--> username
```

方法

```java
setProperty()
getProperty()
populate(Object obj , Map map) // 将map集合的键值对信息，封装到对应的JavaBean对象中
```

