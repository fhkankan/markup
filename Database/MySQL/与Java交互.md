# 与Java交互

## JDBC

### 概述

概念

```
JDBC 规范定义接口，具体的实现由各大数据库厂商来实现。
Java访问数据库的标准规范，真正怎么操作数据库还需要具体的实现类，也就是数据库驱动。每个数据库厂商根据自家数据库的通信格式编写好自己数据库的驱动。所以我们只需要会调用 JDBC 接口中的方法即可，数据库驱动由数据库厂商提供。
```

使用的包

```java
java.sql // 所有与JDBC访问数据库相关的接口和类
javax.sql	// 数据库扩展包，提供数据库额外的功能，如：连接池
数据库驱动  // 数据库厂商提供，需额外下载，是对JDBC接口实现的类
```

JDBC核心API

```java
DriverManager类
// 驱动管理对象，管理和注册数据库驱动，得到数据库连接对象
Connection 接口
// 数据库连接对象，可用于创建Statement和PreparedStatement对象
Statement接口
// 执行sql的对象，用于将SQL语句发送给数据库服务器
PreparedStatement接口
// 一个sql语句对象，是Statement的子接口
ResultSet接口
// 结果集对象，用于封装数据库查询的结果集，返回给客户端Java程序
```

使用步骤

```
1.导入驱动jar包：复制驱动包到libs目录，IDE右键add as library
2.注册驱动
3.获取数据库连接对象
4.定义sql
5.获取执行sql语句的对象statement
6.使用statement对象执行sql，接收返回结果
7.处理结果
8.释放资源
```

目录

```
-test
  -libs
  	mysql-connector-java-5.1.37-bin.jar
  -src
  	-cn
  		-itcast
  			- jdbc
  				demo.java		
```

快速入门

```java
package cn.itcast.jdbc;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;


public class Demo {
    public static void main(String[] args) throws Exception {
        //1.导入驱动jar包
        //2.注册驱动
        Class.forName("com.mysql.jdbc.Driver"); //mysql5之后可以省略注册驱动步骤
        //3.获取数据库连接对象
        Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/db3", "root", "root");
        //4.定义sql语句
        String sql = "update account set balance = 2000 where id = 1";
        //5.获取执行sql的对象 Statement
        Statement stmt = conn.createStatement();
        //6.执行sql
        sql = "xxx";
        int count = stmt.executeUpdate(sql);
        //7.处理结果
        System.out.println(count);
        //8.释放资源
        stmt.close();
        conn.close();
    }
}
```

### DriverManager

类中的方法

```java
// 注册驱动
static void registerDriver(Driver driver)
// mysql5之前需要注册驱动，不直接调用注册驱动方法，而是使用Class.forName("com.mysql.jdbc.Driver")，是由于com.mysql.jdbc.Driver类中有static直接调用了注册驱动方法
// mysql5之后不需要注册驱动声明，是由于META-INF/service/java.sql.Driver中声明了com.mysql.jdbc.Driver

// 获取数据库连接
Connection getConnection (String url, String user, String password)
// 通过连接字符串，用户名，密码来得到数据 库的连接对象
Connection getConnection (String url, Properties info)
// 通过连接字符串，属性对象来得到连接对象
  
// 参数
- user  		登录的用户名
- password 	登录的密码
- url 			连接数据库的URL地址格式：协议名:子协议://服务器名或 IP 地址:端口号/数据库名?参数=参数值
						mysql的写法 jdbc:mysql://localhost:3306/数据库[?参数名=参数值]
						本地服务器端口号3306，可简写为dbc:mysql:///数据库名
						如果数据库出现乱码，可以指定参数?characterEncoding=utf8
- info			驱动类的字符串名 com.mysql.jdbc.Driver
```

使用

```java
package com.itheima;
import java.sql.Connection; 
import java.sql.DriverManager; 
import java.sql.SQLException;

public class Demo2 {
		public static void main(String[] args) throws SQLException {
      // 注册驱动
      Class.forName("com.mysql.jdbc.Driver"); //mysql5之后可以省略注册驱动步骤
      
      // 1.使用用户名、密码、URL 得到连接对象
      String url = "jdbc:mysql://localhost:3306/day24"; 
      Connection connection = DriverManager.getConnection(url, "root", "root"); 
      System.out.println(connection);
      
      // 2.使用属性文件
      String url = "jdbc:mysql://localhost:3306/day24";
      Properties info = new Properties();
      info.setProperty("user","root");
      info.setProperty("password","root");
      Connection connection = DriverManager.getConnection(url, info);
      System.out.println(connection);
		} 
}
```

### Connection

Connection 接口，具体的实现类由数据库的厂商实现，代表一个连接对象。

方法

```java
// 获取执行sql对象
Statement createStatement()//用于执行静态sql语句并返回其生成结果的对象
PreparedStatement prepareStatement(String sql)//指定预编译的SQL语句，SQL语句中变量使用占位符? 创建一个语句对象
 
// 管理事务
void setAutoCommit(boolean autoCommit)//参数是 true,false，如果设置为false，表示关闭自动提交，相当于开启事务
void commit()//提交事务
void rollback()//回滚事务
```

### Statement

代表一条语句对象，用于发送 SQL 语句给服务器，用于执行静态 SQL 语句并返回它所生成结果的对象。

方法

```java
int executeUpdate(String sql)
// 用于发送 DML 语句，增删改的操作，insert、update、delete
// 参数:SQL 语句，
// 返回值:对于DML语句返回对数据库影响的行数，对于DDL语句不返回任何内容

ResultSet executeQuery(String sql)
// 用于发送 DQL 语句，执行查询的操作，select 
// 参数:SQL 语句，
// 返回值:查询的结果集
  
- next()：游标向下移动一行，判断当前是否有数据，无则false，有则true
- getXXX()：获取当前行中列的数据
  参数int：代表列的编号，从1开始
  参数string：代表列的名称 
```

常用数据转换类型

| SQL类型                        | Jdbc对应方法     | 返回类型                                |
| ----------------------------------- | --------------------- | --------------------------------------- |
| BIT(1) bit(n)                       | `getBoolean()`   | boolean                            |
| TINYINT          | `getByte()` | byte |
| SMALLINT                        | `getShort()`      | short                               |
| INT                | `getInt()` | int |
| BIGINT                          | `getLong()`       | long                                |
| CHAR,VARCHAR | `getString() ` | String |
| Text(Clob) Blob                 | `getClob getBlob()` | Clob Blob                           |
| DATE                            | `getDate()`       | java.sql.Date 只代表日期            |
| TIME                            | `getTime()`       | java.sql.Time 只表示时间            |
| TIMESTAMP                       | `getTimestamp()`  | java.sql.Timestamp 同时有日期和时间 |

释放资源

```
1.需要释放的对象:ResultSet结果集，Statement语句，Connection连接 
2.释放原则:先开的后关，后开的先关。ResultSet->Statement->Connection 
3.放在哪个代码块中:finally 块
```

使用-增/删/改

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;


public class Demo {
    public static void main(String[] args) {
        //1. 创建连接
        Connection conn = null;
        Statement statement = null;
        try {
            //2. 通过连接对象得到语句对象
            conn = DriverManager.getConnection("jdbc:mysql:///day24", "root", "root");
            //3. 通过语句对象发送 SQL 语句给服务器
            statement = conn.createStatement();
            //4. 执行 SQL
            sql = "create table student (id int PRIMARY key auto_increment, name varchar(20) not null, gender boolean, birthday date)";
            //5. 返回影响行数(DDL 没有返回值)
            statement.executeUpdate(sql);
            System.out.println("创建表成功");
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            //6. 释放资源,关闭之前要先判断
            if (statement != null) {
                try {
                    statement.close();
                } catch (SQLException e) {
                    e.printStackTrace();
                }
            }
            if (conn != null) {
                try {
                    conn.close();
                } catch (SQLException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
```

使用-查

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.ResultSet;
import java.sql.Date;

public class Demo {
    public static void main(String[] args) throws SQLException {
        //1 得到连接对象
        Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/day24", "root", "root");
        //2 得到语句对象
        Statement statement = connection.createStatement();

        //3 执行SQL语句得到结果集ResultSet对象
        String sql = "select * from student";
        ResultSet rs = statement.executeQuery(sql);
        //4 循环遍历取出每一条记录
        while (rs.next()) {
            int id = rs.getInt("id");
            String name = rs.getString("name");
            boolean gender = rs.getBoolean("gender");
            Date birthday = rs.getDate("birthday");
            //5 输出的控制台上
            System.out.println("编号:" + id + ", 姓名:" + name + ", 性别:" + gender + ", 生日:" + birthday);
        }
        //6 释放资源 
        rs.close();
        statement.close();
        connection.close();
    }
}
```

### PreparedStatement

PreparedStatement 是 Statement 接口的子接口，继承于父接口中所有的方法。它是一个预编译的 SQL 语句。

因为有预先编译的功能，提高 SQL 的执行效率。 可以有效的防止 SQL 注入的问题，安全性更高

执行原理
```
statement对象每执行一条sql语句都会将sql语句发送给数据库，数据库先编译SQL，再执行，若多条sql语句，需要编译多次。
PreparedStatement会先将sql语句发送给数据库预编译，会引用预编译后的结果，可以多次传入不同的参数并执行。若多条sql语句，只需编译一次，减少了编译次数，提高了执行效率。
```

方法

```java
int executeUpdate()					//	执行 DML，增删改的操作，返回影响的行数。
ResultSet executeQuery() 		//	执行 DQL，查询的操作，返回结果集
  
// PreparedStatement 中设置参数的方法
void setDouble(int parameterIndex, double x)		// 将指定参数设置为给定 Java double 值。
void setFloat(int parameterIndex, float x) 			// 将指定参数设置为给定 Java REAL 值。
void setInt(int parameterIndex, int x)					// 将指定参数设置为给定 Java int 值。
void setLong(int parameterIndex, long x) 				// 将指定参数设置为给定 Java long 值。
void setObject(int parameterIndex, Object x)		// 使用给定对象设置指定参数的值。
void setString(int parameterIndex, String x) 		// 将指定参数设置为给定 Java String 值。
```

使用步骤

```
1.编写 SQL 语句，未知内容使用?占位:"SELECT * FROM user WHERE name=? AND password=?"; 获得 PreparedStatement 对象
2.设置实际参数:setXxx(占位符的位置, 真实的值)
3.执行参数化 SQL 语句
4.关闭资源
```

数据库工具类

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.ResultSet;

public class JdbcUtils {
    // 可以把几个字符串定义成常量:用户名，密码，URL，驱动类
    private static final String USER = "root";
    private static final String PWD = "root";
    private static final String URL = "jdbc:mysql://localhost:3306/day24";
    private static final String DRIVER = "com.mysql.jdbc.Driver";

    /**
     * 注册驱动 
     */
    static {
        try {
            Class.forName(DRIVER);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * 得到数据库的连接
     */
    public static Connection getConnection() throws SQLException {
        return DriverManager.getConnection(URL, USER, PWD);
    }

    /**
     * 关闭所有打开的资源
     */
    public static void close(Connection conn, Statement stmt) {
        if (stmt != null) {
            try {
                stmt.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (conn != null) {
            try {
                conn.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * 关闭所有打开的资源
     */
    public static void close(Connection conn, Statement stmt, ResultSet rs) {
        if (rs != null) {
            try {
                rs.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        close(conn, stmt);
    }
}
```

使用-增删改

```java
import java.sql.*;
import java.sql.SQLException;

public class Demo {
    public static void main(String[] args) throws SQLException {
        Connection connection = JdbcUtils.getConnection();
        String sql = "insert into student values (null, ?, ?, ?)";
        PreparedStatement ps = connection.prepareStatement(sql);
        // 设置参数
        ps.setString(1, "小白龙");
        ps.setBoolean(2, true);
        ps.setDate(3, java.sql.Date.valueOf("1999-11-11"))
        int row = ps.executeUpdate();
        System.out.println("插入了"+row + "条记录");
        JdbcUtils.close(connection, ps);
    }
}
```

使用-查

```java
import java.sql.*;
import java.sql.SQLException;

public class Demo {
    public static void main(String[] args) throws SQLException {
        // 创建学生对象
        Student student = new Student();
        Connection connection = JdbcUtils.getConnection();
        String sql = "select * from student where id=?";
        PreparedStatement ps = connection.prepareStatement(sql);
        // 设置参数
        ps.setInt(1, 2);
        ResultSet resultSet = ps.executeQuery();
        if (resultSet.next()) {
            // 封装成一个学生对象
            student.setId(resultSet.getInt("id"));
            student.setName(resultSet.getString("name"));
            student.setGender(resultSet.getBoolean("gender"));
            studdent.setBirthday(resultSet.getDate("birthday"));
        }
        //6 释放资源
        JdbcUtils.close(connection, ps, resultSet);
        System.out.println(student);
    }
}
```

### 事务

开发步骤

```
1) 获取连接
2) 开启事务
3) 获取到 PreparedStatement
4) 使用 PreparedStatement 执行两次更新操作
5) 正常情况下提交事务
6) 出现异常回滚事务
7) 最后关闭资源
```

使用

```java
import java.sql.Connection;
import java.sql.PreparedStatement
import java.sql.SQLException;

public class Demo {
    public static void main(String[] args) throws SQLException {
        // 注册驱动
        Connection connection = null;
        PreparedStatement ps = null;
        try {
            // 获取连接
            connection = JdbcUtils.getConnection();
            // 开启事务
            connection.setAutoCommit(false);
            // 获取PreparedStatement
            String sql1 = "update account set balance=balance - ? where name=?";
            ps = connection.prepareStatement(sql);
            ps.setInt(1, 500);
            ps.setString(2, "jack");
            ps.executeUpdate();
            // 异常
            System.out.println(100 / 0);
            String sql2 = "update account set balance=balance + ? where name=?";
            ps = connection.prepareStatement(sql);
            ps.setInt(1, 500);
            ps.setString(2, "Rose");
            ps.executeUpdate();
            // 提交事务
            connection.commit();
            System.out.println("转账成功");
        } catch (Exception e) {
            e.printStackTrace();
            try {
                // 事务回滚
                connection.rollback();
            } catch (SQLException el) {
                el.printStackTrace();
            }
            System.out.println("转帐失败");
        } finally {
            // 关闭资源
            JdbcUtils.close(connection, ps);
        }
    }
}
```

## 连接池

- 概念

其实就是一个容器(集合)，存放数据库连接的容器。

当系统初始化好后，容器被创建，容器中会申请一些连接对象，当用户来访问数据库时，从容器中获取连接对象，用户访问完之后，会将连接对象归还给容器。

- 实现

标准接口：`DataSource`   javax.sql包下的

方法

```java
getConnection() 		// 获取连接
Connection.close() 	// 归还连接，如果连接对象Connection是从连接池中获取的，那么调用Connection.close()方法，则不会再关闭连接了。而是归还连接
```

使用

```java
// 一般我们不去实现它，有数据库厂商来实现
1. C3P0：数据库连接池技术
2. Druid：数据库连接池实现技术，由阿里巴巴提供的
```

### c3p0

步骤

```
1. 导入jar包 
	数据库驱动jar包，c3p0-0.9.5.2.jar，mchange-commons-java-0.2.12.jar
2. 定义配置文件
	名称： c3p0.properties 或者 c3p0-config.xml
	路径：直接将文件放在src目录下，可自动识别加载
3. 创建核心对象 
	数据库连接池对象 ComboPooledDataSource
4. 获取连接： getConnection
```

- 使用

项目目录

```java
-test
  -libs
  	c3p0-0.9.5.2.jar
  	mchange-commons-java-0.2.12.jar
  	mysql-connector-java-5.1.37-bin.jar
  -src
  	-cn
  		-itcast
  			- datasource
  				-c3p0
  					demo.java		
  	c3p0-config.xml
```

配置

```xml
<c3p0-config>
  <!-- 使用默认的配置读取连接池对象 -->
  <default-config>
  	<!--  连接参数 -->
    <property name="driverClass">com.mysql.jdbc.Driver</property>
    <property name="jdbcUrl">jdbc:mysql://localhost:3306/db4</property>
    <property name="user">root</property>
    <property name="password">root</property>
    
    <!-- 连接池参数 -->
    <!--初始化申请的连接数量-->
    <property name="initialPoolSize">5</property>
    <!--最大的连接数量-->
    <property name="maxPoolSize">10</property>
    <!--超时时间-->
    <property name="checkoutTimeout">3000</property>
  </default-config>

  <named-config name="otherc3p0"> 
    <!--  连接参数 -->
    <property name="driverClass">com.mysql.jdbc.Driver</property>
    <property name="jdbcUrl">jdbc:mysql://localhost:3306/db3</property>
    <property name="user">root</property>
    <property name="password">root</property>
    
    <!-- 连接池参数 -->
    <property name="initialPoolSize">5</property>
    <property name="maxPoolSize">8</property>
    <property name="checkoutTimeout">1000</property>
  </named-config>
</c3p0-config>
```

代码

```java
package cn.itcast.datasource.c3p0;

import com.mchange.v2.c3p0.ComboPooledDataSource;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.SQLException;

public class Demo {
    public static void main(String[] args) throws SQLException {
        // 创建数据库连接池对象
        DataSource ds = new ComboPooledDataSource();// default
        // DataSource ds = new ComboPooledDataSource("otherc3p0");// params
        // 获取连接对象
        Connection conn = ds.getConnection();
        System.out.println(conn);
    }
}
```

### druid

步骤

```
1. 导入jar包 druid-1.0.9.jar
2. 定义配置文件：
	是properties形式的
	可以叫任意名称，可以放在任意目录下，需要手动加载
3. 加载配置文件。Properties
4. 获取数据库连接池对象
	通过工厂来来获取  DruidDataSourceFactory
5. 获取连接：getConnection
```

- 使用

目录

```
-test
  -libs
  	druid-1.0.9.jar
  	mysql-connector-java-5.1.37-bin.jar
  -src
  	-cn
  		-itcast
  			- datasource
  				- druid
  					demo.java	
        - utlis
        	JDBCUtils.java
  	druid.properties
```

配置

```
driverClassName=com.mysql.jdbc.Driver
url=jdbc:mysql:///db3
username=root
password=root
initialSize=5
maxActive=10
maxWait=3000
```

代码

```java
package cn.itcast.datasource.druid;

import com.alibaba.druid.pool.DruidDataSourceFactory;

import javax.sql.DataSource;
import java.io.InputStream;
import java.sql.Connection;
import java.util.Properties;

public class Demo {
    public static void main(String[] args) throws Exception {
        // 加载配置文件
        Properties pro = new Properties();
        InputStream is = Demo.class.getClassLoader().getResourceAsStream("druid.properties");
        pro.load(is);
        // 获取连接对象
        DataSource ds = DruidDataSourceFactory.createDataSource(pro);
        // 获取连接
        Connection conn = ds.getConnection();
        System.out.println(conn);
    }
}
```

工具类

```java
package cn.itcast.datasource.utils;

import com.alibaba.druid.pool.DruidDataSourceFactory;

import javax.sql.DataSource;
import java.io.IOException;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Properties;

public class JDBCUtils {
    private static DataSource ds;

    static {
        try {
            // 加载配置
            Properties pro = new Properties();
            pro.load(JDBCUtils.class.getClassLoader().getResourceAsStream("druid.properties"));
            // 获取DataSource
            ds = DruidDataSourceFactory.createDataSource(pro);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 获取连接池
    public static DataSource getDataSource() {
        return ds;
    }

    // 获取连接
    public static Connection getConnection() throws SQLException {
        return ds.getConnection();
    }

    // 释放资源
    public static void close(Statement stmt, Connection conn) {
        close(null, stmt, conn);
    }

    public static void close(ResultSet rs, Statement stmt, Connection conn) {
        if (rs != null) {
            try {
                rs.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (stmt != null) {
            try {
                stmt.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }

        if (conn != null) {
            try {
                conn.close();//归还连接
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}
```

使用工具类

```java
package cn.itcast.datasource.druid;

import cn.itcast.datasource.utils.JDBCUtils;
import com.alibaba.druid.pool.DruidDataSourceFactory;
import com.mchange.v2.c3p0.ComboPooledDataSource;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

public class Demo {
    public static void main(String[] args) {
        Connection conn = null;
        PreparedStatement pstmt = null;
        try {
            // 获取连接
            conn = JDBCUtils.getConnection();
            // 定义sql
            String sql = "insert int account values (null, ?, ?)";
            // 获取pstmt对象
            pstmt = conn.prepareStatement(sql);
            pstmt.setString(1, "zhangsan");
            pstmt.setDouble(2, 3000);
            // 执行sql
            int count = pstmt.executeUpdate();
            System.out.println(count);
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            // 释放资源
            JDBCUtils.close(pstmt, conn);
        }
    }
}
```

### SpringJDBC

步骤

```
1.导入jar包
2.创建JdbcTemplate对象。依赖于数据源DataSource
	JdbcTemplate template = new JdbcTemplate(ds);
3.调用JdbcTemplate的方法来完成CRUD的操作
	update()// 执行DML语句。增、删、改语句
	queryForMap()//查询结果将结果集封装为map集合，将列名作为key，将值作为value 将这条记录封装为一个map集合。结果集长度是1
	queryForList()//查询结果将结果集封装为list集合，将每一条记录封装为一个Map集合，再将Map集合装载到List集合中
	query()// 查询结果，将结果封装为JavaBean对象
			// query的参数：RowMapper
			// 一般我们使用BeanPropertyRowMapper实现类。可以完成数据到JavaBean的自动封装
			new BeanPropertyRowMapper<类型>(类型.class)
	queryForObject//查询结果，将结果封装为对象，一般用于聚合函数的查询
```

- 使用

目录

```
-test
  -libs
  	spring-beans-5.0.0.RELEASE.jar
  	spring-core-5.0.0.RELEASE.jar
  	spring-jdbc-5.0.0.RELEASE.jar
  	spring-tx-5.0.0.RELEASE.jar
  	mysql-connector-java-5.1.37-bin.jar
  -src
  	-cn
  		-itcast
  			- jdbctemplate
  				Demo.java
  			- domain
  				Emp.java
        - utlis
        	JDBCUtils.java
  	druid.properties
```

简单代码

```java
package cn.itcast.datasource.jdbctemplate;

import cn.itcast.datasource.utils.JDBCUtils;
import org.springframework.jdbc.core.JdbcTemplate;

public class Demo {
    public static void main(String[] args) {
        // 创建JDBCTemplate对象
        JdbcTemplate template = new JdbcTemplate(JDBCUtils.getDataSource());
        // 调用方法
        String sql = "update account set balance = 5000 where id =?";
        int count = template.update(sql, 1001);
        System.out.println(count);
    }
}
```

数据库映射

```java
package cn.itcast.domain;

import java.util.Date;

public class Emp {
    private Integer id;
    private String ename;
    private Integer job_id;
    private Integer mgr;
    private Date joindate;
    private Double salary;
    private Double bonus;
    private Integer dept_id;


    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getEname() {
        return ename;
    }

    public void setEname(String ename) {
        this.ename = ename;
    }

    public Integer getJob_id() {
        return job_id;
    }

    public void setJob_id(Integer job_id) {
        this.job_id = job_id;
    }

    public Integer getMgr() {
        return mgr;
    }

    public void setMgr(Integer mgr) {
        this.mgr = mgr;
    }

    public Date getJoindate() {
        return joindate;
    }

    public void setJoindate(Date joindate) {
        this.joindate = joindate;
    }

    public Double getSalary() {
        return salary;
    }

    public void setSalary(Double salary) {
        this.salary = salary;
    }

    public Double getBonus() {
        return bonus;
    }

    public void setBonus(Double bonus) {
        this.bonus = bonus;
    }

    public Integer getDept_id() {
        return dept_id;
    }

    public void setDept_id(Integer dept_id) {
        this.dept_id = dept_id;
    }

    @Override
    public String toString() {
        return "Emp{" +
                "id=" + id +
                ", ename='" + ename + '\'' +
                ", job_id=" + job_id +
                ", mgr=" + mgr +
                ", joindate=" + joindate +
                ", salary=" + salary +
                ", bonus=" + bonus +
                ", dept_id=" + dept_id +
                '}';
    }
}
```

数据库表操作

```java
package cn.itcast.jdbctemplate;

import cn.itcast.domain.Emp;
import cn.itcast.utils.JDBCUtils;
import org.junit.Test;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;

import java.sql.Date;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;
import java.util.Map;

public class JdbcTemplateDemo2 {

    //Junit单元测试，可以让方法独立执行
    //1. 获取JDBCTemplate对象
    private JdbcTemplate template = new JdbcTemplate(JDBCUtils.getDataSource());
    
  	/**
     * 1. 修改1号数据的 salary 为 10000
     */
    @Test
    public void test1(){
        String sql = "update emp set salary = 10000 where id = 1001";
        int count = template.update(sql);
        System.out.println(count);
    }

    /**
     * 2. 添加一条记录
     */
    @Test
    public void test2(){
        String sql = "insert into emp(id,ename,dept_id) values(?,?,?)";
        int count = template.update(sql, 1015, "郭靖", 10);
        System.out.println(count);

    }

    /**
     * 3.删除刚才添加的记录
     */
    @Test
    public void test3(){
        String sql = "delete from emp where id = ?";
        int count = template.update(sql, 1015);
        System.out.println(count);
    }

    /**
     * 4.查询id为1001的记录，将其封装为Map集合
     * 注意：这个方法查询的结果集长度只能是1
     */
    @Test
    public void test4(){
        String sql = "select * from emp where id = ? or id = ?";
        Map<String, Object> map = template.queryForMap(sql, 1001,1002);
        System.out.println(map);
        //{id=1001, ename=孙悟空, job_id=4, mgr=1004, joindate=2000-12-17, salary=10000.00, bonus=null, dept_id=20}

    }

    /**
     * 5. 查询所有记录，将其封装为List
     */
    @Test
    public void test5(){
        String sql = "select * from emp";
        List<Map<String, Object>> list = template.queryForList(sql);

        for (Map<String, Object> stringObjectMap : list) {
            System.out.println(stringObjectMap);
        }
    }

    /**
     * 6. 查询所有记录，将其封装为Emp对象的List集合
     */

    @Test
    public void test6(){
        String sql = "select * from emp";
        List<Emp> list = template.query(sql, new RowMapper<Emp>() {

            @Override
            public Emp mapRow(ResultSet rs, int i) throws SQLException {
                Emp emp = new Emp();
                int id = rs.getInt("id");
                String ename = rs.getString("ename");
                int job_id = rs.getInt("job_id");
                int mgr = rs.getInt("mgr");
                Date joindate = rs.getDate("joindate");
                double salary = rs.getDouble("salary");
                double bonus = rs.getDouble("bonus");
                int dept_id = rs.getInt("dept_id");

                emp.setId(id);
                emp.setEname(ename);
                emp.setJob_id(job_id);
                emp.setMgr(mgr);
                emp.setJoindate(joindate);
                emp.setSalary(salary);
                emp.setBonus(bonus);
                emp.setDept_id(dept_id);

                return emp;
            }
        });


        for (Emp emp : list) {
            System.out.println(emp);
        }
    }

    /**
     * 6. 查询所有记录，将其封装为Emp对象的List集合
     */

    @Test
    public void test6_2(){
        String sql = "select * from emp";
        List<Emp> list = template.query(sql, new BeanPropertyRowMapper<Emp>(Emp.class));
        for (Emp emp : list) {
            System.out.println(emp);
        }
    }

    /**
     * 7. 查询总记录数
     */

    @Test
    public void test7(){
        String sql = "select count(id) from emp";
        Long total = template.queryForObject(sql, Long.class);
        System.out.println(total);
    }

}
```

