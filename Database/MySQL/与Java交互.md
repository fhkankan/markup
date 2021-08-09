# 与Java交互

## JDBC概述

概念

```
JDBC 规范定义接口，具体的实现由各大数据库厂商来实现。
Java访问数据库的标准规范，真正怎么操作数据库还需要具体的实现类，也就是数据库驱动。每个数据库厂商根据自家数据库的通信格式编写好自己数据库的驱动。所以我们只需要会调用 JDBC 接口中的方法即 可，数据库驱动由数据库厂商提供。
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
// 一个SQL语句对象，是Statement的子接口
ResultSet接口
// 结果集对象，用于封装数据库查询的结果集，返回给客户端Java程序
```

使用步骤

```
1.导入驱动jar包
2.注册驱动
3.获取数据库连接对象
4.定义sql
5.获取执行sql语句的对象statement
6.执行sql，接收返回结果
7.处理结果
8.释放资源
```

快速入门

```java
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;


public class JdbcDemo1 {
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
        int count = stmt.executeUpdate(sql);
        //7.处理结果
        System.out.println(count);
        //8.释放资源
        stmt.close();
        conn.close();
    }
}
```

## JDBC对象

### DriverManager

