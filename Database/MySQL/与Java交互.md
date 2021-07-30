# 与Java交互

## JDBC

使用的包

```java
java.sql // 所有与JDBC访问数据库相关的接口和类
javax.sql	// 数据库扩展包，提供数据库额外的功能，如：连接池
数据库驱动  // 数据库厂商提供，需额外下载，是对JDBC接口实现的类
```

JDBC核心API

```java
DriverManager类
// 管理和注册数据库驱动，得到数据库连接对象
Connection 接口
// 一个连接对象，可用于创建Statement和PreparedStatement对象
Statement接口
// 一个SQL语句对象，用于将SQL语句发送给数据库服务器
PreparedStatement接口
// 一个SQL语句对象，是Statement的子接口
ResultSet接口
// 用于封装数据库查询的结果集，返回给客户端Java程序
```

