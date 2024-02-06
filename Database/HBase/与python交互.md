# python交互

[参考文档](https://happybase.readthedocs.io/en/latest/api.html)

HappyBase 是FaceBook员工开发的操作HBase的python库, 其基于Python Thrift, 但使用方式比Thrift简单, 已被广泛应用

## 安装

```shell
# 服务器启动hbase thrift server 
hbase-daemon.sh start thrift
# 客户端安装happybase
pip install happybase
```

## 使用

### 建立连接

```python
import happybase
connection = happybase.Connection('somehost')

# 当连接建立时, 会自动创建一个与 HBase Thrift server的socket链接. 可以通过参数禁止自动链接, 然后再需要连接是调用Connection.open()
connection = happybase.Connection('somehost', autoconnect=False)
# before first use:
connection.open()
```

### 操作表

```python
# 获取所有表
print(connection.tables())

# 创建表
connection.create_table('users',{'cf1': dict()})

# 创建表之后可以传入表名获取到Table类的实例:
table = connection.table('mytable')

# 删除表
#api
conn.delete_table(table_name, True)
#函数封装
def delete_table(table_name):
    pretty_print('delete table %s now.' % table_name)
    conn.delete_table(table_name, True)
```

### 查询操作

```python
# api
table.scan() #全表查询
table.row(row_keys[0]) # 查询一行
table.rows(row_keys) # 查询多行

#封装函数
def show_rows(table, row_keys=None):
    if row_keys:
        print('show value of row named %s' % row_keys)
        if len(row_keys) == 1:
            print(table.row(row_keys[0]))
        else:
            print(table.rows(row_keys))
    else:
        print('show all row values of table named %s' % table.name)
        for key, value in table.scan():
            print(key, value)
```

### 插入数据

```python
#api
table.put(row_key, {cf:cq:value})
def put_row(table, column_family, row_key, value):
    print('insert one row to hbase')
    #put 'user','rowkey_10','base_info:username','Tom'
    #{'cf:cq':’数据‘}
    table.put(row_key, {'%s:name' % column_family:'name_%s' % value})

def put_rows(table, column_family, row_lines=30):
    print('insert rows to hbase now')
    for i in range(row_lines):
        put_row(table, column_family, 'row_%s' % i, i)
```

### 删除数据

```python
#api
table.delete(row_key, cf_list)
    
#函数封装    
def delete_row(table, row_key, column_family=None, keys=None):
    if keys:
        print('delete keys:%s from row_key:%s' % (keys, row_key))
        key_list = ['%s:%s' % (column_family, key) for key in keys]
        table.delete(row_key, key_list)
    else:
        print('delete row(column_family:) from hbase')
        table.delete(row_key)
```

### 集中示例

```python
  import happybase
  
  hostname = '192.168.199.188'
  table_name = 'users'
  column_family = 'cf'
  row_key = 'row_1'
  
  conn = happybase.Connection(hostname)
  
  def show_tables():
      print('show all tables now')
      tables =  conn.tables()
      for t in tables:
          print t
  
  def create_table(table_name, column_family):
      print('create table %s' % table_name)
      conn.create_table(table_name, {column_family:dict()})
  
  
  def show_rows(table, row_keys=None):
      if row_keys:
          print('show value of row named %s' % row_keys)
          if len(row_keys) == 1:
              print table.row(row_keys[0])
          else:
              print table.rows(row_keys)
      else:
          print('show all row values of table named %s' % table.name)
          for key, value in table.scan():
              print key, value
  
  def put_row(table, column_family, row_key, value):
      print('insert one row to hbase')
      table.put(row_key, {'%s:name' % column_family:'name_%s' % value})
  
  def put_rows(table, column_family, row_lines=30):
      print('insert rows to hbase now')
      for i in range(row_lines):
          put_row(table, column_family, 'row_%s' % i, i)
  
  def delete_row(table, row_key, column_family=None, keys=None):
      if keys:
          print('delete keys:%s from row_key:%s' % (keys, row_key))
          key_list = ['%s:%s' % (column_family, key) for key in keys]
          table.delete(row_key, key_list)
      else:
          print('delete row(column_family:) from hbase')
          table.delete(row_key)
  
  def delete_table(table_name):
      pretty_print('delete table %s now.' % table_name)
      conn.delete_table(table_name, True)
  
  def pool():
      pretty_print('test pool connection now.')
      pool = happybase.ConnectionPool(size=3, host=hostname)
      with pool.connection() as connection:
          print connection.tables()
  
  def main():
      # show_tables()
      # create_table(table_name, column_family)
      # show_tables()
  
      table = conn.table(table_name)
      show_rows(table)
      put_rows(table, column_family)
      show_rows(table)
      #
      # # 更新操作
      # put_row(table, column_family, row_key, 'xiaoh.me')
      # show_rows(table, [row_key])
      #
      # # 删除数据
      # delete_row(table, row_key)
      # show_rows(table, [row_key])
      #
      # delete_row(table, row_key, column_family, ['name'])
      # show_rows(table, [row_key])
      #
      # counter(table, row_key, column_family)
      #
      # delete_table(table_name)
  
  if __name__ == "__main__":
      main()
```

