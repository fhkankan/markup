# JSON

使用Newtonsoft.Json

## 序列化

- 已知类型的对象转换成Json字符串

```c#
Student student = new Student();
student.UserName = "张三";
string json = Newtonsoft.Json.JsonConvert.SerializeObject(student);
//最终输出的结果为:{UserName:"张三"}
```

- 匿名对象转换成Json字符串

在开发过程中我们往往临时需要构造一段Json字符串，但是按照Newtonsoft库使用的方法需要定义一个类，然后根据这个类的实例才能生成Json字符串。很显然这种方法是比较繁琐的。所以这里给大家演示一下匿名对象转换成Json字符串的方法，这也是我在日常开发中经常使用的方法。

```c#
var student = new {
     UserName ="张三";
}
string json = Newtonsoft.Json.JsonConvert.SerializeObject(student);
//最终输出的结果为:{UserName:"张三"}
```

## 反序列化

```c#
string Json = "{UserName:"张三"}";

Newtonsoft.Json.JsonConvert.DeserializeObject<Object>(Json);
```

通过上面的这段代码即可完成对Json字符串的反序列化，如果你仅仅只想解析拿到Json某个Key的Value可以这样子做。

```c#
string Json = "{UserName:"张三"}";
dynamic student = Newtonsoft.Json.JsonConvert.DeserializeObject<dynamic>(Json);
string username = student.UserName;

// 或者这样子做
string Json = "{UserName:"张三"}";
JObject student = Newtonsoft.Json.JsonConvert.DeserializeObject<JObject>(Json);
string username = student["UserName"].Value<string>();
```

这样就省去了为Json字符串的反序列化而特地写一个类的麻烦。