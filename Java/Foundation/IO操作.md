# IO操作

## File

`File`类位于`java.io`包中，是一个用于操作文件和目录的类，`File`类直接继承`Object`类，实现了`Comparable`接口，说明`File`类的对象具备Lee比较大小的能力。`File`类的对象只能用于操作磁盘上文件的属性，而无法读写磁盘上文件中的内容。

`File`类的定义

```java
public class File implemeents Serializable, Comparable<File>
```

常用构造方法

| 构造方法                          | 描述                                       |
| --------------------------------- | ------------------------------------------ |
| `File(String pathname)`           | 通过给定路径名字符串创建一个`File`类的对象 |
| `File(File parent, String child)` | 根据父目录与子文件名创建一个`File`类的对象 |

操作文件方法

| 返回值类型 | 方法名              | 描述                                                         |
| ---------- | ------------------- | ------------------------------------------------------------ |
| `boolean`  | `createNewFile()`   | 创建文件，如果磁盘上的文件不存在，创建成功返回true,否则返回false |
| `boolean`  | `delete()`          | 删除文件，如果磁盘上的文件存在，直接从磁盘上删除，不经过回收站，如果不存在返回false |
| `boolean`  | `exists()`          | 判断磁盘上的文件是否存在，返回值为true或false                |
| `String`   | `getPath()`         | 获取相对路径，相对于项目的根目录                             |
| `String`   | `getAbsolutePath()` | 获取绝对路径                                                 |
| `boolean`  | `isFile()`          | 判断一个File对象是否是文件                                   |
| `String`   | `getName()`         | 获取文件名称                                                 |
| `long`     | `length()`          | 获取文件的大小，以字节为单位                                 |

操作目录方法

| 返回值类型 | 方法名          | 描述                                               |
| ---------- | --------------- | -------------------------------------------------- |
| `boolean`  | `mkdir()`       | 创建单层目录                                       |
| `boolean`  | `mkdirs()`      | 创建多层目录                                       |
| `boolean`  | `delete()`      | 只能删除空目录                                     |
| `boolean`  | `isDirectory()` | 判断是否是目录                                     |
| `String[]` | `list()`        | 获取指定目录下的所有目录和文件路径的字符串表示形式 |
| `File[]`   | `listFiles()`   | 获取指定目录下的所有目录和文件对象                 |

## IO流

File类只能查看文件或目录的属性，但是不能查看文件中的内容也不能操作文件中的内容，想要查看或操作文件中的内容需要使用IO流。流是一连串流动的字符，以先进先出的方式发送到

根据流的流向，可以分为输入流和输出流，输入流的两大基类为`InputStream,Reader`，输出流的两大基类是`OutputStream,Writer`。

根据流的处理单元，可以分为字节流和字符流，字节流的两大基类为`InputStream,OutputStream`，字符流的两大基类为`Reader,Writer`。

根据流的功能不同，可以分为直接从数据源活目的地读写数据的额节点流和不直接连接到数据源活目的地的处理流。

### 字节流

- 字节输入流`InputStream`

字节流又被称为万能的字节流，应为在磁盘上存储的文档、音频、视频、图片等都是按照字节存储的。所以只要是按照字节存储的，都可以使用字节流。

字节输出流常用方法

| 返回值类型     | 方法名称         | 描述                       |
| -------------- | ---------------- | -------------------------- |
| `abstract int` | `read()`         | 读取一个字节数据           |
| `int`          | `read(byte[] b)` | 将数据读取到字节数组b中    |
| `void`         | `close()`        | 关闭流                     |
| `int`          | `available()`    | 返回输入流中估计的额字节数 |

`InputStream`是`abstract`类 ，要想使用输入流就需要找数据源，而数据源是文件的输入流，`FileInputStream`是`InputStream`的子类，用于从文件中读取内容到程序。

`FileInputStream`的常用构造方法

| 构造方法                           | 描述                                         |
| ---------------------------------- | -------------------------------------------- |
| `FileInputStream(File file)`       | 使用FIle对象来构造维恩旧爱你输入流对象       |
| `FileInputStream(String pathName)` | 使用String类型的字符串路径构造文件输入流对象 |

实现

```java
// 读取文件中内容 
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建FileInputStream的对象，使用父类创建子类对象
        InputStream is = new FileInputStream("./demo.txt");
        // 查看输入流中估计的字节数
        System.out.println("输入流中估计的字节数：" + is.available());
        // 使用read()从输入流中读取一个字节
        int b = is.read();
        System.out.println("读取到的内容为：" + b + "--" + (char) b);
        System.out.println("输入流中剩余的字节数：" + is.available());
        // 关闭流
        is.close();
    }
}

// 循环读取文件中的内容
import java.io.FileInputStream;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建FIleInputStream对象，搭建程序与磁盘上的数据源之间的桥梁
        FileInputStream fis = null;
        fis = new FileInputStream("./demo.txt");
        // 使用循环依次读取文件中的内容
        int b = 0;
        while ((b = fis.read()) != -1) {
            System.out.println("文中内容" + b + "->" + (char) b);
        }
        System.out.println("输入流中剩余的字节数：" + fis.available());
        // 关闭流
        fis.close();
    }
}

// 使用read(byte[] buf)方法读取文件中的内容
import java.io.FileInputStream;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建FIleInputStream对象，搭建程序与磁盘上的数据源之间的桥梁
        FileInputStream fis = null;
        fis = new FileInputStream("./demo.txt");
        // 存储从文件中读取到的额字节的个数
        int length = 0;
        byte[] buf = new byte[1024];
        // 使用read(byte[] buf)读取，读到文件的末尾为-1
        length = fis.read(buf);
        while (length != -1) {
            System.out.println("文中的内容为：" + new String(buf, 0, length));
            length = fis.read(buf);
        }
        // 关闭流
        fis.close();
    }
}
```

- 字节输出流`OutputStream`

`OutputStream`用于将内存中的数据输送到目的地去，所以输出流需要找目的地。

字节输出流常用方法

| 返回值类型      | 方法名称            | 描述                        |
| --------------- | ------------------- | --------------------------- |
| `abstract void` | `write(int b)`      | 写入一个字节数据            |
| `void`          | `write(byte[] buf)` | 写入一个buf数组中的所有数据 |
| `void`          | `close()`           | 关闭输出流                  |

`FileOutputStream`是`OutputStream`的子类，目的地是磁盘上的文件。

常用构造方法

| 构造方法                                        | 描述                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `FileOutputStream(File file)`                   | 使用File对象构造文件输出流对象                               |
| `FileOutputStream(String path)`                 | 使用String类型的路径构造文件输出流对象                       |
| `FileOutputStream(File file, boolean append)`   | 使用File对象构造文件输出流对象，append的值为true表示将内容追加到原有文件内容的后面，append为false表示将覆盖原文件中文件的内容 |
| `FileOutputStream(String path, boolean append)` | 使用String类型的路径构造文件输出流对象，append的值为true表示将内容追加到原有文件内容的后面，append为false表示将覆盖原文件中文件的内容 |

实现

```java
// 覆写字节
import java.io.FileOutputStream;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        // 声明文件输出流对象
        FileOutputStream fos = null;
        // 创建文件输出流对象，搭建与目的地的桥梁
        fos = new FileOutputStream("./demo.txt");
        // 向文件写入一个字节
        fos.write(97);
        fos.close();
    }
}

// 追加数组
import java.io.FileOutputStream;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        // 声明文件输出流对象
        FileOutputStream fos = null;
        // 创建文件输出流对象，搭建与目的地的桥梁，在源文件内容后追加
        fos = new FileOutputStream("./demo.txt", true);
        // 将字节数组中的内容写入到文件
        byte[] buf = new byte[] { 97, 98, 99 };
        fos.write(buf);
        fos.close();
    }
}
```

### 字节缓冲流

缓冲字节流有输入的缓冲字节流`BufferedInputStreeam`和输出的缓冲字节流`BufferedOutputStream`，使用缓冲字节流在进行文件复制时可以提高复制的效率。

```java
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建缓冲字节输入流对象，输出流对象
        BufferedInputStream bis = null;
        BufferedOutputStream bos = null;
        long startTime = System.currentTimeMillis(); // 开始时间
        bis = new BufferedInputStream(new FileInputStream("./demo.txt"));
        bos = new BufferedOutputStream(new FileOutputStream("./test.txt"));
        int length = 0;
        byte[] buf = new byte[1024]; // 中转站大小
        while ((length = bis.read(buf)) != -1) {
            bos.write(buf, 0, length);
        }
        long endTime = System.currentTimeMillis(); // 结束时间
        System.out.println("文件复制耗时：" + (endTime - startTime) + "毫秒");
        // 关闭流
        bos.close();
        bis.close();
    }
}
```

### 字符流

- 字符输入流`Reader`

一个英文占一个字节，一个汉字占两个字节，无论是一个英文还是一个汉字都被称为一个字符(char)。

`Reader`就是一个用于读取字符流的抽象类。

常用方法

| 返回值类型 | 方法名               | 描述                                                  |
| ---------- | -------------------- | ----------------------------------------------------- |
| `int`      | `read()`             | 读取一个在整数范围0~65535之间的字符，达到文件末尾为-1 |
| `int`      | `read(char[], cbuf)` | 将字符读取到char类型的数组中                          |
| `void`     | `close()`            | 关闭流                                                |

`FileReader`是`Reader`的子类，用于从数据源文件中按字符去读取数据。

实现

```java
// 循环读取数据源文件中的字符
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

public class Test {
    public static void main(String[] args) throws IOException {
        // 声明字符流对象
        Reader r = null;
        // 创建字符流输入对象，搭建与数据源的管道
        r  = new FileReader("./demo.txt");
        // 读取数据，按照字符去读取
        int temp = 0;  // 用于存储读到的字符的整数值
        StringBuilder sb = new StringBuilder(); // 一个用来存储字符的容器
        while ((temp = r.read()) != -1) { // 判断是否达到了文件的末尾
            char c = (char) temp;
            sb.append(c);
        }
        // 输出StringBuilder中的内容
        System.out.println(sb.toString()); // 专程String类型
        // 关闭流
        r.close();
    }
}

// 读取字符到字符数组中
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;

public class Test {
    public static void main(String[] args) throws IOException {
        // 声明字符流对象
        Reader r = null;
        // 创建字符流输入对象，搭建与数据源的管道
        r  = new FileReader("./demo.txt");
        // 声明char类型数组，用于存储读到的内容
        char[] ch = new char[1024];
        // 声明int类型的变量用于存车处读到取的字符的个数 
        int length = r.read(ch);
        // 判断是否达到了文件的末尾
        while (length != -1) {
            // 借助String类的构造方法查看读到的内容
            System.out.println(new String(ch,  0, length));
            length = r.read(ch);
        }
        // 关闭流
        r.close();
    }
}
```

- 字符输出流`Writer`

`Writer`是字符输出流的抽象类，用于将字符写入到目的地。

常用方法

| 返回值类型      | 方法名              | 描述                   |
| --------------- | ------------------- | ---------------------- |
| `void`          | `write(int c)`      | 向目的地写入一个字符   |
| `void`          | `write(String str)` | 向目的地写入一个字符串 |
| `abstract void` | `flush()`           | 刷新输出流             |
| `abstract void` | `close()`           | 关闭输出流             |

`FileWriter`是`Writer`的子类，用于向目的地写入字符。

实现

```java
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;

public class Test {
    public static void main(String[] args) throws IOException {
        // 声明字符输出流对象
        Writer w = null;
        // 创建字符输出流对象，搭建与目的地的桥梁
        w  = new FileWriter("./demo.txt");
        // 向目的地写入一个字符
        w.write(97);
        // 向目的地写入一串字符
        w.write("hello world!");
        // 关闭流
        w.close();
    }
}
```

- `OutputStream,Writer`区别

字节输出流`OutputStream`是直接将数据写入到目的地，而字符输出流`Writer`是将数据写入到缓存，需要手动刷新缓存或直接关闭流，将数据再写到目的地。

### 字符缓冲流

字符缓冲流分为`BufferedWriter.BufferedReader`，分别是`Writer，Reader`的子类。缓冲流的构造方法中需要一个`Writer`或`Reader`对象，所以缓冲流是一个处理流。

```java
// BufferedWriter
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建字节流对象
        FileOutputStream fos = new FileOutputStream("./demo.txt");
        // 创建转换流底箱或者字符流对象
        // FileWriter fw = new FileWriter("./demo.txt");
        OutputStreamWriter isw = new OutputStreamWriter(fos, "utf-8");
        // 创建缓冲流对象
        // BufferedWriter bw = new BufferedWriter(fw);
        BufferedWriter bw = new BufferedWriter(isw);
        // 写入数据
        bw.write("北京欢迎你！");
        bw.newLine(); // 换行
        bw.write("welcome to Beijing!");
        // 关闭流
        bw.close();
        isw.close();
        fos.close();
    }
}

// BufferedReader
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class Test {
    public static void main(String[] args) throws IOException {
        // 声明缓冲流对象
        BufferedReader br = null;
        // 创建输入缓冲流对象
        br = new BufferedReader(new InputStreamReader(new FileInputStream("./demo.txt"), "utf-8"));
        // 读取数据，每行读取一行
        String str = null;
        while ((str = br.readLine()) != null) {
            System.out.println(str);
        }
        // 关闭流
        br.close();
    }
}

```

### 转换流

按照流的处理单元将流分为字节流与字符流，而转换流则是在字符流与字节流之间进行转换的流。分为转换输出流`OutputStreamWriter`与转换输入流`InputStreamReader`，转换流是一个处理流，因为它的构造方法中使用到了字节流对象。

- 转换输出流`OutputStreamWriter`

`OutputStreamWriter`直接继承自字符输出流的抽象类`Writer`，是字符流向字节流的桥梁。

常用的构造方法

```java
OutputStreamWriter(OutputStream out, String charsetName)
```

实现

```java
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建字节输出流对象，搭建与目的地之间的桥梁
        OutputStream os = new FileOutputStream("./demo.txt", true);
        // 创建转换输出流对象，将字符流转换成字节流，写入目的地
        OutputStreamWriter w = new OutputStreamWriter(os, "utf-8");
        // 将数据写入目的地
        w.write("北京欢迎你！");
        // 手动刷新缓存
        w.flush();
        // 关闭流
        w.close();
        os.close();
    }
}

```

- 转换输入流`InputStreamReader`

`InputStreamReader`是字符输入流`Reader`的直接子类，是字节流通向字符流的桥梁。

常用构造方法

```
OutputStreamReader(InputStream in, String charsetName)
```

实现

```java
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

public class Test {
    public static void main(String[] args) throws IOException {
       // 声明输入转换流对象
       InputStreamReader isr = null;
       // 创建输入转换流的对象，使用指定的编码格式
       isr =  new InputStreamReader(new FileInputStream("./demo.txt"), "utf-8");
       char[] ch = new char[1024];
       // 读取内容
       int length = isr.read(ch);
       // 判断
       while (length != -1) {
           System.out.println("文件中的内容为：" + new String(ch, 0, length));
           length = isr.read(ch);
       }
       // 关闭流
       isr.close();
    }
}

```

## System类

`System`类位于`java.lang`包中，在使用时不需要导包，其中定义了3个静态常量

```
System.in	标准输入流
System.out	标准输出流
System.err	标准错误流
```

- `System.in`

标准的输入设备是键盘，所以`System.in`指的是从键盘获取的数据，返回值类型`InputStream`.

```java
// 从键盘获取一个字节
import java.io.IOException;
import java.io.InputStream;

public class Test {
    public static void main(String[] args) throws IOException {
        System.out.println("请输入数据：");
        // 创建字节流对象，数据源是键盘
        InputStream is = System.in;
        // 将输入的数据展示
        System.out.println(is.read());
        // 关闭流
        is.close();
    }
}


// 使用缓冲流从键盘获取数据
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class Test {
    public static void main(String[] args) throws IOException {
       // 创建缓冲流对象
       BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
       // 创建StringBuilder对象，用于存储从键盘获取的数据
       StringBuilder sb = new StringBuilder();
       String str = null;
       while (!(str=br.readLine()).equals("bye")) {
           sb.append(str);
       }
       System.out.println("获取的内容为：" + sb);
       // 关闭流
       br.close();
    }
}
```

- `System.out`

标准的输出设备是显示器，`System.out`是将数据输出到显示器上显示 。返回值类型是`PrintStream`，其父类是`OutputStream`。

```java
import java.io.IOException;
import java.io.OutputStream;


public class Test {
    public static void main(String[] args) throws IOException {
       // 获取字节流对象
       OutputStream os = System.out;
       // 输出数据到显示器
       os.write("welecome to beijing!".getBytes());
       // 关闭流
       os.close();
    }
}
```

## Scanner类

导包

```java
import java.util.Scanner
```

构造

```java
Scanner(File source)  // 构造一个新的Scanner，产生从指定文件扫描的值
Scanner(InputStream source)  // 构造一个新的Scanner，产生从指定输入流扫描的值
```

创建

```java
Scanner input = new Scanner(new FileInputStream("./demo.txt"))
Scanner input = new Scanner(System.in)  // 从键盘输入
```

使用

```java
// 文件
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建Scanner对象，指明数据源是文件
        Scanner input = new Scanner(new FileInputStream("./demo.txt"));
        // 读取数据，Scanner类实现了Iterator接口（实现了hasNext()和next()）
        String str = null;
        while (input.hasNext()) { // 判断是否有数据，返回值为true/false
            str = input.next(); 
            System.out.println(str);
        }
        input.close();
    }
}

// 键盘
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Scanner;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建Scanner对象
        Scanner input = new Scanner(System.in);
        // 读取数据，Scanner类实现了Iterator接口（实现了hasNext()和next()）
        String str = null;
        while (!(str = input.next()).equals("bye")) {
            System.out.println(str);
        }
        input.close();
    }
}
```

## 打印流

打印流是输出流的一种，分为字节打印流`PrintStream`与字符打印流`PrintWriter`，常用的方法为`print(),println()`，`PrintStream`的父类为`OutputStream`，`PrintWriter`的父类为`Writer`。

```java
// PrintStream
import java.io.IOException;
import java.io.PrintStream;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建打印流对象，目的地是显示器
        PrintStream ps = System.out;
        // PrintStream中的方法
        ps.print("hello");
        ps.print("world!");
        // write(byte[] buf)是从PrintStream的父类OutputStream中继承
        ps.write("welcome to beijing!".getBytes());
        // 关闭流
        ps.close();
    }
}

// PrintWriter
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建字符打印流对象，目的地是文件
        PrintWriter ps = new PrintWriter(new FileOutputStream("./demo.txt"));
        // 向目的地打印输出数据
        ps.println("hello world");
        ps.write("welcome to beijing!");  // 从父类继承的方法
        // 关闭流
        ps.close();
    }
}
```

## 数据流

`DataInputStream, DataOutputStream`被称为数据流，二者配合使用可以按照与平台无关的方式中从流中读取/写入基本数据类型的数据，还可使用`WriteUTF(String str), redUTF()`方法写入/读取采用utf-8字符编码格式的字符串。

```java
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建数据输出流对象
        DataOutputStream dos = new DataOutputStream(new FileOutputStream("./demo.txt"));
        // 使用数据流向目的地写入数据
        dos.writeInt(100); // 写入一个int类型的值
        dos.writeChar('\n'); // 写入一个换行
        dos.writeBoolean(true); // 写入一个boolean数据
        dos.writeChar('\n'); // 写入一个换行
        dos.writeUTF("北京欢迎你");
        // 关闭流
        dos.close();
        // 创建数据输入流对象
        DataInputStream dis = new DataInputStream(new FileInputStream("./demo.txt"));
        // 使用数据流从数据源读取数据
        System.out.println(dis.readInt()); // 读取int类型
        dis.readChar(); // 读取换行
        System.out.println(dis.readBoolean()); // 读取boolean数据
        dis.readChar(); // 读取换行
        System.out.println(dis.readUTF()); // 读取String类型的值
        // 关闭流
        dis.close();
    }
}
```

## 对象流

对象流是将对象的内存表示形式以二进制的形式存储到目的地，当需要使用的时候再从数据源以二进制的形式还原为对象的内存表示形式。使用数据流可以再网络中进行对象传输。

根据流的方向分为对象输出流`ObjectOutputStream`和对象输入流`ObjectInputStream`。对象流的构造方法需要其他的流对象，所以对象流是一个处理流。

```java
ObjectOutputStream(OutputStream out)
// 构造方法，构造对象输出流对象
ObjectInputStream(InputStream in)
// 构造方法，构造对象输入流对象

void writeObject(Object obj)
// 普通方法，用于将对象写入目的地
Object readObject()
// 普通方法，用于从数据源获取对象
```

- 对象输出流`ObjectOutputStream`

用于将对象的状态以二进制数据的形式输出到目的地，将对象的状态信息输出到目的地的过程也被称为对象的序列化。

```java
// 写入一个对象
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建Person对象
        Person p = new Person("lucy", 12);
        // 创建对象输出流对象
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./demo.txt"));
        // 将Person类的对象p写入目的地
        oos.writeObject(p); // 向上类型转换
        // 关闭流
        oos.close();
    }
}

// 写入一组对象
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

public class Test {
    public static void main(String[] args) throws IOException {
        // 创建集合对象
        ArrayList<Person> al = new ArrayList<Person>();
        // 创建Person对象
        Person p1 = new Person("lucy", 12);
        Person p2 = new Person("jack", 12);
        // 将Person对象添加到集合中
        al.add(p1);
        al.add(p2);
        // 创建对象输出流对象
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("./demo.txt"));
        // 将Person类的对象集合写入目的地
        oos.writeObject(al); // 向上类型转换
        // 关闭流
        oos.close();
    }
}

// Person.java
import java.io.Serializable;

public class Person implements Serializable {
    /**
     * serialVersionUID用于标识写的对象和对的对象是否是同一个对象
     */
    private static final long serialVersionUID = 1L;
    private String name;
    private int age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public Person(String name, int age) {
        super();
        this.name = name;
        this.age = age;
    }

    public Person() {
        super();
    }

}

```



- 对象输入流`ObjectInputStream`

用于将数据源中存储的对象状态信息读取到程序中，对对象的过程被称为反序列化。

```java
// 读取一个对象
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class Test {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // 创建对象输入流对象
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./demo.txt"));
        // 从数据源读取对象
        Person p = (Person) ois.readObject(); // 向下转换
        System.out.println(p.getName() + "\t" + p.getAge());
        // 关闭流
        ois.close();
    }
}
// 读取一组对象
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;

public class Test {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        // 创建对象输入流对象
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream("./demo.txt"));
        // 从数据源读取对象
        ArrayList<Person> al = (ArrayList<Person>) ois.readObject();
        for (Person per : al) {
            System.out.println(per.getName() + "\t" + per.getAge());
        }
        // 关闭流
        ois.close();
    }
}

// Person.java
import java.io.Serializable;

public class Person implements Serializable {
    /**
     * serialVersionUID用于标识写的对象和对的对象是否是同一个对象
     */
    private static final long serialVersionUID = 1L;
    private String name;
    private int age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public Person(String name, int age) {
        super();
        this.name = name;
        this.age = age;
    }

    public Person() {
        super();
    }

}
```

## 字节数组流

字节数组流`ByteArrayInputStream,ByteArrayOutputStream`的数据源和目的地均为`byte`类型的字节数组，该流可以将各种数据类型的数据转换成`byte`类型的数组用于在网络中进行各种数据传输。

```java
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class Test {
    public static void main(String[] args) throws Exception {
        // 将各种数据类型转换成byte类型数组
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(bos);
        oos.writeInt(100);
        oos.writeUTF("hello");
        oos.writeBoolean(true);
        oos.writeObject(new Person());
        // 将各种数据类型的数据转换成byte类型的数组
        byte[] buf = bos.toByteArray();
        ByteArrayInputStream bis = new ByteArrayInputStream(buf);
        ObjectInputStream ois = new ObjectInputStream(bis);
        System.out.println(ois.readInt());
        System.out.println(ois.readUTF());
        System.out.println(ois.readBoolean());
        System.out.println(ois.readObject());
    }
}

// Person.java
import java.io.Serializable;

public class Person implements Serializable {
    /**
     * serialVersionUID用于标识写的对象和对的对象是否是同一个对象
     */
    private static final long serialVersionUID = 1L;
    private String name;
    private int age;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public Person(String name, int age) {
        super();
        this.name = name;
        this.age = age;
    }

    public Person() {
        super();
    }

}
```

