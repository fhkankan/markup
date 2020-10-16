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

## Scanner类

## 打印流

## 数据流

## 对象流

## 字节数组流