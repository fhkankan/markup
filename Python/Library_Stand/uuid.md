# UUID

## 概述

UUID（Universally Unique Identity）的缩写，是一种软件建构的标准，通常由32字节16进制数表示（128位），它可以保证时间和空间的唯一性。目前应用最广泛的UUID事微软的GUIDs。

## 作用

UUID让分布式系统所有元素都有唯一的辨识信息，让每个人都可以建立与他人不同的UUID，不需考虑数据库建立时的名称重复问题。随机生成字符串，当成token、用户账号、订单等。

## 原理

UUID是指一台机器上生成的数字，他保证同一时空所有机器都是唯一的。

UUID由以下几部分构成：
```
- 时间戳：根据当前时间或者时钟序列生成字符串
- 全剧唯一的机器识别号，根据网卡MAC地址或者IP获取，如果没有网卡则以其他方式获取。
- 随机数：机器自动随机一组序列
```

## 算法

uuid有5种生成算法，分别是uuid1()、uuid2()、uuid3()、uuid4()、uuid5()。

- uuid1基于时间戳

由MAC地址、当前时间戳、随机数字。保证全球范围内的唯一性。但是由于MAC地址使用会带来安全问题，局域网内使用IP代替MAC

- uuid2基于分布式环境DCE

算法和uuid1相同，不同的是把时间戳前四位换成POIX的UID，实际很少使用。注意：python中没有这个函数

- uuid3基于名字和MD5散列值

通过计算名字和命名空间的MD5散列值得到的，保证了同一命名空间中不同名字的唯一性，不同命名空间的唯一性。但是同一命名空间相同名字生成相同的uuid。

- uuid4基于随机数

由伪随机数得到的，有一定重复概率，这个概率是可以算出来的

- uuid5基于名字和SAHI值

算法和uuid3相同，不同的是使用SAHI算法　

## 使用

经验

```
- 由于python中没有DCE，所以uuid2()可以忽略
- uuid4()存在概率重复性，由于无映射性，最好不使用
- 如果是全局的分布式环境下，最好使用uuid1()
- 若名字的唯一性要求，最好使用uuid3()或者uuid5()
```

使用

```python
import uuid

res = str(uuid.uuid1())
res = str(uuid.uuid3(uuid.NAMESPACE_DNS, 'hello'))
res = str(uuid.uuid5(uuid.NAMESPACE_DNS, 'hello'))
```

## API

属性

| name       | desc                                                         |
| ---------- | ------------------------------------------------------------ |
| `bytes`    | UUID为16字节的字符串（包含按big-endian字节顺序排列的六个整数字段）。 |
| `bytes_le` | UUID为16字节的字符串（time_low，time_mid和time_hi_version以小尾数字节顺序排列）。 |
| `fields`   | UUID的六个整数字段的元组，也可以作为六个单独的属性和两个派生的属性使用 |
| `hex`      | UUID为32个字符的十六进制字符串。                             |
| `urn`      |                                                              |
| `is_safe`  | SafeUUID的枚举，指示平台是否以多处理安全的方式生成UUID。(3.7新增) |

使用

```python
from uuid import uuid1

res = uuid1().hex
```

