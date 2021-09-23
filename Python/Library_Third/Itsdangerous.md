# Itsdangerous

[参考](https://itsdangerous.palletsprojects.com/en/2.0.x/)

有时您想将一些数据发送到不受信任的环境，然后再将其取回。为了安全地执行此操作，必须对数据进行签名以检测更改。

给定只有您知道的密钥，您可以对数据进行加密签名并将其交给其他人。当你取回数据时，你可以确保没有人篡改它。

接收者可以看到数据，但除非他们也有你的密钥，否则他们无法修改它。所以如果你保持密钥的秘密和复杂，你会没事的。

它提供了两个级别的数据处理。签名接口是基于给定签名参数对给定字节值进行签名的基本系统。序列化接口包装了一个签名者，以启用对字节以外的其他数据的序列化和签名。

通常，您需要使用序列化程序，而不是签名者。您可以通过序列化程序配置签名参数，甚至可以提供回退签名者以将旧令牌升级为新参数。

## 适用案例
-  在取消订阅某个通讯时，你可以在URL里序列化并且签名一个用户的ID。这种情况下你不需要生成一个一次性的token并把它们存到数据库中。在任何的激活账户的链接或类似的情形下，同样适用。
-  被签名的对象可以被存入cookie中或其他不可信来源，这意味着你不需要在服务端保存session，这样可以降低数据库读取的次数。
-  通常签名后的信息可以安全地往返与服务端与客户端之间，这个特性可以用于将服务端的状态传递到客户端再传递回来。

## 安装
```shell
pip install -U itsdangerous
```

## 密钥/盐

### 密钥

签名由`secret_key`保护。通常，所有签名者都使用一个密钥，盐用于区分不同的上下文。更改密钥将使现有令牌无效。

它应该是一长串随机字节。此值必须保密，不应保存在源代码中或提交给版本控制。如果攻击者获悉密钥，他们可以更改和放弃数据以使其看起来有效。如果您怀疑发生了这种情况，请更改密钥以使现有令牌无效。

将密钥分开的一种方法是从环境变量中读取它。首次部署时，在运行应用程序时生成密钥并设置环境变量。所有进程管理器（如 systemd）和托管服务都有指定环境变量的方法。

```python
import os
from itsdangerous.serializer import Serializer
SECRET_KEY = os.environ.get("SECRET_KEY")
s = Serializer(SECRET_KEY)
```

产生密钥的一种方法

```shell
python3 -c 'import os; print(os.urandom(16).hex())'
```

### 盐

盐与秘钥相结合，推导出唯一的密钥，用于区分不同的上下文。与密钥不同，盐不必是随机的，可以保存在代码中。它只需要在上下文之间是唯一的，而不是私有的。

例如，您希望通过电子邮件发送激活链接以激活用户帐户，以及升级链接以将用户升级到付费帐户。如果您签署的只是用户 ID，并且您不使用不同的盐，则用户可以重复使用激活链接中的令牌来升级帐户。如果您使用不同的盐，签名将不同，并且在其他上下文中无效。

```python
from itsdangerous.url_safe import URLSafeSerializer

s1 = URLSafeSerializer("secret-key", salt="activate")
s1.dumps(42)
'NDI.MHQqszw6Wc81wOBQszCrEE_RlzY'

s2 = URLSafeSerializer("secret-key", salt="upgrade")
s2.dumps(42)
'NDI.c0MpsD6gzpilOAeUPra3NShPXsE'
```

由于盐不同，第二个序列化程序无法加载第一个序列化程序转储的数据。

```shell
s2.loads(s1.dumps(42))
Traceback (most recent call last):
  ...
BadSignature: Signature does not match
```

只有使用相同盐的序列化器才能成功把值加载出来：

```shell
s2.loads(s2.dumps(42))
42
```

### 密钥轮换

密钥轮换可以为发现秘密密钥的攻击者提供额外的缓解措施。轮换系统将保留有效密钥列表，生成新密钥并定期删除最旧的密钥。如果攻击者破解密钥需要 4 周时间，但密钥在 3 周后轮换使用，他们将无法使用他们破解的任何密钥。但是，如果用户在三周内没有刷新他们的令牌，它也将无效。

生成和维护此列表的系统不在 ItsDangerous 的范围内，但 ItsDangerous 确实支持针对密钥列表进行验证。

不是传递一个单个键，您可以传递一个从最旧到最新的键列表。签名时将使用最后一个（最新）密钥，并且在验证每个密钥时，将在引发验证错误之前从最新到最旧尝试。

```python
SECRET_KEYS = ["2b9cd98e", "169d7886", "b6af09f5"]

# sign some data with the latest key
s = Serializer(SECRET_KEYS)
t = s.dumps({"id": 42})

# rotate a new key in and the oldest key out
SECRET_KEYS.append("cf9b3588")
del SECRET_KEYS[0]

s = Serializer(SECRET_KEYS)
s.loads(t)  # valid even though it was signed with a previous key
```

### 摘要方法安全

签名者配置了一个`digest_method`，这是一个哈希函数，在生成HMAC 签名时用作中间步骤。默认方法是`hashlib.sha1()`。有时，用户会担心这种默认设置，因为他们听说过与 SHA-1 的哈希冲突。

当用作 HMAC 中的中间迭代步骤时，SHA-1 不是不安全的。事实上，即使是 MD5 在 HMAC 中仍然是安全的。在 HMAC 中使用时，哈希本身的安全性并不适用。

如果项目认为 SHA-1 是一种风险，他们可以使用不同的摘要方法配置签名者，例如 `hashlib.sha512()`。可以配置 SHA-1 的后备签名者，以便升级旧令牌。 SHA-512 生成更长的哈希值，因此令牌将占用更多空间，这与 cookie 和 URL 相关。

## 序列化接口

签名接口只对字节进行签名。为了对其他类型进行签名，Serializer 类提供了一个类似于 Python 的 json 模块的转储/加载接口，它将对象序列化为一个字符串，然后对其进行签名。

使用 `dumps() `对数据进行序列化和签名：

```python
from itsdangerous.serializer import Serializer
s = Serializer("secret-key")
s.dumps([1, 2, 3, 4])
b'[1, 2, 3, 4].r7R9RhGgDPvvWl3iNzLuIIfELmo'
```

使用`loads()` 来验证签名并反序列化数据。

```python
s.loads('[1, 2, 3, 4].r7R9RhGgDPvvWl3iNzLuIIfELmo')
[1, 2, 3, 4]
```

默认情况下，数据使用内置的 json 模块序列化为 JSON。这个内部序列化器可以通过子类化来改变。

### 对失败的响应

异常具有有用的属性，允许您在签名检查失败时检查有效负载。这必须格外小心，因为那时您知道有人篡改了您的数据，但这可能对调试有用。

```python
from itsdangerous.serializer import URLSafeSerializer 
from itsdangerous.exc import BadSignature, BadData

s = URLSafeSerializer('secret-key')
decoded_payload = None
try:
    decoded_payload = s.loads(data)
    # This payload is decoded and safe
except BadSignature, e:
    if e.payload is not None:
        try:
            decoded_payload = s.load_payload(e.payload)
        except BadData:
            pass
        # 这里的数据可以解码出来，但不是安全的，因为有某人改动了签名。
        # 解码步骤(load_payload)是显式的，因为将数据反序列化可能是不安全的
        #（请设想被解码的不是json,而是pickle）
```

如果你不想检查到底是哪里出错了，你也可以使用不安全的加载方式:

```
sig_okay, payload = s.loads_unsafe(data)
```

返回的元组中第一项是一个布尔值，表明了签名是否是正确的。

### 后备签名

您可能希望在不立即使现有签名无效的情况下升级签名参数。例如，您可能决定要使用不同的摘要方法。新签名应使用新方法，但旧签名仍应进行验证。

可以提供`fallback_signers`列表，如果与当前签名者取消签名失败，将尝试该列表。列表中的每一项可以是

- `signer_kwargs`字典，用于实例化传递给序列化程序的签名者类。
- 使用传递给序列化程序的 `secret_key,salt,signer_kwargs` 实例化的 `Signer` 类。
- ` (signer_class, signer_kwargs)` 的元组使用给定的变量实例化给定的类。

例如，这是一个使用 SHA-512 签名的序列化程序，但将使用 SHA-512 或 SHA-1 取消签名

```python
s = Serializer(
    signer_kwargs={"digest_method": hashlib.sha512},
    fallback_signers=[{"digest_method": hashlib.sha1}]
)
```

## 签名接口

最基本的接口是签名接口。Signer类可以用来将一个签名附加到指定的字符串上：
```python
from itsdangerous import signer
s = Signer("secret-key")
s.sign("my string")
b'my string.wh6tMHxLgJqB6oY1uT73iMlyrOA'
```
签名会被加在字符串尾部，中间由句号 (.)分隔。验证字符串，使用`unsign()`方法：
```shell
s.unsign('my string.wh6tMHxLgJqB6oY1uT73iMlyrOA')
b'my string'
```
如果被签名的是一个unicode字符串，那么它将隐式地被转换成utf-8。然而，在反签名时，你没法知道它原来是unicode还是字节串。

如果该值发生更改，则签名将不再匹配，并且取消签名将引发 BadSignature 异常：
```python
s.unsign(b"different string.wh6tMHxLgJqB6oY1uT73iMlyrOA")
Traceback (most recent call last):
  ...
BadSignature: Signature does not match
```

## 使用时间戳签名
如果你想要可以过期的签名，可以使用 TimestampSigner 类，它会加入时间戳信息并签名。在反签名时，你可以验证时间戳有没有过期：
```python
from itsdangerous.timed import TimestampSigner
s = TimestampSigner('secret-key')
string = s.sign('foo')

s.unsign(string, max_age=5)
Traceback (most recent call last):
  ...
itsdangerous.exc.SignatureExpired: Signature age 15 > 5 seconds
```
## URL安全序列化
如果能够向只有字符受限的环境中传递可信的字符串的话，将十分有用。因此，itsdangerous也提供了一个URL安全序列化工具：
```python
from itsdangerous.url_safe import URLSafeSerializer
s = URLSafeSerializer("secret-key")
s.dumps([1, 2, 3, 4])
'WzEsMiwzLDRd.wSPHqC0gR7VUqivlSukJ0IeTDgo'
s.loads("WzEsMiwzLDRd.wSPHqC0gR7VUqivlSukJ0IeTDgo")
[1, 2, 3, 4]
```

## JSON Web 签名

目前弃用

从“itsdangerous” 0.18版本开始，也支持了JSON Web签名。它们的工作方式与原有的URL安全序列化器差不多，但是会根据当前JSON Web签名（JWS）草案（10） [draft-ietf-jose-json-web-signature] 来生成header。
```
>>> from itsdangerous import JSONWebSignatureSerializer
>>> s = JSONWebSignatureSerializer('secret-key')
>>> s.dumps({'x': 42})
>>> 'eyJhbGciOiJIUzI1NiJ9.eyJ4Ijo0Mn0.ZdTn1YyGz9Yx5B5wNpWRL221G1WpVE5fPCPKNuc6UAo'
```
在将值加载回来时，默认会像其他序列化器一样，不会返回header。但是你可以通过传入 return_header=True 参数来得到header。 Custom header fields can be provided upon serialization:
```
>>> s.dumps(0, header_fields={'v': 1})
>>> 'eyJhbGciOiJIUzI1NiIsInYiOjF9.MA.wT-RZI9YU06R919VBdAfTLn82_iIQD70J_j-3F4z_aM'
>>> s.loads('eyJhbGciOiJIUzI1NiIsInYiOjF9.MA.wT-RZI9YU06R919VBdAf'
>>> ...         'TLn82_iIQD70J_j-3F4z_aM', return_header=True)
>>> ...
>>> (0, {u'alg': u'HS256', u'v': 1})
```
itsdangerous目前只提供HMAC SHA的派生算法以及不使用算法，不支持基于ECC的算法。header中的算法将与序列化器中的进行核对，如果不匹配，将引发 BadSignature 异常。

## python3中注意
在Python 3中，itsdangerous的接口在一开始可能让人困扰。由于它封装在内部的序列化器，函数返回值不一定是unicode字符串还是字节对象。内置的签名器总是基于字节的。

这是为了允许模块操作不同的序列化器，独立于它们的实现方式。模块通过执行一个空对象的序列化，来决定使用哪种序列化器。

