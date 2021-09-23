# pyjwt

[文档](https://pyjwt.readthedocs.io/en/stable/installation.html)

PyJWT 是一个 Python 库，它允许您对 JSON Web 令牌 (JWT) 进行编码和解码。 JWT 是一种开放的行业标准 (RFC 7519)，用于在两方之间安全地表示声明。

## 安装

```
pip install pyjwt
```

## 使用

### hs256

```shell
>>> import jwt
>>> key = "secret"
>>> encoded = jwt.encode({"some": "payload"}, key, algorithm="HS256")
>>> print(encoded)
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb21lIjoicGF5bG9hZCJ9.4twFt5NiznN84AWoo1d7KO1T_yoc0Z6XOpOVswacPZg
>>> jwt.decode(encoded, key, algorithms="HS256")
{'some': 'payload'}
```

### RS256(RSA)

```shell
>>> import jwt
>>> private_key = b"-----BEGIN PRIVATE KEY-----\nMIGEAgEAMBAGByqGSM49AgEGBS..."
>>> public_key = b"-----BEGIN PUBLIC KEY-----\nMHYwEAYHKoZIzj0CAQYFK4EEAC..."
>>> encoded = jwt.encode({"some": "payload"}, private_key, algorithm="RS256")
>>> print(encoded)
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb21lIjoicGF5bG9hZCJ9.4twFt5NiznN84AWoo1d7KO1T_yoc0Z6XOpOVswacPZg
>>> decoded = jwt.decode(encoded, public_key, algorithms=["RS256"])
{'some': 'payload'}
```

如果您的私钥需要密码短语，您需要从`cryptography`传入一个 `PrivateKey` 对象。

```python
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

pem_bytes = b"-----BEGIN PRIVATE KEY-----\nMIGEAgEAMBAGByqGSM49AgEGBS..."
passphrase = b"your password"

private_key = serialization.load_pem_private_key(
    pem_bytes, password=passphrase, backend=default_backend()
)
encoded = jwt.encode({"some": "payload"}, private_key, algorithm="RS256")
```

### 特定header

```shell
>>> jwt.encode(
...     {"some": "payload"},
...     "secret",
...     algorithm="HS256",
...     headers={"kid": "230498151c214b788dd97f22b85410a5"},
... )
'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjIzMDQ5ODE1MWMyMTRiNzg4ZGQ5N2YyMmI4NTQxMGE1In0.eyJzb21lIjoicGF5bG9hZCJ9.DogbDGmMHgA_bU05TAB-R6geQ2nMU2BRM-LnYEtefwg'
```

### 无验证读取声明集

如果您希望在不验证签名或任何已注册声明名称的情况下读取 JWT 的声明集，您可以将 verify_signature 选项设置为 False。

注意：除非您清楚地了解什么，否则使用此功能通常是不明智的。如果没有数字签名信息，则无法信任声明集的完整性或真实性。

```shell
>>> jwt.decode(encoded, options={"verify_signature": False})
{'some': 'payload'}
```

### 无验证读取Heades

某些 API 要求您在未经验证的情况下读取 JWT 标头。例如，在令牌发行者使用多个密钥而您无法事先知道发行者的公钥或共享秘密中的哪一个用于验证的情况下，发行者可能会在标头中包含密钥的标识符。

```shell
>>> jwt.get_unverified_header(encoded)
{'alg': 'RS256', 'typ': 'JWT', 'kid': 'key-id-12345...'}
```

### 注册声明名称

JWT 规范定义了一些已注册的声明名称并定义了它们的使用方式。 PyJWT 支持这些已注册的声明名称

```
“exp” (Expiration Time) Claim
“nbf” (Not Before Time) Claim
“iss” (Issuer) Claim
“aud” (Audience) Claim
“iat” (Issued At) Claim
```

- exp

“exp”声明标识了 JWT 不得被接受处理的过期时间或之后。 “exp”声明的处理要求当前日期/时间必须在“exp”声明中列出的到期日期/时间之前。实施者可以提供一些小的余地，通常不超过几分钟，以解决时钟偏差。它的值必须是一个包含 NumericDate 值的数字。此声明的使用是可选的。

您可以将到期时间作为 UTC UNIX 时间戳（整数）或日期时间传递，后者将被转换为整数。例如：

```python
jwt.encode({"exp": 1371720939}, "secret")
jwt.encode({"exp": datetime.utcnow()}, "secret")
```

过期时间在 `jwt.decode() `中自动验证，如果过期时间在过去，则会引发 jwt.ExpiredSignatureError ：

```python
try:
    jwt.decode("JWT_STRING", "secret", algorithms=["HS256"])
except jwt.ExpiredSignatureError:
    # Signature has expired
    ...
```

到期时间将与当前 UTC 时间进行比较（由 `timegm(datetime.utcnow().utctimetuple()) `给出），因此请确保在编码中使用 UTC 时间戳或日期时间。

您可以使用 options 参数中的 verify_exp 参数关闭过期时间验证。

PyJWT 还支持过期时间定义的 leeway 部分，这意味着您可以验证过去但不是很远的过期时间。例如，如果您有一个 JWT 负载，其过期时间设置为创建后 30 秒，但您知道有时您会在 30 秒后处理它，您可以设置 10 秒的余地以留出一些余量：

```python
jwt_payload = jwt.encode(
    {"exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=30)}, "secret"
)

time.sleep(32)

# JWT payload is now expired
# But with some leeway, it will still validate
jwt.decode(jwt_payload, "secret", leeway=10, algorithms=["HS256"])
```

可以使用 datetime.timedelta 实例，而不是将余地指定为秒数。上面例子中的最后一行相当于：

```python
jwt.decode(
    jwt_payload, "secret", leeway=datetime.timedelta(seconds=10), algorithms=["HS256"]
)
```

- nbf

“nbf”声明标识了不得接受 JWT 进行处理的时间。 “nbf”声明的处理要求当前日期/时间必须晚于或等于“nbf”声明中列出的日期/时间。实施者可以提供一些小的余地，通常不超过几分钟，以解决时钟偏差。它的值必须是一个包含 NumericDate 值的数字。此声明的使用是可选的。

```python
jwt.encode({"nbf": 1371720939}, "secret")
jwt.encode({"nbf": datetime.utcnow()}, "secret")
```

- iss

“iss”声明标识了颁发 JWT 的委托人。此声明的处理通常是特定于应用程序的。 “iss”值是包含 StringOrURI 值的区分大小写的字符串。此声明的使用是可选的。

```python
payload = {"some": "payload", "iss": "urn:foo"}

token = jwt.encode(payload, "secret")
decoded = jwt.decode(token, "secret", issuer="urn:foo", algorithms=["HS256"])
```

- aud

“aud”声明标识了 JWT 的目标接收者。每个打算处理 JWT 的主体必须使用受众声明中的值来标识自己。如果处理该声明的委托人在此声明存在时未使用“aud”声明中的值标识自己，则必须拒绝 JWT。

在一般情况下，“aud”值是一个区分大小写的字符串数组，每个字符串包含一个 StringOrURI 值。

```python
payload = {"some": "payload", "aud": ["urn:foo", "urn:bar"]}

token = jwt.encode(payload, "secret")
decoded = jwt.decode(token, "secret", audience="urn:foo", algorithms=["HS256"])
```

在 JWT 有一个受众的特殊情况下，“aud”值可以是包含 StringOrURI 值的单个区分大小写的字符串。

```python
payload = {"some": "payload", "aud": "urn:foo"}

token = jwt.encode(payload, "secret")
decoded = jwt.decode(token, "secret", audience="urn:foo", algorithms=["HS256"])
```

如果接受多个受众，则 jwt.decode 的受众参数也可以是可迭代的

```python
payload = {"some": "payload", "aud": "urn:foo"}

token = jwt.encode(payload, "secret")
decoded = jwt.decode(
    token, "secret", audience=["urn:foo", "urn:bar"], algorithms=["HS256"]
)
```

受众价值的解释通常是特定于应用程序的。此声明的使用是可选的。如果受众声明不正确，将引发` jwt.InvalidAudienceError`。

- iat

iat (issued at) 声明标识了 JWT 的发布时间。此声明可用于确定 JWT 的年龄。它的值必须是一个包含 NumericDate 值的数字。此声明的使用是可选的。

如果 iat 声明不是数字，则会引发 jwt.InvalidIssuedAtError 异常。

```python
jwt.encode({"iat": 1371720939}, "secret")
jwt.encode({"iat": datetime.utcnow()}, "secret")
```

### 要求存在声明

如果您希望声明集中存在一个或多个声明，您可以设置 require 参数以包含这些声明。

```shell
>>> jwt.decode(encoded, options={"require": ["exp", "iss", "sub"]})
{'exp': 1371720939, 'iss': 'urn:foo', 'sub': '25c37522-f148-4cbf-8ee6-c4a9718dd0af'}
```

### 从 JWKS 端点解析RSA 签名密钥

```shell
>>> import jwt
>>> from jwt import PyJWKClient
>>> token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImtpZCI6Ik5FRTFRVVJCT1RNNE16STVSa0ZETlRZeE9UVTFNRGcyT0Rnd1EwVXpNVGsxUWpZeVJrUkZRdyJ9.eyJpc3MiOiJodHRwczovL2Rldi04N2V2eDlydS5hdXRoMC5jb20vIiwic3ViIjoiYVc0Q2NhNzl4UmVMV1V6MGFFMkg2a0QwTzNjWEJWdENAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vZXhwZW5zZXMtYXBpIiwiaWF0IjoxNTcyMDA2OTU0LCJleHAiOjE1NzIwMDY5NjQsImF6cCI6ImFXNENjYTc5eFJlTFdVejBhRTJINmtEME8zY1hCVnRDIiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIn0.PUxE7xn52aTCohGiWoSdMBZGiYAHwE5FYie0Y1qUT68IHSTXwXVd6hn02HTah6epvHHVKA2FqcFZ4GGv5VTHEvYpeggiiZMgbxFrmTEY0csL6VNkX1eaJGcuehwQCRBKRLL3zKmA5IKGy5GeUnIbpPHLHDxr-GXvgFzsdsyWlVQvPX2xjeaQ217r2PtxDeqjlf66UYl6oY6AqNS8DH3iryCvIfCcybRZkc_hdy-6ZMoKT6Piijvk_aXdm7-QQqKJFHLuEqrVSOuBqqiNfVrG27QzAPuPOxvfXTVLXL2jek5meH6n-VWgrBdoMFH93QEszEDowDAEhQPHVs0xj7SIzA"
>>> kid = "NEE1QURBOTM4MzI5RkFDNTYxOTU1MDg2ODgwQ0UzMTk1QjYyRkRFQw"
>>> url = "https://dev-87evx9ru.auth0.com/.well-known/jwks.json"
>>> jwks_client = PyJWKClient(url)
>>> signing_key = jwks_client.get_signing_key_from_jwt(token)
>>> data = jwt.decode(
...     token,
...     signing_key.key,
...     algorithms=["RS256"],
...     audience="https://expenses-api",
...     options={"verify_exp": False},
... )
>>> print(data)
{'iss': 'https://dev-87evx9ru.auth0.com/', 'sub': 'aW4Cca79xReLWUz0aE2H6kD0O3cXBVtC@clients', 'aud': 'https://expenses-api', 'iat': 1572006954, 'exp': 1572006964, 'azp': 'aW4Cca79xReLWUz0aE2H6kD0O3cXBVtC', 'gty': 'client-credentials'}
```

