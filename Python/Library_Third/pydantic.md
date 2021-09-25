# pydantic

> v1.8.2

[参考](https://pydantic-docs.helpmanual.io)

[githup](https://github.com/samuelcolvin/pydantic/)

使用 python 类型注释的数据验证和设置管理。

pydantic 在运行时强制执行类型提示，并在数据无效时提供用户友好的错误。

定义数据应该如何在纯、规范的 python 中；用 pydantic 验证它。

## 安装

标准安装

```shell
pip install pydantic

conda install pydantic -c conda-forge
```

扩展安装

```shell
pip install pydantic[email]  # 对email检查
# or
pip install pydantic[dotenv]  # 对.env文件支持
# or just
pip install pydantic[email,dotenv]
```

## 使用

### 模型

在 pydantic 中定义对象的主要方法是通过模型（模型只是继承自 BaseModel 的类）。

您可以将模型视为类似于严格类型语言中的类型，或者作为 API 中单个端点的要求。

不可信数据可以传递给模型，经过解析和验证后，pydantic 保证生成的模型实例的字段将符合模型上定义的字段类型。

