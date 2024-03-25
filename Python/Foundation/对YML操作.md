# YML

## 语法规范

[文档](https://www.yiibai.com/yaml/)
注释规范

```
使用 #作为注释开始，YAML中只有行注释。
```

语法规范

```
1）配置大小写敏感；
2）使用缩进代表层级关系；
3）缩进只能使用空格，不能使用TAB，不要求空格个数，只需要相同层级左对齐（一般2个或4个空格）；
```

数据类型

```
纯量：单个的不可再分的值；
数组：一组按次序排列的值，又称为序列（sequence） / 列表（list）；
对象：键值对的集合，又称为映射（mapping）/ 哈希（hashes） / 字典（dictionary）；
```

示例

```yml
# 对象，冒号后加空格
key: value


# 数组
mysql_A:
  host : 77.77.77.77
  port : 5432
  dbname : db_name_a
  user : user_name
  passwd : password_a
```

## python操作

[文档](https://pyyaml.org/wiki/PyYAMLDocumentation)

安装

```
pip install pyyaml
```

实现

```python
import os.path

import yaml

pwd_path = os.path.abspath(".")
file_path = os.path.join(pwd_path, "dbEnvConf.yaml")

with open(file_path,mode="r",encoding="utf-8") as f:
    yamlConf = yaml.safe_load(f)
    yamlConf = yaml.load(f, Loader=yaml.FullLoader)

    print(yamlConf)



```

