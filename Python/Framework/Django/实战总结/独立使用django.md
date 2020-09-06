# 独立使用

若要独立使用django采用如下配置

```python
#独立使用django的model
import sys
import os

# 当前文件路径
pwd = os.path.dirname(os.path.realpath(__file__))
# 将项目根目录追加至python搜索路径中
sys.path.append(pwd+"../")
# django项目配置文件
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MxShop.settings")

import django
django.setup()

...
```

