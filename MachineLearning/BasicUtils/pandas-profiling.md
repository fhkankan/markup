# pandas-profiling

从pandas DataFrame生成配置文件报告。`pandas df.describe()`函数很棒，但对于严肃的探索性数据分析来说有点基础。pandas_profiling使用`df.profile_report()`扩展pandas DataFrame，以进行快速数据分析。

安装

```shell
pip install pandas-profiling
```

## 使用

jupyter notebook

```python
import numpy as np
import pandas as pd
import pandas_profiling

df = pd.DataFrame(
    np.random.rand(100, 5),
    columns=['a', 'b', 'c', 'd', 'e']
)

# 展示报告
df.profile_report()  
# 检索出由于高相关性而被拒绝的变量列表
rejected_variables = profile.get_rejected_variables(threshold=0.9)
# 生成html报告
profile = df.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file="output.html")
```

