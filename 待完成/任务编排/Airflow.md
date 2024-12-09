# Airflow

[文档](https://airflow.apache.org/docs/apache-airflow/stable/index.html)

## 概述

Apache Airflow是一个开源平台，用于开发、调度和监控面向批处理的工作流。Airflow的可扩展Python框架使您能够构建与几乎任何技术连接的工作流。web界面有助于管理工作流的状态。Airflow 可以通过多种方式部署，从笔记本电脑上的单线程到分布式设置，甚至可以支持最大的工作流程。

Airflow工作流的主要特征是所有工作流都是用Python代码定义的。“工作流即代码”有几个目的：

`Dynamic`：Airflow管道配置使用python，这允许编写可动态实例化管道的代码。

`Extensible`：Airflow框架包含许多运算符来连接各种技术。Airflow的所有组件都是可扩展的。轻松适合于您的环境。

`Flexible`：使用功能强大的`Jinja`模板引擎，将脚本参数化内置于Airflow的核心中。

- 优点

Airflow是一个批处理工作流编排平台，包含了许多运算符来连接各种技术，并且很容易扩展以与新技术连接。如果工作流有明确的开始和结束，并且定期执行，可以将它们编程为Airflow的DAG。

如果喜欢编码甚于点击，Airflow是更合适的工具。工作流被定义为python代码，具有一下优势：1.工作流可以做版本控制以便回滚到之前版本；2.工作流可以由多人同时开发；3.可以编写测试来验证功能；4.组件可扩展，可以基于现有广泛的组件进行构建。

丰富的调度和执行语义使您能够轻松定义以规则间隔运行的复杂管道。回填允许您在更改逻辑后对历史数据（重新）运行管道。在解决错误后重新运行部分管道的能力有助于最大限度地提高效率。

Airflow的用户界面提供：1、深入展示两件事：管道和任务；2.随时间推移的管道概览。从界面上，可以检查日志和管理任务，例如在失败时重试任务。

- 缺点

Airflow是为有限批处理工作流程而构建的。虽然`CLI`和`REST API`确实允许触发工作流，但Airflow并不是为无限运行基于事件的工作流而构建的。Airflow不是流式解决方案。然而，像Apache Kafka这样的流数据系统经常与Apache Airflow一起工作。Kafka可用于实时摄取和处理，事件数据被写入存储位置，Airflow定期启动工作流处理一批数据。

如果你更喜欢点击而不是编码，那么Airflow可能不是正确的解决方案。web界面旨在使管理工作流程尽可能简单，Airflow框架不断改进以使开发人员体验尽可能流畅。然而，Airflow的理念是将工作流程定义为代码，因此始终需要编码。

## 安装

pypi

```
# constrains文件固定了依赖的版本号，便于配合主包部署应用
pip install 'apache-airflow==2.10.3' \
 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.3/constraints-3.9.txt"
```

## 使用

### 管道定义

示例

```python
# airflow/example_dags/tutorial.py
import textwrap
from datetime import datetime, timedelta
# The DAG object; we'll need this to instantiate a DAG
from airflow.models.dag import DAG
# Operators; we need this to operate!
from airflow.operators.bash import BashOperator


with DAG(
    "tutorial",
    # These args will get passed on to each operator
    # You can override them on a per-task basis during operator initialization
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        # 'queue': 'bash_queue',
        # 'pool': 'backfill',
        # 'priority_weight': 10,
        # 'end_date': datetime(2016, 1, 1),
        # 'wait_for_downstream': False,
        # 'sla': timedelta(hours=2),
        # 'execution_timeout': timedelta(seconds=300),
        # 'on_failure_callback': some_function, # or list of functions
        # 'on_success_callback': some_other_function, # or list of functions
        # 'on_retry_callback': another_function, # or list of functions
        # 'sla_miss_callback': yet_another_function, # or list of functions
        # 'on_skipped_callback': another_function, #or list of functions
        # 'trigger_rule': 'all_success'
    },
    description="A simple tutorial DAG",
    schedule=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:

    # t1, t2 and t3 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
    )

    t2 = BashOperator(
        task_id="sleep",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=3,
    )
    t1.doc_md = textwrap.dedent(
        """\
    #### Task Documentation
    You can document your task using the attributes `doc_md` (markdown),
    `doc` (plain text), `doc_rst`, `doc_json`, `doc_yaml` which gets
    rendered in the UI's Task Instance Details page.
    ![img](https://imgs.xkcd.com/comics/fixing_problems.png)
    **Image Credit:** Randall Munroe, [XKCD](https://xkcd.com/license.html)
    """
    )

    dag.doc_md = __doc__  # providing that you have a docstring at the beginning of the DAG; OR
    dag.doc_md = """
    This is a documentation placed anywhere
    """  # otherwise, type it like this
    templated_command = textwrap.dedent(
        """
    {% for i in range(5) %}
        echo "{{ ds }}"
        echo "{{ macros.ds_add(ds, 7)}}"
    {% endfor %}
    """
    )

    t3 = BashOperator(
        task_id="templated",
        depends_on_past=False,
        bash_command=templated_command,
    )

    t1 >> [t2, t3]
```

