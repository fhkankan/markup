# Haystack

Haystack 是在Django中对接搜索引擎的框架，搭建了用户和搜索引擎之间的沟通桥梁。

我们在Django中可以通过使用 Haystack 来调用 Elasticsearch 搜索引擎。

Haystack 可以在不修改代码的情况下使用不同的搜索后端（比如 `Elasticsearch`、`Whoosh`、`Solr`等等）。

[官方文档](https://django-haystack.readthedocs.io/en/master/tutorial.html#installation)

[参考文档1](https://my.oschina.net/u/4354181/blog/3469030)

## 安装

```shell
pip install django-haystack

# 安装对应的后端搜索引擎
# elasticsearch，对于中文分词需安装elasticsearch时安装插件elasticsearch-analysis-ik
pip install elasticsearch==2.4.1 
# whoosh，对于中文分词需要特殊处理，见参考文档1或django实战案例介绍
pip install whoosh 
pip install jieba
```

## 配置

`settings.py`

```python
INSTALLED_APPS = [
    'haystack', # 全文检索
]


# Haystack
HAYSTACK_CONNECTIONS = {
    # 使用elasticsearch
    'default': {
        'ENGINE': 'haystack.backends.elasticsearch_backend.ElasticsearchSearchEngine',
        'URL': 'http://192.168.103.158:9200/', # Elasticsearch服务器ip地址，端口号固定为9200
        'INDEX_NAME': 'haystack', # Elasticsearch建立的索引库的名称
    },
    # 使用whoosh
    'default': {
        # 默认英文
        'ENGINE': 'haystack.backends.whoosh_backend.WhooshEngine',
        #索引文件路径
        'PATH': os.path.join(BASE_DIR, 'whoosh_index'),
    },
    # 使用Solr
    'default': {
        'ENGINE': 'haystack.backends.solr_backend.SolrEngine',
        'URL': 'http://127.0.0.1:8983/solr'
        # ...or for multicore...
        # 'URL': 'http://127.0.0.1:8983/solr/mysite',
    },
    # 使用Xapian
    'default': {
        'ENGINE': 'xapian_backend.XapianEngine',
        'PATH': os.path.join(os.path.dirname(__file__), 'xapian_index'),
    },
}

# 当添加、修改、删除数据时，自动生成索引
# 配置项保证了在Django运行起来后，有新的数据产生时，Haystack仍然可以让Elasticsearch实时生成新数据的索引
HAYSTACK_SIGNAL_PROCESSOR = 'haystack.signals.RealtimeSignalProcessor'

# 搜索结果页面分数的每页数量
HAYSTACK_SEARCH_RESULTS_PER_PAGE = 10
```

## 建立索引

- 创建索引类

通过创建索引类，来指明让搜索引擎对哪些字段建立索引，也就是可以通过哪些字段的关键字来检索数据。

```python
# 本项目中对SKU信息进行全文检索，所以在`goods`应用中新建`search_indexes.py`文件，用于存放索引类。
from haystack import indexes
from .models import SKU


class SKUIndex(indexes.SearchIndex, indexes.Indexable):
    """SKU索引数据模型类"""
    # 接收索引字段：使用文档定义索引字段，并使用模板语法渲染
    # document=True表明有相关文档说明。use_template=True表示使用模板语法说明字段。
    text = indexes.CharField(document=True, use_template=True)

    def get_model(self):
        """返回建立索引的模型类"""
        return SKU

    def index_queryset(self, using=None):
        """返回要建立索引的数据查询集"""
        return self.get_model().objects.filter(is_launched=True)
```

- 创建模版索引文件

在`templates`目录中创建`text字段`使用的模板文件

具体在`templates/search/indexes/goods/sku_text.txt`文件中定义

```python
{{ object.id }}
{{ object.name }}
{{ object.caption }}
```

模板文件说明：当将关键词通过text参数名传递时此模板指明SKU的`id`、`name`、`caption`作为`text`字段的索引值来进行关键字索引查询。

## 设置视图

- 添加`SearchView`到`URLconf`

```python
import haystack.urls

# 这会拉取Haystack的默认URLconf，它由单独指向SearchView实例的URLconf组成。你可以通过传递几个关键参数或者完全重新它来改变这个类的行为。
url(r'^search/', include('haystack.urls')),

# 视图返回数据
query：搜索关键字
paginator：分页paginator对象
page：当前页的page对象（遍历page中的对象，可以得到result对象）
result.objects: 当前遍历出来的SKU对象。
```

- 创建搜索模版

`templates/base.html`

```html
<div class="search con fl">
	<form method="get" action="/search/">
		<input type="text" class="input_text fl" name="q" placeholder="搜索商品">
		<input type="button" class="input_btn fr" value="搜索">
	</form>
</div>
```

`templates/search/indexes/search.html`

```html
<div class="main_wrap clearfix">
    <div class=" clearfix">
        <ul class="goods_type_list clearfix">
            {% for result in page %}
            <li>
                {# object取得才是sku对象 #}
                <a href="/detail/{{ result.object.id }}/"><img src="{{ result.object.default_image.url }}"></a>
                <h4><a href="/detail/{{ result.object.id }}/">{{ result.object.name }}</a></h4>
                <div class="operate">
                    <span class="price">￥{{ result.object.price }}</span>
                    <span>{{ result.object.comments }}评价</span>
                </div>
            </li>
            {% else %}
                <p>没有找到您要查询的商品。</p>
            {% endfor %}
        </ul>
        <div class="pagenation">
            <div id="pagination" class="page"></div>
        </div>
    </div>
</div>
```

分页

```html
<div class="main_wrap clearfix">
    <div class=" clearfix">
        ......
        <div class="pagenation">
            <div id="pagination" class="page"></div>
        </div>
    </div>
</div>


<script type="text/javascript">
    $(function () {
        $('#pagination').pagination({
            currentPage: {{ page.number }},
            totalPage: {{ paginator.num_pages }},
            callback:function (current) {
                {#window.location.href = '/search/?q=iphone&amp;page=1';#}
                window.location.href = '/search/?q={{ query }}&page=' + current;
            }
        })
    });
</script>
```

- 手动生成初始索引

把数据库中的数据放入索引了，Haystack附带的一个命令行管理工具使它变得很容易。

```shell
python manage.py rebuild_index
```

`elasticsearch`生成的索引位于搜索引擎所在服务器上，`whoosh`生成的索引位于项目所在服务器上。