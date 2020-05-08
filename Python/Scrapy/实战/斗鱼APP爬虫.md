## 手机App抓包爬虫

### items.py

```
class DouyuspiderItem(scrapy.Item):
    name = scrapy.Field()# 存储照片的名字
    imagesUrls = scrapy.Field()# 照片的url路径
    imagesPath = scrapy.Field()# 照片保存在本地的路径
```

### spiders/douyu.py

```
import scrapy
import json
from douyuSpider.items import DouyuspiderItem

class DouyuSpider(scrapy.Spider):
    name = "douyu"
    allowd_domains = ["http://capi.douyucdn.cn"]

    offset = 0
    url = "http://capi.douyucdn.cn/api/v1/getVerticalRoom?limit=20&offset="
    start_urls = [url + str(offset)]

  def parse(self, response):
      # 返回从json里获取 data段数据集合
      data = json.loads(response.text)["data"]

      for each in data:
          item = DouyuspiderItem()
          item["name"] = each["nickname"]
          item["imagesUrls"] = each["vertical_src"]

          yield item

      self.offset += 20
      yield scrapy.Request(self.url + str(self.offset), callback = self.parse)
```

###setting.py

```
ITEM_PIPELINES = {'douyuSpider.pipelines.ImagesPipeline': 1}

# Images 的存放位置，之后会在pipelines.py里调用
IMAGES_STORE = "/Users/Power/lesson_python/douyuSpider/Images"

# user-agent
USER_AGENT = 'DYZB/2.290 (iPhone; iOS 9.3.4; Scale/2.00)'
```

### pipelines.py

```
import scrapy
import os
from scrapy.pipelines.images import ImagesPipeline
from scrapy.utils.project import get_project_settings

class ImagesPipeline(ImagesPipeline):
    IMAGES_STORE = get_project_settings().get("IMAGES_STORE")

    def get_media_requests(self, item, info):
        image_url = item["imagesUrls"]
        yield scrapy.Request(image_url)

    def item_completed(self, results, item, info):
        # 固定写法，获取图片路径，同时判断这个路径是否正确，如果正确，就放到 image_path里，ImagesPipeline源码剖析可见
        image_path = [x["path"] for ok, x in results if ok]

        os.rename(self.IMAGES_STORE + "/" + image_path[0], self.IMAGES_STORE + "/" + item["name"] + ".jpg")
        item["imagesPath"] = self.IMAGES_STORE + "/" + item["name"]

        return item

#get_media_requests的作用就是为每一个图片链接生成一个Request对象，这个方法的输出将作为item_completed的输入中的results，results是一个元组，每个元组包括(success, imageinfoorfailure)。如果success=true，imageinfoor_failure是一个字典，包括url/path/checksum三个key。
```

###main.py
在项目根目录下新建main.py文件,用于调试
```
from scrapy import cmdline
cmdline.execute('scrapy crawl douyu'.split())
```

###执行程序

```
py2 main.py
```