# selenium

## 入门

> 启动

```python
from selenium import webdriver

driver=webdriver.Chrome()
driver=webdirver.Firefox()
driver=webdriver.Ie()
```

> 关闭

```
// 关闭当前窗口
driver.close()
// 关闭浏览器
driver.quit()
```

> 常用api

```
driver.current_url
driver.title
driver.page_source  # 源码
driver.name
driver.current_window_handle  # 当前窗口的句柄
```

> 页面处理

```
from selenium import webdriver  
  
driver = webdriver.Firefox()
driver.maximize_window()	# 页面最大化
driver.get('https://www.baidu.com')
driver.minimize_window()	# 页面最小化
```

> 跳转

```
driver.get("http://www.baidu.com")
driver.forward()
driver.back()
driver.refresh()

直到页面加载完全继续执行后面的程序，不等待ajax操作
配合time.sleep()方法使用
```

eg

```python
# 打开3个窗口，通过switch_to_window切换关闭窗口
from selenium import webdriver
driver=webdriver.Chrome()
driver.get("http://www.baidu.com/")
print driver.window_handles
driver.switch_to_window(driver.window_handles[1])
driver.close()

# 打开百度首页，输入搜索内容并进行光标移动后回车
from selenium import webdriver
driver=webdriver.Chrome()
driver.maximize.window() #窗口最大化
driver.get("http://www.baidu.com/")
element=driver.find_element_by_id('kw')
element.clear()
element.send_keys('抗战胜利70周年'.decode('gbk'))
element.send_keys(Keys.ARROW_DOWN) #光标向下
element.send_keys(Keys.ENTER) #回车
```

> 常用元素操作

```
element.get_attribute('class')
element.is_displayed()
element.is_enabled()	# 是否可点击可输入等
element.is_selected()
element.location	# 坐标
element.parent	# 上一级
element.size	# 长宽
element.tag_name	# 返回标签名
element.text	# 返回文本
```

> 页面交互select操作

```
from selenium.webdriver.support.ui import Select

select=Select(driver.find_element_by_name('name'))
select.select_by_index(index)
select.select_by_visible_text("text")
select.select_by_value(value)
select.deselect_all()
select.options
select.all_selected_options
```

eg

[![wKiom1XpytLCm9m9AACOqe5mOzo697.jpg](http://s3.51cto.com/wyfs02/M01/72/A5/wKiom1XpytLCm9m9AACOqe5mOzo697.jpg)](http://s3.51cto.com/wyfs02/M01/72/A5/wKiom1XpytLCm9m9AACOqe5mOzo697.jpg)

如上代码处理下拉框元素

```
// 方法一
from selenium import webdriver
driver=webdriver.Chrome()
driver.get('http://localhost/test.html')
element=driver.find_element_by_id('lang')
options=element.find_elements_by_tag_name('option')
for i in options:
print i.get_attribute('value')
for i in options:
print i.get_attribute('text')

// 方法二
from selenium import webdriver
from selenium.webdriver.support.ui import Select
driver=webdriver.Chrome()
Select(driver.find_element_by_id('lang')).select_by_visible_text('简体'.decode('gbk'))
```

> 页面交互keys操作

```
from selenium.webdriver.common.keys import Keys
ALT
ARROW_DOWN /LEFT/RIGHT/UP
BACKSPACE
CONTROL
ENTER
ESCAPE
F1 /2/3/4/5...
SHIFT
APACE
TAB
```

> 页面交互wait操作--implicit wait

```
# 进行find操作时，等待固定秒数，成功退出计时
driver.implicitly_wait(10)
```

> 页面交互wait操作--explicit wait

```
//按照一定条件执行wait操作
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
element=WebDriverWait(driver,1).until(expected_conditions.presence_of_elemrnt_located(By.ID,"idx"))
//条件列表如下：
title_is
title_contains
presence_of_element_located
visibility_of_element_located
visibility_of
presence_of_all_elements_located
text_to_be_present_in_element
text_to_be_present_in_element_value
frame_to_be_available_and_switch_to_it
invisibility_of_element_located
element_to_be_clickable
staleness_of
element_to_be_selected
element_located_to_be_selected
element_selection_state_to_be
element_located_selection_state_to_be
alert_is_present
```

> cookie操作

```
driver.get_cookies()
cookie={'name':'zz','value':18}
driver.add_cookie(cookie)
driver.get_cookie('zz')
driver.delete_cookie('zz')
```

> js操作

```
js='alert("hello")'
driver.execute_script(js)
js='console.log("hello")'
driver.excute_script(js)
```

## 元素定位方法

> 定位单元素

```
find_element_by_id
find_element_by_name
find_element_by_xpath
find_element_by_link_text
find_element_by_partial_link_text
find_element_by_tag_name
find_element_by_class_name
find_element_by_css_selecror
```

> 定位多元素

```
find_elements_by_name
find_elements_by_xpath
find_elements_by_link_text
find_elements_by_partial_link_text
find_elements_by_tag_name
find_elements_by_class_name
find_elements_by_css_selecror
```

> 通用方法

```
from selenium.webdriver.common.by import By
driver.find_element(By.XPATH,'//button[text()="some text"]')
driver.find_elements(By.XPATH,'//button')
```