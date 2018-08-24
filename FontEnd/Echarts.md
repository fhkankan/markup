# ECharts

ECharts是基于HTML5中Canvas的一款JS图形可视化工具。

官网:http://echarts.baidu.com/index.html

## 准备工作

引入

```html
// 方法一
<script src="echarts.min.js"></script>

// 方法二
<script src="http://echarts.baidu.com/dist/echarts.min.js"></script>
```

准备画板

```html
<div id="main" style="width:600px;height:400px;"></div>
```

## 绘制图形

官网实例：

初始化

```javascript
// 新建一个script标签并开始编写JS代码，使用如下代码获取画板，并进行初始化工作
var myChart = echarts.init(document.getElementById("main"[,配色主题]))
```

图形配置

```javascript
// 使用一个JS对象指定图形的全部配置内容，使用键值对设置配置项，包括图形的标题、图例、类型、数据、坐标轴等
var option = {
    // 标题
    title:{
        text:"Echarts 入门示例"
    },
    // 提示框
    tooltip:{},
    // 图例
    legend:{
        data:["销量"]
    },
    // x轴
    xAxis:{
        data:["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]
    },
    // y轴
    yAxis:{},
    // 数据
    series:[{
        name:'销量',
        type:'bar',
        data:[5,20,36,10,,10,20]
    }]
}
```

绑定myChart

```javascript
myChart.setOption(option);
```

## 常用配置项

官网:http://echarts.baidu.com/option.html

| 配置项 | 说明           |
| ------ | -------------- |
| title  | 图形的标题     |
| legend | 图形的图例     |
| grid   | 图形的绘图范围 |
|        |                |
|        |                |
|        |                |
|        |                |
|        |                |
|        |                |

