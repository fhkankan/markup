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

## 保存图片

方法一

```javascript
// 工具栏
toolbox: 
{
    show: true,
    feature: {
        dataZoom: {
            yAxisIndex: 'none'
        }, //区域缩放，区域缩放还原
        dataView: {
            readOnly: false
        }, //数据视图
        magicType: {
            type: ['line', 'bar']
        },  //切换为折线图，切换为柱状图
        restore: {},  //还原
        saveAsImage: {}   //保存为图片
    }
},
```
方法二
```javascript
// 自定义方法
exportpic(val)
{
    let myChart = this.$echarts.init(document.getElementById(val));
    let picInfo=myChart.getDataURL({
        type: 'png',
        pixelRatio: 1.5,  //放大两倍下载，之后压缩到同等大小展示。解决生成图片在移动端模糊问题
        backgroundColor: '#fff'
    });//获取到的是一串base64信息
 
    const elink = document.createElement('a');
    elink.download = '统计图';
    elink.style.display = 'none';
    elink.href = picInfo;
    document.body.appendChild(elink);
    elink.click();
    URL.revokeObjectURL(elink.href); // 释放URL 对象
    document.body.removeChild(elink)
},
```

方法三

```javascript
// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('dailyCurveCharts'));
// 获取echarts中的canvas
var mycanvas = $("#dailyCurveCharts").find("canvas")[0];
// 点击按钮到处图片
$("#exportExcel").click(function () {
    var mycanvas = $("#dailyCurveCharts").find("canvas")[0];
 
    var image = mycanvas.toDataURL("image/jpeg");
 
    var $a = document.createElement('a');
    $a.setAttribute("href", image);
    $a.setAttribute("download", "");
    $a.click();
 
//    window.location.href=image; // it will save locally
});
// 设置到处图片的名称
$a.setAttribute("download", "echarts图片下载");
```

