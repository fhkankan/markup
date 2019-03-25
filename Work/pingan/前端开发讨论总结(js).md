# js开发经验

1. 框架选型

```
js
vue
```

2. 代码封装

```
页面一类
逻辑一类
机器人交互一类
```

3. 多人开发

```
封装好进入/退出功能，方便别人调用
```

4. 常用文件夹

```
config	机器人信息、常变更部分做配置文件
common	公共函数
third	第三方模块
sdk		sdk
logic	主逻辑
```

5. 模块间处理

```
出功能模块前，清除此功能模块的所有功能
```

6. 交互

```
json
注意：数字、字符串
```

7. 测试

```
测试样例
```

8. 开发规范

```
git提交
注释
代码行数
变量命名、函数命名
```

# 开发模块

## 常用逻辑

### 定时器

```javascript
/**
 * 公用js eg：管理定时器函数
 * @param {int} waitTime 设定的定时时间
 * @param {string} task 设定的定时执行任务
 */
 function Timer(waitTime,task){
    this.WAITTIME = waitTime; //定时时间
    this.task = task; //定时要执行的任务
    this.TIMEROBJ = new Object(); //时间管理器句柄

    /**
     * 启动定时器
     */
    this.startTimer = function(){
        this.TIMEROBJ = setTimeout(this.task,this.WAITTIME);
        // console.log("开启定时器")
    }

    /**
     * 关闭定时器
     */
    this.shutdownTimer = function() {
        clearTimeout(this.TIMEROBJ);
        // console.log("关闭定时器")
    }

    /**
     * 重置定时器
     */
    this.resetTimer = function() {
        this.shutdownTimer();
        this.startTimer();
        // console.log("重置定时器")
    }
}   
```

### 比较时间

```javascript
/**
 * 计算两个时间戳的差值
 *
 * @param {*} oldTime 旧时间
 * @param {*} newTime 新时间，若缺省则默认为当前时间
 * @returns 两个时间的差值(s)
 */
function diffTime_s(oldTime, newTime){
    if (oldTime === undefined){
        return 99999
    }
    newStamp = (newTime === undefined? CTStamp_s():newTime);
    timeDiff = (newStamp - oldTime);
    return timeDiff
}
/**
 *获取当前时间的时间戳(单位为s)
 *
 * @returns 当前时间戳(s)
 */
function CTStamp_s(){
    return (new Date()).getTime()/1000
}
```

### 请求跨域

```javascript
// 使用sdk请求，使用andriod转发
 RobotSDK.httpRequest({
    	"type": "post",
    	"url":HttpServer.healthCardExist,
    	"data":{
    	    "secondToken":HttpServer.secondToken,
    	    "regionCode": HttpServer.cityCode,
    	},
    	"header":{
    	    "Authorization":"Bearer "+ HttpServer.accessToken,
    	},
    	"dataType":"application/json", 
    	"successCallback": function(data){}, 
    	'failureCallback': function(data){
    	    console.log('failureCallback');
    	}
});
```

### 状态配置

```javascript
RobotStatus = {
    "openApp":{
        "voice":["0"],
        "motion":["1开机完成"],
        "emotion":"blink",
        "led":"" 
    },
    "faceDetect":{
        "voice":["1"],
        "motion":["9人脸识别b","3人脸识别a","18","2"],
        "emotion":"smile01",
        "led":"" 
    },
    doRandomMotion:function(scene){
        motionList = RobotStatus[scene]["motion"];
        length = motionList.length;
        var index = Math.floor(Math.random()*length);
        RobotSDK.runMotionGroup({"groupName": motionList[index]});
    },
    playRandomVoice:function(scene){
        voiceList = RobotStatus[scene]["voice"];
        length = voiceList.length;
        var index = Math.floor(Math.random()*length);
        RobotSDK.playVoice({"text": voiceList[index]});
    },
    setLedColors:function(scene){
        ledParams = RobotStatus[scene]["led"]
        RobotSDK.setLedColors(ledParams);
        // console.log("setLedColors: " + JSON.stringify(ledParams))
    },
    setFaceEmotion:function(scene){
        emotion = RobotStatus[scene]["emotion"];
        RobotSDK.setFaceEmotion({"emotionName":emotion});
        // console.log("setFaceEmotion: " + emotion);
    },
    changeStatus:function(scene){
        RobotStatus.doRandomMotion(scene);
        RobotStatus.playRandomVoice(scene);
        RobotStatus.setFaceEmotion(scene);
        RobotStatus.setLedColors(scene);

    },
    changeStatusLess:function(scene){
        RobotStatus.doRandomMotion(scene);
        RobotStatus.setFaceEmotion(scene);
        RobotStatus.setLedColors(scene);
    }
}
```

### 页面逻辑

```javascript
var home = {
    // 页面变量
    typeKeyWordsObj: new Object(),
    robotFaceObj: new Object(),
    robotASRObj: new Object(),
    faceDetectTime:{"recognise":{},"unrecognise":0},
    faceVoiceInterval:300,
    // 进入主页
    setHomeDisplay: function(){
        home.initHomeDisplay();
        home.addEventFromRobot()
        console.log("进入主页")
    },
    // 主页初始化
    initHomeDisplay: function(){}
    // 机器人事件
    addEventFromRobot: function(){}
	// 主页消失
    setHomeNone: function(){}
	// 关闭应用
	closeApp:function(){}
}
```

### 人脸检测

```javascript
var home = {
    robotFaceObj: new Object(),
    // 状态变量
    faceDetectTime:{"recognise":{},"unrecognise":0},
    faceVoiceInterval:300,
    addEventFromRobot: function(){
    	//订阅人脸事件
        home.robotFaceObj = RobotSDK.addRobotEvent({
            'eventName': 'EVENT_RECEIVE_FACE_RECOGNITION_RESULT',
            'description': '人脸检测事件',
            'isBlock': 'false',// true则拦截人脸跟踪
            'successCallback': function(data){
                retStr = JSON.stringify(data)
                console.log("faceRecognise Result: " + retStr)
                if(retStr.indexOf("name")>-1){
                    name = data['msg']['eventDetail']['result']['name'];
                    var nameStamp = home.faceDetectTime["recognise"][name]
                    if (diffTime_s(nameStamp) > home.faceVoiceInterval){
                        RobotStatus.changeStatus("faceDetect")
                    }
              		home.faceDetectTime["recognise"][name] = CTStamp_s()
                }
                else{
                    var nameStamp = home.faceDetectTime["unrecognise"]
                    if (diffTime_s(nameStamp) > home.faceVoiceInterval){
                        RobotStatus.changeStatus("faceDetect")
                    }
                    home.faceDetectTime["unrecognise"] = CTStamp_s()
                    
                }
                home.enterBusyStatus(); 
                console.log('successCallback');
            },
            'failureCallback': function(data){
                console.log('failureCallback')
            }
        });
        console.log("initHomeDisplay robotFaceObj");
}			
```

### 常规函数回调

```javascript
RobotSDK.getStarFace({
       'params':'',
       'successCallback': function(data){
           console.log(data.toString());
       },
       'failureCallback': function(data){
           console.log(data.toString());
        }
});
```

## 事件订阅

### 语音对话事件

- 事件信息

eventName | description | details（content）| 
---| ---| ---| ---| ---| 
```EVENT_ROBOT_START_SPEAKING``` | 开始说话事件 | 说话的内容 | 播放的内容是tts还是音频文件
```EVENT_ROBOT_STOP_SPEAKING``` | 停止说话事件 | 说话的内容 | 播放的内容是tts还是音频文件
```EVENT_ROBOT_START_LISTENING``` | 机器人开始监听事件 | 
```EVENT_ROBOT_STOP_LISTENING``` | 机器人停止监听事件 | 
```EVENT_ASR_ERROR``` | 机器人监听异常事件 | 错误内容
```EVENT_RECEIVE_NLP_REQUEST``` | 语音识别结果事件
```EVENT_RECEIVE_NLPREPLY``` | 语义回复事件
```EVENT_ROBOT_START_THINKING``` | 机器人开始思考事件

- js调用

订阅ASR识别结果

```javascript
Home.robotASRObj = RobotSDK.addRobotEvent({
        'eventName':'EVENT_RECEIVE_NLP_REQUEST',//订阅接收ASR识别结果
        'description': 'NLP事件 Autonomous',
        'isBlock':'true',
        'successCallback':function(data){},
  		'failureCallback':function (data) {}
});

sendRobotEvent  方行继续执行


// 参数data
/*
* data = {
*      "status":"200",
*      "msg":{
*            xxx
*          }
*    	}
*/
```

订阅机器人语音回复

```javascript
Home.robotReplyObj = RobotSDK.addRobotEvent({
        'eventName': 'EVENT_RECEIVE_NLPREPLY', //NLP回复话术
        'description': 'NLP回复事件',
        'isBlock':'true',//拦截服务器的语义回复
        'successCallback':function (data) {},
        'failureCallback':function (data) {}
});
```

订阅机器人说话

```javascript
RobotSDK.playVoice({
    "text":ttsTxt,
    "successCallback":function (data) {}
});
```

使用关键字

```javascript
Home.typeKeyWordsObj = RobotSDK.addUserKeyWords({
        'keyword': '爵士音乐 流行音乐 古典音乐',
        'successCallback': function(){
            console.log('type successCallback');
        },
        'failureCallback': function(){
            var ttsText = "很抱歉，请您再重复一次您选择的音乐类型，或点击选择"
            RobotSDK.playVoice({"text": ttsText})
            console.log('type failureCallback');
        }
}); 
        
Home.robotASRObj = RobotSDK.addRobotEvent({
        'eventName': 'EVENT_RECEIVE_NLP_REQUEST',
        'description': '订阅ASR开始事件',
        'isBlock': 'true', //true,拦截机器人回复,false,机器人可以自由回复
        'successCallback': function(data){
            var retStr = JSON.stringify(data);
            ret = JSON.parse(retStr);
            var asrResult = ret.msg.eventDetail.question;
            RobotSDK.IsContainKeywords(asrResult);
            console.log('asr successCallback');
            },
        'failureCallback': function(data){
            console.log('asr failureCallback')
            }
});       
```

### 人脸检测事件

- 事件信息

eventName | description | details（content）| 
---| ---| ---| ---|
`EVENT_RECEIVE_FACE_RECOGNITION_RESULT` | 人脸检测 | json |
数据格式
```
{
	"eventName":"EVENT_RECEIVE_FACE_RECOGNITION_RESULT",
	"eventDescription":""
	"eventDetails":{
	    "startX":int,
		"startY":int,
		"endX":int,
		"endY":int,
		"centerX":int,
		"centerY":int,
		"image":base64,
		"result":{"name":"","confidence":int,"gender":"","age":""}
	}
}
```
参数

details字段 | 说明 | 值
---|--- | ---
startX | 人脸框起始点x坐标 | int
startY | 人脸框起始点y坐标 | int
endX | 人脸框对角点x坐标 | int
endY | 人脸框对角点y坐标 | int
centerX | 人脸中心x坐标 | int
centerY | 人脸中心y坐标 | int
image | 人脸图片 | base64格式
result | 人脸识别数据 | json格式

result字段 | 说明 | 值
---|--- | ---
name | 姓名 | 
confidence | 置信度
gender | 性别
age | 年龄 

- js调用

```javascript
//开启人脸检测 订阅人脸检测
Home.robotFaceObj = RobotSDK.addRobotEvent({
        'eventName': 'EVENT_RECEIVE_FACE_RECOGNITION_RESULT',
        'description': '人脸检测事件 startGreetDetect',
        'isBlock': 'false',
    	'successCallback': function(data){
        	console.log('faceSuccessCallback')
    	},
        'failureCallback': function(data){
            console.log('faceFailureCallback');
        }
    });
```

## 事件清除

- 全部清空

```javascript
RobotSDK.clearRobotEvent();
```

- 清除特定事件

```javascript
RobotSDK.removeRobotEvent(robotEventObject);
```

- 清除关键词

```javascript
RobotSDK.removeUserKeywords(keywordObject);
```

## 测试用例

ASR识别结果

```
RobotSDK.invokeJsMethod('{"function":"addRobotEvent", "params":{"eventName":"EVENT_RECEIVE_NLP_REQUEST","eventDetail":{"question":"播放红绿灯视频"}},"callbackId":"1440954275"}');

RobotSDK.invokeJsMethod('{"function":"addRobotEvent", "params":{"eventName":"EVENT_RECEIVE_NLP_REQUEST","eventDetail":{"question":"小天"}},"callbackId":"1440954275"}');
```

语音结束

```
RobotSDK.invokeJsMethod('{"function":"playVoice", "params":{},"callbackId":"7018941681"}');
```

人脸识别

```
# VIP 男
RobotSDK.invokeJsMethod('{"function":"addRobotEvent", "params":{"eventName":"EVENT_RECEIVE_FACE_RECOGNITION_RESULT","eventDetail":{"result":{"name":"Harry","confidence":82,"gender":"男","age":"28","vip":"1","vipConfidence":0.9}}},"callbackId":"7018941681"}')

# 非VIP 男
RobotSDK.invokeJsMethod('{"function":"addRobotEvent", "params":{"eventName":"EVENT_RECEIVE_FACE_RECOGNITION_RESULT","eventDetail":{"result":{"gender":"男"}}},"callbackId":"7471267009"}')
```

明星脸

```
RobotSDK.invokeJsMethod('{"function":"getStarFace","callbackId":"8734797621","params":{"normalPersonPath":"liushishi.jpeg","starPersonPath":"liushishi.jpeg","similarity":"88","starName":"周杰伦"}}');
```

播放视频

```
RobotSDK.invokeJsMethod('{"function":"playVideo", "params":{"viewId":"1","result":"complete"},"callbackId":"0012576846"}')
```

音乐生成

```
# 生成成功
RobotSDK.invokeJsMethod('{"function":"requestForMusic", "params":{"result":"success","error":"xxxx"},"callbackId":"0012576846"}')

# 生成失败
RobotSDK.invokeJsMethod('{"function":"requestForMusic", "params":{"result":"fail","error" :"xxxx"},"callbackId":"0012576846"}')
```

ajax

```
RobotSDK.invokeJsMethod('{"function":"httprequest", "params":{"key":"1"},"callbackId":"0012576846"}')
```

