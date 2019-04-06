# Vue高阶

## vue-resource

Vue.js是数据驱动的，这使得我们并不需要直接操作DOM，如果我们不需要使用jQuery的DOM选择器，就没有必要引入jQuery。vue-resource是Vue.js的一款插件，它可以通过XMLHttpRequest或JSONP发起请求并处理响应。也就是说，$.ajax能做的事情，vue-resource插件一样也能做到，而且vue-resource的API更为简洁。另外，vue-resource还提供了非常有用的inteceptor功能，使用inteceptor可以在请求前和请求后附加一些行为，比如使用inteceptor在ajax请求时显示loading界面。

> 优点

```
1.体积小
vue-resource非常小巧，在压缩以后只有大约12KB，服务端启用gzip压缩后只有4.5KB大小，这远比jQuery的体积要小得多。

2.支持主流的浏览器
和Vue.js一样，vue-resource除了不支持IE 9以下的浏览器，其他主流的浏览器都支持。

3.支持Promise API和URI Templates
Promise是ES6的特性，Promise的中文含义为“先知”，Promise对象用于异步计算。
URI Templates表示URI模板，有些类似于ASP.NET MVC的路由模板。

4.支持拦截器
拦截器是全局的，拦截器可以在请求发送前和发送请求后做一些处理。
拦截器在一些场景下会非常有用，比如请求发送前在headers中设置access_token，或者在请求失败时，提供共通的处理方式
```

### 引用

```html
<script src="js/vue.js"></script>
<script src="js/vue-resource.js"></script>
```

### 使用

> Get/Post请求

```js
// 基于全局Vue对象使用http
Vue.http.get('/someUrl', [options]).then(successCallback, errorCallback);
Vue.http.post('/someUrl', [body], [options]).then(successCallback, errorCallback);

// 在一个Vue实例内使用$http
this.$http.get('/someUrl', [options]).then(successCallback, errorCallback);
this.$http.post('/someUrl', [body], [options]).then(successCallback, errorCallback);
```

> Then响应

在发送请求后，使用then方法来处理响应结果，then方法有两个参数，第一个参数是响应成功时的回调函数，第二个参数是响应失败时的回调函数。
then方法的回调函数也有两种写法，第一种是传统的函数写法，第二种是更为简洁的ES 6的Lambda写法：

```js
// 传统写法
this.$http.get('/someUrl', [options]).then(function(response){
  // 响应成功回调
}, function(response){
  // 响应错误回调
});

// Lambda写法
this.$http.get('/someUrl', [options]).then((response) => {
  // 响应成功回调
}, (response) => {
  // 响应错误回调
});

```

### API

vue-resource的请求API是按照REST风格设计的，它提供了7种请求API

```
get(url, [options])
head(url, [options])
delete(url, [options])
jsonp(url, [options])
post(url, [body], [options])
put(url, [body], [options])
patch(url, [body], [options])
```

除了jsonp以外，另外6种的API名称是标准的HTTP方法。当服务端使用REST API时，客户端的编码风格和服务端的编码风格近乎一致，这可以减少前端和后端开发人员的沟通成本。

| 客户端请求方法         | 服务端处理方法 |
| ---------------------- | -------------- |
| this.$http.get(...)    | Getxxx         |
| this.$http.post(...)   | Postxxx        |
| this.$http.put(...)    | Putxxx         |
| this.$http.delete(...) | Deletexxx      |

> options对象

| 参数        | 类型                    | 描述                                                         |
| ----------- | ----------------------- | ------------------------------------------------------------ |
| url         | string                  | 请求的URL                                                    |
| method      | string                  | 请求的HTTP方法，例如：'GET', 'POST'或其他HTTP方法            |
| body        | Object, FormData string | request body                                                 |
| params      | Object                  | 请求的URL参数对象                                            |
| headers     | Object                  | request header                                               |
| timeout     | number                  | 单位为毫秒的请求超时时间 (0 表示无超时时间)                  |
| before      | function(request)       | 请求发送前的处理函数，类似于jQuery的beforeSend函数           |
| progress    | function(event)         | ProgressEvent回调处理函数                                    |
| credentials | boolean                 | 表示跨域请求时是否需要使用凭证                               |
| emulateHTTP | boolean                 | 发送PUT, PATCH, DELETE请求时以HTTP POST的方式发送，并设置请求头的X-HTTP-Method-Override |
| emulateJSON | boolean                 | 将request body以application/x-www-form-urlencoded content type发送 |

emulateHTTP

```
如果Web服务器无法处理PUT, PATCH和DELETE这种REST风格的请求，你可以启用enulateHTTP现象。
启用该选项后，请求会以普通的POST方法发出，并且HTTP头信息的X-HTTP-Method-Override属性会设置为实际的HTTP方法。
Vue.http.options.emulateHTTP = true;
```

emulateJSON

```
如果Web服务器无法处理编码为application/json的请求，你可以启用emulateJSON选项。
启用该选项后，请求会以application/x-www-form-urlencoded作为MIME type，就像普通的HTML表单一样。
Vue.http.options.emulateJSON = true;

```

> response对象

response对象包含以下方法、属性

| 方法       | 类型    | 描述                                          |
| ---------- | ------- | --------------------------------------------- |
| text()     | string  | 以string形式返回response body                 |
| json()     | Object  | 以JSON对象形式返回response body               |
| blob()     | Blob    | 以二进制形式返回response body                 |
| 属性       | 类型    | 描述                                          |
| ok         | boolean | 响应的HTTP状态码在200~299之间时，该属性为true |
| status     | number  | 响应的HTTP状态码                              |
| statusText | string  | 响应的状态文本                                |
| headers    | Object  | 响应头                                        |

### 实例

> get

这段程序的then方法只提供了successCallback，而省略了errorCallback。
catch方法用于捕捉程序的异常，catch方法和errorCallback是不同的，errorCallback只在响应失败时调用，而catch则是在整个请求到响应过程中，只要程序出错了就会被调用。

```js
var demo = new Vue({
  el: '#app',
  data: {
    gridColumns: ['customerId', 'companyName', 'contactName', 'phone'],
    gridData: [],
    apiUrl: 'http://211.149.193.19:8080/api/customers'
  },
  ready: function() {
    this.getCustomers()
  },
  methods: {
    getCustomers: function() {
      this.$http.get(this.apiUrl)
        .then((response) => {
          this.$set('gridData', response.data)
        })
        .catch(function(response) {
          console.log(response)
        })
    }
  }
})
```

在then方法的回调函数内，你也可以直接使用this，this仍然是指向Vue实例的：

```js
getCustomers: function() {
  this.$http.get(this.apiUrl)
    .then((response) => {
      this.$set('gridData', response.data)
    })
    .catch(function(response) {
      console.log(response)
    })
}
```

> jsonp

```js
getCustomers: function() {
  this.$http.jsonp(this.apiUrl).then(function(response){
    this.$set('gridData', response.data)
  })
}
```

> post

```js
var demo = new Vue({
  el: '#app',
  data: {
    show: false,
    gridColumns: [{
      name: 'customerId',
      isKey: true
    }, {
      name: 'companyName'
    }, {
      name: 'contactName'
    }, {
      name: 'phone'
    }],
    gridData: [],
    apiUrl: 'http://211.149.193.19:8080/api/customers',
    item: {}
  },
  ready: function() {
    this.getCustomers()
  },
  methods: {
    closeDialog: function() {
      this.show = false
    },
    getCustomers: function() {
      var vm = this
      vm.$http.get(vm.apiUrl)
        .then((response) => {
          vm.$set('gridData', response.data)
        })
    },
    createCustomer: function() {
      var vm = this
      vm.$http.post(vm.apiUrl, vm.item)
        .then((response) => {
          vm.$set('item', {})
          vm.getCustomers()
        })
      this.show = false
    }
  }
})
```

> put

```js
updateCustomer: function() {
  var vm = this
  vm.$http.put(this.apiUrl + '/' + vm.item.customerId, vm.item)
    .then((response) => {
      vm.getCustomers()
    })
}
```

> delete

```js
deleteCustomer: function(customer){
  var vm = this
  vm.$http.delete(this.apiUrl + '/' + customer.customerId)
    .then((response) => {
      vm.getCustomers()
    })
}
```

vue-resource是一个非常轻量的用于处理HTTP请求的插件，它提供了两种方式来处理HTTP请求：
1、使用Vue.http或this.$http
2、使用Vue.resource或this.$resource

```js
data(){
    return{
      toplist:[],
      alllist:[]
    }
  },
  //vue-router
  route:{
    data({to}){
      //并发请求，利用 Promise 
      return Promise.all([
        //简写
        this.$http.get('http://192.168.30.235:9999/rest/knowledge/list',{'websiteId':2,'pageSize':5,'pageNo':1,'isTop':1}),
        //this.$http.get('http://192.168.30.235:9999/rest/knowledge/list',{'websiteId':2,'pageSize':20,'pageNo':1,'isTop':0})
        //不简写
        this.$http({
          method:'GET',
          url:'http://192.168.30.235:9999/rest/knowledge/list',
          data:{'websiteId':2,'pageSize':20,'pageNo':1,'isTop':0},
          headers: {"X-Requested-With": "XMLHttpRequest"},
          emulateJSON: true
          })
        ]).then(function(data){//es5写法
           return{
            toplist:data[0].data.knowledgeList,
            alllist:data[1].data.knowledgeList
          }
        //es6写法 .then()部分
        //.then(([toplist,alllist])=>({toplist,alllist})) 
      },function(error){
        //error
      })
    }
  }
```

## vue axios

vue2.0之后，就不再对vue-resource更新，而是推荐使用axios。基于 Promise 的 HTTP 请求客户端，可同时在浏览器和 Node.js 中使用。

> 特点

```
1、在浏览器中发送 XMLHttpRequests 请求
2、在 node.js 中发送 http请求
3、支持 Promise API
4、拦截请求和响应
5、转换请求和响应数据
6、取消请求
7、自动转换 JSON 数据
8、客户端支持保护安全免受 CSRF/XSRF 攻击
```

### 安装

> npm

```
// 安装
$ npm install axios
// 在要使用的文件中引入axios
import axios from 'axios'
```

> 单文件

```html
<script src="/static/js/axios.js"></script>
```

### 使用

#### get

```js
// 向具有指定ID的用户发出请求
axios.get('/user?ID=12345')
.then(function (response) {
	console.log(response);
})
.catch(function (error) {
	console.log(error);
});
 
// 也可以通过 params 对象传递参数
axios.get('/user', {
	params: {
		ID: 12345
	}
})
.then(function (response) {
	console.log(response);
})
.catch(function (error) {
	console.log(error);
});
```

#### post

```js
axios.post('/user', {
	firstName: 'Fred',
	lastName: 'Flintstone'
})
.then(function (response) {
	console.log(response);
})
.catch(function (error) {
	console.log(error);
});
```

#### 多个并发

```js
function getUserAccount() {
	return axios.get('/user/12345');
}
 
function getUserPermissions() {
	return axios.get('/user/12345/permissions');
}
 
axios.all([getUserAccount(), getUserPermissions()])
.then(axios.spread(function (acct, perms) {
	//两个请求现已完成
}));
```

### API

#### 常规使用

```js
axios(config)
// 发送一个 POST 请求
axios({
method: 'post',
url: '/user/12345',
data: {
firstName: 'Fred',
lastName: 'Flintstone'
}
});

axios(url[, config])
// 发送一个 GET 请求 (GET请求是默认请求模式)
axios('/user/12345');
```

#### 请求方法别名

```js
//为了方便起见，已经为所有支持的请求方法提供了别名。
axios.request（config）
axios.get（url [，config]）
axios.delete（url [，config]）
axios.head（url [，config]）
axios.post（url [，data [，config]]）
axios.put（url [，data [，config]]）
axios.patch（url [，data [，config]]）
//注意:当使用别名方法时，不需要在config中指定url，method和data属性。
```

#### 并发

帮助函数处理并发请求。

```
axios.all（iterable）
axios.spread（callback）
```

#### 实例

> 创建实例

也可以使用自定义配置创建axios的新实例。
`axios.create（[config]）`

```
var instance = axios.create({
baseURL: 'https://some-domain.com/api/',
timeout: 1000,
headers: {'X-Custom-Header': 'foobar'}
});
```

> 实例方法

可用的实例方法如下所示。 指定的配置将与实例配置合并

```
axios＃request（config）
axios＃get（url [，config]）
axios＃delete（url [，config]）
axios＃head（url [，config]）
axios＃post（url [，data [，config]]）
axios＃put（url [，data [，config]]）
axios＃patch（url [，data [，config]]）
```

#### 请求配置

这些是用于发出请求的可用配置选项。 只有url是必需的。 如果未指定方法，请求将默认为GET

```js
{
// `url`是将用于请求的服务器URL
url: '/user',
 
// `method`是发出请求时使用的请求方法
method: 'get', // 默认
 
// `baseURL`将被添加到`url`前面，除非`url`是绝对的。
// 可以方便地为 axios 的实例设置`baseURL`，以便将相对 URL 传递给该实例的方法。
baseURL: 'https://some-domain.com/api/',
 
// `transformRequest`允许在请求数据发送到服务器之前对其进行更改
// 这只适用于请求方法'PUT'，'POST'和'PATCH'
// 数组中的最后一个函数必须返回一个字符串，一个 ArrayBuffer或一个 Stream
 
transformRequest: [function (data) {
// 做任何你想要的数据转换
 
return data;
}],
 
// `transformResponse`允许在 then / catch之前对响应数据进行更改
transformResponse: [function (data) {
// Do whatever you want to transform the data
 
return data;
}],
 
// `headers`是要发送的自定义 headers
headers: {'X-Requested-With': 'XMLHttpRequest'},
 
// `params`是要与请求一起发送的URL参数
// 必须是纯对象或URLSearchParams对象
params: {
ID: 12345
},
 
// `paramsSerializer`是一个可选的函数，负责序列化`params`
// (e.g. https://www.npmjs.com/package/qs, http://api.jquery.com/jquery.param/)
paramsSerializer: function(params) {
return Qs.stringify(params, {arrayFormat: 'brackets'})
},
 
// `data`是要作为请求主体发送的数据
// 仅适用于请求方法“PUT”，“POST”和“PATCH”
// 当没有设置`transformRequest`时，必须是以下类型之一：
// - string, plain object, ArrayBuffer, ArrayBufferView, URLSearchParams
// - Browser only: FormData, File, Blob
// - Node only: Stream
data: {
firstName: 'Fred'
},
 
// `timeout`指定请求超时之前的毫秒数。
// 如果请求的时间超过'timeout'，请求将被中止。
timeout: 1000,
 
// `withCredentials`指示是否跨站点访问控制请求
// should be made using credentials
withCredentials: false, // default
 
// `adapter'允许自定义处理请求，这使得测试更容易。
// 返回一个promise并提供一个有效的响应（参见[response docs]（＃response-api））
adapter: function (config) {
/* ... */
},
 
// `auth'表示应该使用 HTTP 基本认证，并提供凭据。
// 这将设置一个`Authorization'头，覆盖任何现有的`Authorization'自定义头，使用`headers`设置。
auth: {
username: 'janedoe',
password: 's00pers3cret'
},
 
// “responseType”表示服务器将响应的数据类型
// 包括 'arraybuffer', 'blob', 'document', 'json', 'text', 'stream'
responseType: 'json', // default
 
//`xsrfCookieName`是要用作 xsrf 令牌的值的cookie的名称
xsrfCookieName: 'XSRF-TOKEN', // default
 
// `xsrfHeaderName`是携带xsrf令牌值的http头的名称
xsrfHeaderName: 'X-XSRF-TOKEN', // default
 
// `onUploadProgress`允许处理上传的进度事件
onUploadProgress: function (progressEvent) {
// 使用本地 progress 事件做任何你想要做的
},
 
// `onDownloadProgress`允许处理下载的进度事件
onDownloadProgress: function (progressEvent) {
// Do whatever you want with the native progress event
},
 
// `maxContentLength`定义允许的http响应内容的最大大小
maxContentLength: 2000,
 
// `validateStatus`定义是否解析或拒绝给定的promise
// HTTP响应状态码。如果`validateStatus`返回`true`（或被设置为`null` promise将被解析;否则，promise将被
  // 拒绝。
validateStatus: function (status) {
return status >= 200 && status < 300; // default
},
 
// `maxRedirects`定义在node.js中要遵循的重定向的最大数量。
// 如果设置为0，则不会遵循重定向。
maxRedirects: 5, // 默认
 
// `httpAgent`和`httpsAgent`用于定义在node.js中分别执行http和https请求时使用的自定义代理。
// 允许配置类似`keepAlive`的选项，
// 默认情况下不启用。
httpAgent: new http.Agent({ keepAlive: true }),
httpsAgent: new https.Agent({ keepAlive: true }),
 
// 'proxy'定义代理服务器的主机名和端口
// `auth`表示HTTP Basic auth应该用于连接到代理，并提供credentials。
// 这将设置一个`Proxy-Authorization` header，覆盖任何使用`headers`设置的现有的`Proxy-Authorization` 自定义 headers。
proxy: {
host: '127.0.0.1',
port: 9000,
auth: : {
username: 'mikeymike',
password: 'rapunz3l'
}
},
 
// “cancelToken”指定可用于取消请求的取消令牌
// (see Cancellation section below for details)
cancelToken: new CancelToken(function (cancel) {
})
}
```

使用 then 时，将收到如下响应

```js
axios.get('/user/12345')
.then(function(response) {
console.log(response.data);
console.log(response.status);
console.log(response.statusText);
console.log(response.headers);
console.log(response.config);
});
```

#### 配置默认值

全局axios默认值

```js
axios.defaults.baseURL = 'https://api.example.com';
axios.defaults.headers.common['Authorization'] = AUTH_TOKEN;
axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded';
```

自定义实例默认值

```js
//在创建实例时设置配置默认值
var instance = axios.create（{
   baseURL：'https://api.example.com'
}）;
//在实例创建后改变默认值
instance.defaults.headers.common ['Authorization'] = AUTH_TOKEN;
```

配置优先级顺序

```js
// 配置将与优先顺序合并。 顺序是lib / defaults.js中的库默认值，然后是实例的defaults属性，最后是请求的config参数。 后者将优先于前者。 这里有一个例子。

//使用库提供的配置默认值创建实例
//此时，超时配置值为`0`，这是库的默认值
var instance = axios.create（）;
 
//覆盖库的超时默认值
//现在所有请求将在超时前等待2.5秒
instance.defaults.timeout = 2500;
 
//覆盖此请求的超时，因为它知道需要很长时间
instance.get（'/ longRequest'，{
   timeout：5000
}）;
```

#### 拦截器

可以截取请求或响应在被 then 或者 catch 处理之前

```js
//添加请求拦截器
axios.interceptors.request.use（function（config）{
     //在发送请求之前做某事
     return config;
   }，function（error）{
     //请求错误时做些事
     return Promise.reject（error）;
   }）;
 
//添加响应拦截器
axios.interceptors.response.use（function（response）{
     //对响应数据做些事
     return response;
   }，function（error）{
     //请求错误时做些事
     return Promise.reject（error）;
   }）;
```

如果你以后可能需要删除拦截器。

```js
var myInterceptor = axios.interceptors.request.use(function () {/*...*/});
axios.interceptors.request.eject(myInterceptor);

```

你可以将拦截器添加到axios的自定义实例

```js
var instance = axios.create();
instance.interceptors.request.use(function () {/*...*/});
```

#### 处理错误

```
axios.get（'/ user / 12345'）
   .catch（function（error）{
     if（error.response）{
       //请求已发出，但服务器使用状态代码进行响应
       //落在2xx的范围之外
       console.log（error.response.data）;
       console.log（error.response.status）;
       console.log（error.response.headers）;
     } else {
       //在设置触发错误的请求时发生了错误
       console.log（'Error'，error.message）;
     }}
     console.log（error.config）;
   }）;
```

您可以使用validateStatus配置选项定义自定义HTTP状态码错误范围。

```js
axios.get（'/ user / 12345'，{
   validateStatus：function（status）{
     return status < 500; //仅当状态代码大于或等于500时拒绝
   }}
}）
```

#### 消除

可以使用取消令牌取消请求

axios cancel token API基于可取消的promise提议，目前处于阶段1，可以使用CancelToken.source工厂创建一个取消令牌，如下所示

```js
var CancelToken = axios.CancelToken;
var source = CancelToken.source（）;
 
axios.get('/user/12345', {
cancelToken: source.token
}).catch(function(thrown) {
if (axios.isCancel(thrown)) {
console.log('Request canceled', thrown.message);
} else {
// 处理错误
}
});
 
//取消请求（消息参数是可选的）
source.cancel（'操作被用户取消。'）;
```

还可以通过将执行器函数传递给CancelToken构造函数来创建取消令牌

```js
var CancelToken = axios.CancelToken;
var cancel;
 
axios.get（'/ user / 12345'，{
   cancelToken：new CancelToken（function executor（c）{
     //一个执行器函数接收一个取消函数作为参数
     cancel = c;
   }）
}）;
 
// 取消请求
clear();
```

注意：您可以使用相同的取消令牌取消几个请求

#### 使用application / x-www-form-urlencoded格式

默认情况下，axios将JavaScript对象序列化为JSON。 要以应用程序/ x-www-form-urlencoded格式发送数据，您可以使用以下选项之一

浏览器中

```js
// 在浏览器中，您可以使用URLSearchParams API
var params = new URLSearchParams();
params.append('param1', 'value1');
params.append('param2', 'value2');
axios.post('/foo', params);
// 请注意，所有浏览器都不支持URLSearchParams，但是有一个polyfill可用（确保polyfill全局环境）。

// 或者，您可以使用qs库对数据进行编码
var qs = require('qs');
axios.post('/foo', qs.stringify({ 'bar': 123 });
```

Node.js

```js
// 在node.js中，可以使用querystring模块
var querystring = require('querystring');
axios.post('http://something.com/', querystring.stringify({ foo: 'bar' });
```

TypeScript

```
// axios包括TypeScript定义
import axios from 'axios';
axios.get('/user?ID=12345');
```

## vuex