# Vue路由懒加载

[参考](https://blog.csdn.net/xm1037782843/article/details/88225104?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.pc_relevant_is_cache&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.pc_relevant_is_cache)

## vue的异步组件

vue-router配置路由 , 使用vue的异步组件技术 , 可以实现按需加载 . 但是,这种情况下一个组件生成一个js文件

```javascript
{ path: '/home', name: 'home', component: resolve => require(['@/components/home'],resolve) }
{ path: '/index', name: 'Index', component: resolve => require(['@/components/index'],resolve) },
{ path: '/about', name: 'about', component: resolve => require(['@/components/about'],resolve) }
```

懒加载

```javascript
import Vue frim 'vue'
import Router from 'vue-router'

Vue.user(Router)

const Router. = new Router({
    routes:[
        { path: '/home', name: 'home', component: resolve => require(['@/components/home'],resolve) }
    ]
})
```

非懒加载

```javascript
import Vue frim 'vue'
import Router from 'vue-router'
import Home from '@/components/home'

Vue.user(Router)

const Router. = new Router({
    routes:[
        { path: '/home', name: 'home', component: Home},
    ]
})
```

## es的import

```
const 组件名=() => import('组件路径');
```

示例

```javascript
// 下面2行代码，没有指定webpackChunkName，每个组件打包成一个js文件。
const Home = () => import('@/components/home')
const Index = () => import('@/components/index')
const About = () => import('@/components/about') 

// 把组件按组分块，指定了相同的webpackChunkName，会合并打包成一个js文件。
const Home = () => import(/* webpackChunkName: 'ImportFuncDemo' */ '@/components/home')
const Index = () => import(/* webpackChunkName: 'ImportFuncDemo' */ '@/components/index')
const About = () => import(/* webpackChunkName: 'ImportFuncDemo' */ '@/components/about')


const Router. = new Router({
    routes:[
       	{ path: '/home', component: Home },
		{ path: '/index', component: Index }, 
		{ path: '/about', component: About },  
    ]
})
```

## webpack的require

vue-router配置路由，使用webpack的require.ensure技术，也可以实现按需加载。 
这种情况下，多个路由指定相同的chunkName，会合并打包成一个js文件。

```javascript
{ path: '/home', name: 'home', component: r => require.ensure([], () => r(require('@/components/home')), 'demo') },
{ path: '/index', name: 'Index', component: r => require.ensure([], () => r(require('@/components/index')), 'demo') },
{ path: '/about', name: 'about', component: r => require.ensure([], () => r(require('@/components/about')), 'demo-01') }


// r就是resolve
const list = r => require.ensure([], () => r(require('../components/list/list')), 'list');
// 路由也是正常的写法  这种是官方推荐的写的 按模块划分懒加载 
const router = new Router({
    routes: [
        {
           path: '/list/blog',
           component: list,
           name: 'blog'
        }
    ]
})
```

