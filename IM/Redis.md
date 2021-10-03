# Redis

## 消息队列

Redis通过`list`数据结构来实现消息队列.主要使用到如下命令：
```
- lpush和rpush入队列
- lpop和rpop出队列
- blpop和brpop阻塞式出队列
```
实现

```shell
$redis = new Redis();
$redis->connect('127.0.0.1', 6379);

# 发送消息
$redis->lPush($list, $value);

# 消费消息
while (true) {
    try {
        $msg = $redis->rPop($list);
        if (!$msg) {
            sleep(1);
        }
        //业务处理
     
    } catch (Exception $e) {
        echo $e->getMessage();
    }
}
```

上面代码会有个问题如果队列长时间是空的，那pop就不会不断的循环，这样会导致redis的QPS升高，影响性能。所以我们使用`sleep`来解决，当没有消息的时候阻塞一段时间。但其实这样还会带来另一个问题，就是`sleep`会导致消息的处理延迟增加。这个问题我们可以通过`blpop/brpop` 来阻塞读取队列。

`blpop/brpop`在队列没有数据的时候，会立即进入休眠状态，一旦数据到来，则立刻醒过来。消息的延迟几乎为零。用`blpop/brpop`替代前面的`lpop/rpop`，就完美解决了上面的问题。

还有一个需要注意的点是我们需要是用`try/catch`来进行异常捕获，如果一直阻塞在那里，Redis服务器一般会主动断开掉空链接，来减少闲置资源的占用。

## 延时队列

你是否在做电商项目的时候会遇到如下场景：

- 订单下单后超过一小时用户未支付，需要关闭订单
- 订单的评论如果7天未评价，系统需要自动产生一条评论

这个时候我们就需要用到延时队列了，顾名思义就是需要延迟一段时间后执行。Redis可通过`zset`来实现。我们可以将有序集合的value设置为我们的消息任务，把value的score设置为消息的到期时间，然后轮询获取有序集合的中的到期消息进行处理。

实现

```shell
$redis = new Redis();
$redis->connect('127.0.0.1', 6379);

# 发送消息
$redis->zAdd($delayQueue,$tts, $value);

# 消费消息
while(true) {
    try{
        $msg = $redis->zRangeByScore($delayQueue,0,time(),0,1);
        if($msg){
            continue;
        }
        //删除消息
        $ok = $redis.zrem($delayQueue,$msg);
        if($ok){
            //业务处理
        }
    } catch(Exception $e) {

    }
}
```

这里又产生了一个问题，同一个任务可能会被多个进程取到之后再使用 zrem 进行争抢，那些没抢到的进程都是白取了一次任务，这是浪费。解决办法：将 `zrangebyscore`和`zrem`使用lua脚本进行原子化操作,这样多个进程之间争抢任务时就不会出现这种浪费了。

