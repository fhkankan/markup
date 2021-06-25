# vue锚点定位

## scrollIntoview

```javascript
this.$nextTick(() => {
    // 定位到要锚点的元素
    let cur = document.getElementsByClassName('cur')[0];
    // 进行置顶，默认参数true，false为置底
    if(cur){ cur.scrollIntoView() } 
})
```

## scrollTop

```javascript
this.$nextTick(() => {
	// 定位锚点的元素（列表中的序列）
    if(index==0){
      this.$refs.contentBox.scrollTop = 0;
    }else{
     let obj = this.$refs.contentBox.children;
     let distanceTop = obj[index].offsetTop - obj[index].offsetHeight;
     this.$refs.contentBox.scrollTop = distanceTop;
   }
})
```

