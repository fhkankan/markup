# 搜索

搜索是在队列中找到一个特定元素的算法过程。

常见方法：顺序，二分，二叉树，哈希

# 顺序查找

算法描述

```
从头到尾或者从尾到头的遍历查找
```

# 二分法

算法描述

```
适用于不经常变动而查找频繁的有序列表

（设查找的数组期间为array[low, high]）
（1）确定该期间的中间位置K
（2）将查找的值T与array[k]比较。若相等，查找成功返回此位置；否则确定新的查找区域，继续二分查找。区域确定如下：
a.array[k]>T 由数组的有序性可知array[k,k+1,……,high]>T;故新的区间为array[low,……，K-1]
b.array[k]<T 类似上面查找区间为array[k+1,……，high]。每一次查找与中间值比较，可以确定是否查找成功，不成功当前查找区间缩小一半。递归找，即可。
```

## 递归二分

算法描述

```
函数本身就是对一个队列进行二分比较的功能，而子对垒进行二分比较，也是调用函数自身
```

时间复杂度

```

```

实现

```python
def binary_search(alist, item):
    # 获取列表长度
    n = len(alist)
    # 判断列表是否为空
    if n == 0:
        return False
    # 列表从中间切开
    mid = n // 2
    # 中间值匹配
    if alist[mid] == item:
        return True
    # 左侧二分查找
    elif item < alist[mid]:
        return binary_search(alist[:mid], item)
    # 右侧二分查找
    else:
        return binary_search(alist[mid+1:], item)

def main():
    li = [0,1,2,3,4,5]
    print(binary_search(li, 8))

if __name__ == '__main__':
    main()
```

## 普通二分

算法描述

```
一次二分组，遍历子组找元素
```

时间复杂度

```
O(log2n)
```

实现

```python
def BinarySearch(alist, item):
    # 最低位的索引
    low = 0
    # 最高位的索引
    height = len(alist)-1
    # 只要范围没有缩小到只包含1个元素，继续循环
    while low < height:
        # 判断中间位置
        mid = (low+height)/2
        if array[mid] == item
        	return True
        # 不匹配的话，移动对列的边界
        elif alist[mid] < item:
            low = mid + 1
        else array[mid] > t:
            height = mid - 1
    # 若没有指定元素，返回空
    return None


my_list = [1,3,5,7,9]
result1 = binary_search(my_list,3)
print(result1)
```

# 简单查找

时间复杂度
```
O(n)
```

# 快速查找

时间复杂度
```
O(nlogn)
```
# 二叉树广/深度优先

广度优先

```
查看数据：
从上到下，分层查看，每层从左向右依次查看，直至所有数据查看完毕

添加数据：
从上到下，分层添加，每层从左向右依次添加
```

深度优先

```
首先递归方法看最深的分支元素，再看其他的节点元素

先序遍历：根--->左--->右
中序遍历：左--->根--->右
后序遍历：左--->右--->根
```

实现

```python
class BaseNode(object):
    """定义结点的基本属性"""
    def __init__(self, item):
        # 结点存储的内容
        self.item = item
        # 左侧子节点的索引
        self.lsub = None
        # 右侧子节点的索引
        
class Tree(object):
	"""树类"""
    def __init__(self, root=None):
        self.root = root
    
    # 广度优先
    def breadth_first_search(self):
        # 判断有没有根节点，没有直接退出
        if self.root == None:
            return
        # 创建一个临时元素存放队列，将根节点放入进去
        queue = []
        queue.append(self.root)
        # 判断临时队列不为空，表示还有未查看的父节点
        while len(queue) > 0:
            # 获取父节点并打印信息
            node = queue.pop(0)
            print(node.item,end=" ")
            # 左侧节点不为空，将其放入待处理队列
            if node.lusb:
                qeue.append(node.lsub)
            # 右侧节点不为空，将其放入待处理队列
            if node.rsub:
                queue.append(node.rsub)
        # 修复print功能
        print("")  
    
    # 深度优先
    # 先序遍历
    def pre_travel(self, root):
        # 先对传入节点的内容是否为空进行判断
        if root:
            # 打印根节点内容
            print(root.item, end=" ")
            # 打印左侧节点内容
            self.pre_travel(root.lsub)
            # 打印右侧节点内容
            self.pre_travel(root.rsub)
            
    # 中序遍历
    def in_travel(self, root):
        # 先对传入节点的内容是否为空进行判断
        if root:
            # 打印左侧节点内容
            self.in_travel(root.lsub)
            # 打印根节点内容
            print(root.item, end=" ")
            # 打印右侧节点内容
            self.in_travel(root.rsub)
            
    # 后序遍历
    def aft_travel(self, root):
    	# 先对传入节点的内容是否为空进行判断
        if root:
            # 打印左侧节点内容
            self.aft_travel(root.lsub)
            # 打印右侧节点内容
            self.aft_travel(root.rsub)
            # 打印根节点内容
            print(root.item, end=" ")
        
     
```

# 图深度/广度优先

首先有一个概念：回溯

　　回溯法(探索与回溯法)是一种选优搜索法，按选优条件向前搜索，以达到目标。但当探索到某一步时，发现原先选择并不优或达不到目标，就退回一步重新选择，这种走不通就退回再走的技术为回溯法，而满足回溯条件的某个状态的点称为“回溯点”。

深度优先算法：

（1）访问初始顶点v并标记顶点v已访问。
（2）查找顶点v的第一个邻接顶点w。
（3）若顶点v的邻接顶点w存在，则继续执行；否则回溯到v，再找v的另外一个未访问过的邻接点。
（4）若顶点w尚未被访问，则访问顶点w并标记顶点w为已访问。
（5）继续查找顶点w的下一个邻接顶点wi，如果v取值wi转到步骤（3）。直到连通图中所有顶点全部访问过为止。

广度优先算法：

（1）顶点v入队列。
（2）当队列非空时则继续执行，否则算法结束。
（3）出队列取得队头顶点v；访问顶点v并标记顶点v已被访问。
（4）查找顶点v的第一个邻接顶点col。
（5）若v的邻接顶点col未被访问过的，则col入队列。
（6）继续查找顶点v的另一个新的邻接顶点col，转到步骤（5）。直到顶点v的所有未被访问过的邻接点处理完。转到步骤（2）。

```python
class Graph(object):
    def __init__(self,*args,**kwargs):
        self.node_neighbors = {}
        self.visited = {}

    def add_nodes(self,nodelist):
        for node in nodelist:
            self.add_node(node)

    def add_node(self,node):
        if not node in self.nodes():
            self.node_neighbors[node] = []

    def add_edge(self,edge):
        u,v = edge
        if(v not in self.node_neighbors[u]) and ( u not in self.node_neighbors[v]):
            self.node_neighbors[u].append(v)
            if(u!=v):
                self.node_neighbors[v].append(u)

    def nodes(self):
        return self.node_neighbors.keys()

    def depth_first_search(self,root=None):
        order = []
        def dfs(node):
            self.visited[node] = True
            order.append(node)
            for n in self.node_neighbors[node]:
                if not n in self.visited:
                    dfs(n)

        if root:
            dfs(root)

        for node in self.nodes():
            if not node in self.visited:
                dfs(node)

        print order
        return order

    def breadth_first_search(self,root=None):
        queue = []
        order = []
        def bfs():
            while len(queue)> 0:
                node  = queue.pop(0)

                self.visited[node] = True
                for n in self.node_neighbors[node]:
                    if (not n in self.visited) and (not n in queue):
                        queue.append(n)
                        order.append(n)

        if root:
            queue.append(root)
            order.append(root)
            bfs()

        for node in self.nodes():
            if not node in self.visited:
                queue.append(node)
                order.append(node)
                bfs()
        print order

        return order


if __name__ == '__main__':
    g = Graph()
g.add_nodes([i+1 for i in range(8)])
g.add_edge((1, 2))
g.add_edge((1, 3))
g.add_edge((2, 4))
g.add_edge((2, 5))
g.add_edge((4, 8))
g.add_edge((5, 8))
g.add_edge((3, 6))
g.add_edge((3, 7))
g.add_edge((6, 7))
print "nodes:", g.nodes()

order = g.breadth_first_search(1)
order = g.depth_first_search(1)
```