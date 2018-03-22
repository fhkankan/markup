# 栈

http://docs.python.org/2/tutorial/datastructures.html#more-on-lists

The list methods make it very easy to use a list as a stack, where the last element added is the first element retrieved (“last-in, first-out”). To add an item to the top of the stack, use `append()`. To retrieve an item from the top of the stack, use `pop()` without an explicit index. For example:

**实现**

```
class Stack: 
    """模拟栈""" 
    def __init__(self): 
        self.items = [] 

    def isEmpty(self): 
        return len(self.items)==0  

    def push(self, item): 
        self.items.append(item) 

    def pop(self): 
        return self.items.pop()  

    def peek(self): 
        if not self.isEmpty(): 
            return self.items[len(self.items)-1] 

    def size(self): 
        return len(self.items) 
```

**操作**

```
Stack()    建立一个空的栈对象
push()     把一个元素添加到栈的最顶层
pop()      删除栈最顶层的元素，并返回这个元素
peek()     返回最顶层的元素，并不删除它
isEmpty()  判断栈是否为空
size()     返回栈中元素的个数
```

**使用**

```
s=Stack() 
print(s.isEmpty()) 
s.push(4) 
s.push('dog') 
print(s.peek()) 
s.push(True) 
print(s.size()) 
print(s.isEmpty()) 
s.push(8.4) 
print(s.pop()) 
print(s.pop()) 
print(s.size()) 
```

# 队

It is also possible to use a list as a queue, where the first element added is the first element retrieved (“first-in, first-out”); however, lists are not efficient for this purpose. While appends and pops from the end of list are fast, doing inserts or pops from the beginning of a list is slow (because all of the other elements have to be shifted by one).

To implement a queue, use [`collections.deque`](https://docs.python.org/3.6/library/collections.html#collections.deque)which was designed to have fast appends and pops from both ends. For example:

```
>>> from collections import deque
>>> queue = deque(["Eric", "John", "Michael"])
>>> queue.append("Terry")           # Terry arrives
>>> queue.append("Graham")          # Graham arrives
>>> queue.popleft()                 # The first to arrive now leaves
'Eric'
>>> queue.popleft()                 # The second to arrive now leaves
'John'
>>> queue                           # Remaining queue in order of arrival
deque(['Michael', 'Terry', 'Graham'])

```

# 单链表

链表的定义：

　　链表(linked list)是由一组被称为结点的数据元素组成的数据结构，每个结点都包含结点本身的信息和指向下一个结点的地址。由于每个结点都包含了可以链接起来的地址信息，所以用一个变量就能够访问整个结点序列。也就是说，结点包含两部分信息：一部分用于存储数据元素的值，称为信息域；另一部分用于存储下一个数据元素地址的指针，称为指针域。链表中的第一个结点的地址存储在一个单独的结点中，称为头结点或首结点。链表中的最后一个结点没有后继元素，其指针域为空。　

![img](https://images0.cnblogs.com/blog/51154/201311/08101605-4dca2917e4164bcfaf78b3625b589b96.jpg)

![img](https://images0.cnblogs.com/blog/51154/201311/08101617-50347cb58527433cb83665ce43c7b338.jpg)

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

class Node(object):
    def __init__(self,val,p=0):
        self.data = val
        self.next = p

class LinkList(object):
    def __init__(self):
        self.head = 0

    def __getitem__(self, key):

        if self.is_empty():
            print 'linklist is empty.'
            return

        elif key <0  or key > self.getlength():
            print 'the given key is error'
            return

        else:
            return self.getitem(key)



    def __setitem__(self, key, value):

        if self.is_empty():
            print 'linklist is empty.'
            return

        elif key <0  or key > self.getlength():
            print 'the given key is error'
            return

        else:
            self.delete(key)
            return self.insert(key)

    def initlist(self,data):

        self.head = Node(data[0])

        p = self.head

        for i in data[1:]:
            node = Node(i)
            p.next = node
            p = p.next

    def getlength(self):

        p =  self.head
        length = 0
        while p!=0:
            length+=1
            p = p.next

        return length

    def is_empty(self):

        if self.getlength() ==0:
            return True
        else:
            return False

    def clear(self):

        self.head = 0


    def append(self,item):

        q = Node(item)
        if self.head ==0:
            self.head = q
        else:
            p = self.head
            while p.next!=0:
                p = p.next
            p.next = q


    def getitem(self,index):

        if self.is_empty():
            print 'Linklist is empty.'
            return
        j = 0
        p = self.head

        while p.next!=0 and j <index:
            p = p.next
            j+=1

        if j ==index:
            return p.data

        else:

            print 'target is not exist!'

    def insert(self,index,item):

        if self.is_empty() or index<0 or index >self.getlength():
            print 'Linklist is empty.'
            return

        if index ==0:
            q = Node(item,self.head)

            self.head = q

        p = self.head
        post  = self.head
        j = 0
        while p.next!=0 and j<index:
            post = p
            p = p.next
            j+=1

        if index ==j:
            q = Node(item,p)
            post.next = q
            q.next = p


    def delete(self,index):

        if self.is_empty() or index<0 or index >self.getlength():
            print 'Linklist is empty.'
            return

        if index ==0:
            q = Node(item,self.head)

            self.head = q

        p = self.head
        post  = self.head
        j = 0
        while p.next!=0 and j<index:
            post = p
            p = p.next
            j+=1

        if index ==j:
            post.next = p.next

    def index(self,value):

        if self.is_empty():
            print 'Linklist is empty.'
            return

        p = self.head
        i = 0
        while p.next!=0 and not p.data ==value:
            p = p.next
            i+=1

        if p.data == value:
            return i
        else:
            return -1


l = LinkList()
l.initlist([1,2,3,4,5])
print l.getitem(4)
l.append(6)
print l.getitem(5)

l.insert(4,40)
print l.getitem(3)
print l.getitem(4)
print l.getitem(5)

l.delete(5)
print l.getitem(5)

l.index(5)
```

# 双链表

和单链表类似，只不过是增加了一个指向前面一个元素的指针而已。

示意图：

![img](https://images0.cnblogs.com/blog/51154/201311/08102601-17d62d106d8449c8ad759d5ad264584f.png)

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

class Node(object):
    def __init__(self,val,p=0):
        self.data = val
        self.next = p
        self.prev = p

class LinkList(object):
    def __init__(self):
        self.head = 0

    def __getitem__(self, key):

        if self.is_empty():
            print 'linklist is empty.'
            return

        elif key <0  or key > self.getlength():
            print 'the given key is error'
            return

        else:
            return self.getitem(key)



    def __setitem__(self, key, value):

        if self.is_empty():
            print 'linklist is empty.'
            return

        elif key <0  or key > self.getlength():
            print 'the given key is error'
            return

        else:
            self.delete(key)
            return self.insert(key)

    def initlist(self,data):

        self.head = Node(data[0])

        p = self.head

        for i in data[1:]:
            node = Node(i)
            p.next = node
            node.prev  = p
            p = p.next

    def getlength(self):

        p =  self.head
        length = 0
        while p!=0:
            length+=1
            p = p.next

        return length

    def is_empty(self):

        if self.getlength() ==0:
            return True
        else:
            return False

    def clear(self):

        self.head = 0


    def append(self,item):

        q = Node(item)
        if self.head ==0:
            self.head = q
        else:
            p = self.head
            while p.next!=0:
                p = p.next
            p.next = q
            q.prev = p


    def getitem(self,index):

        if self.is_empty():
            print 'Linklist is empty.'
            return
        j = 0
        p = self.head

        while p.next!=0 and j <index:
            p = p.next
            j+=1

        if j ==index:
            return p.data

        else:

            print 'target is not exist!'

    def insert(self,index,item):

        if self.is_empty() or index<0 or index >self.getlength():
            print 'Linklist is empty.'
            return

        if index ==0:
            q = Node(item,self.head)

            self.head = q

        p = self.head
        post  = self.head
        j = 0
        while p.next!=0 and j<index:
            post = p
            p = p.next
            j+=1

        if index ==j:
            q = Node(item,p)
            post.next = q
            q.prev = post
            q.next = p
            p.prev = q


    def delete(self,index):

        if self.is_empty() or index<0 or index >self.getlength():
            print 'Linklist is empty.'
            return

        if index ==0:
            q = Node(item,self.head)

            self.head = q

        p = self.head
        post  = self.head
        j = 0
        while p.next!=0 and j<index:
            post = p
            p = p.next
            j+=1

        if index ==j:
            post.next = p.next
            p.next.prev = post

    def index(self,value):

        if self.is_empty():
            print 'Linklist is empty.'
            return

        p = self.head
        i = 0
        while p.next!=0 and not p.data ==value:
            p = p.next
            i+=1

        if p.data == value:
            return i
        else:
            return -1


l = LinkList()
l.initlist([1,2,3,4,5])
print l.getitem(4)
l.append(6)
print l.getitem(5)

l.insert(4,40)
print l.getitem(3)
print l.getitem(4)
print l.getitem(5)

l.delete(5)
print l.getitem(5)

l.index(5)
```

# 二叉树

**树的定义**　　

　　树是一种重要的非线数据结构，直观地看，它是数据元素（在树中称为结点）按分支关系组织起来的结构，很象自然界中的树那样。树结构在客观世界中广泛存在，如人类社会的族谱和各种社会组织机构都可用树形象表示。树在计算机领域中也得到广泛应用，如在编译源程序时，可用树表示源程序的语法结构。又如在数据库系统中，树型结构也是信息的重要组织形式之一。一切具有层次关系的问题都可用树来描述。 

​      树结构的特点是：它的每一个结点都可以有不止一个直接后继，除根结点外的所有结点都有且只有一个直接前驱。

　　树的递归定义如下：（1）至少有一个结点（称为根）（2）其它是互不相交的子树 

 **二叉树**：　

二叉树是由n（n≥0）个结点组成的有限集合、每个结点最多有两个子树的有序树。它或者是空集，或者是由一个根和称为左、右子树的两个不相交的二叉树组成。

特点：

（1）二叉树是有序树，即使只有一个子树，也必须区分左、右子树；

（2）二叉树的每个结点的度不能大于2，只能取0、1、2三者之一；

（3）二叉树中所有结点的形态有5种：空结点、无左右子树的结点、只有左子树的结点、只有右子树的结点和具有左右子树的结点。

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-

class TreeNode(object):
    def __init__(self,data,left,right):
        self.data = data
        self.left = left
        self.right = right


class BTree(object):
    def __init__(self,root=0):
        self.root = root
```

# 二叉树遍历

实现二叉树的三种遍历，先序遍历，中序遍历，后序遍历

```
#!/usr/bin/python
# -*- coding: utf-8 -*-

class TreeNode(object):
    def __init__(self,data=0,left=0,right=0):
        self.data = data
        self.left = left
        self.right = right


class BTree(object):
    def __init__(self,root=0):
        self.root = root

    def is_empty(self):
        if self.root is 0:
            return True
        else:
            return False

    def preOrder(self,treenode):
        if treenode is 0:
            return
        print treenode.data
        self.preOrder(treenode.left)
        self.preOrder(treenode.right)

    def inOrder(self,treenode):
        if treenode is 0:
            return
        self.inOrder(treenode.left)
        print treenode.data
        self.inOrder(treenode.right)

    def postOrder(self,treenode):
        if treenode is 0:
            return
        self.postOrder(treenode.left)
        self.postOrder(treenode.right)
        print treenode.data


n1 = TreeNode(data=1)
n2 = TreeNode(2,n1,0)
n3 = TreeNode(3)
n4 = TreeNode(4)
n5 = TreeNode(5,n3,n4)
n6 = TreeNode(6,n2,n5)
n7 = TreeNode(7,n6,0)
n8 = TreeNode(8)
root = TreeNode('root',n7,n8)

bt = BTree(root)
print 'preOrder......'
print bt.preOrder(bt.root)
print 'inOrder......'
print bt.inOrder(bt.root)
print 'postOrder.....'
print bt.postOrder(bt.root)
```

# 图

python数据结构之图的实现，官方有一篇文章介绍，http://www.python.org/doc/essays/graphs.html

下面简要的介绍下：

比如有这么一张图：

```
    A -> B
    A -> C
    B -> C
    B -> D
    C -> D
    D -> C
    E -> F
    F -> C
```

可以用字典和列表来构建

```
 graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}
```

找到一条路径：

```
def find_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        for node in graph[start]:
            if node not in path:
                newpath = find_path(graph, node, end, path)
                if newpath: return newpath
        return None
```

找到所有路径：

```
def find_all_paths(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not graph.has_key(start):
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths
```

找到最短路径：

```
def find_shortest_path(graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if not graph.has_key(start):
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
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

```
#!/usr/bin/python
# -*- coding: utf-8 -*-

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



