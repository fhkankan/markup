# 数据结构

- 物理形式：顺序表、链表
- 逻辑形式：集合、线性、树形、图形

# 线性表

概念： 将一大堆同类型的数据，按照某种关系，依次排列出来，形成一条线

特点：数据之间，只有前后关系

分类：顺序表，链表

```python
# 顺序表
将元素顺序地存放在一块连续的存储区里，元素间的顺序关系由它们的存储顺序自然表示。

# 链表
将元素存放在通过链接构造起来的一系列存储块中。表中不仅有自己的数据信息，也有下一个数据节点的地址信息

分类：单向链表，双向链表，单向循环链表
```

# 顺序表

- 形式

```python
# 基本布局
存储同类数据，每个数据占用一个固定存储空间，多个数据组合在一块的时候，物理地址就会连续起来

# 元素外置
存储异类数据，不同类型的数据A-D的占用存储空间不一样，不能按照基本布局来存储，但是他们的逻辑地址编号都是同一的数据类型，所以存储他们的时候，可以单独申请一块连续空间，来存储A-D数据的逻辑地址
```

- 结构

```python
# 顺序表结构
A+B

A：存储数据表的基本信息，申请空间时定义，地址容量+元素个数
地址容量:决定B的荣联大小
元素个数：决定B容量范围内存储数据的个数
B：实际分配的用于存储数据的空间

# 形式
1.一体式：地址容量+元素个数+元素存储空间
存储表信息与元素存储区以连续的方式安排在一块存储区里，形成一个完整的顺序表对象
特点：易于管理，顺序表创建后，元素存储区就固定了
2.分离式：地址容量+元素个数+存储地址->元素存储空间
顺序表对象保存基本信息(地址容量+元素个数)和匀速存储空间的一个逻辑地址(A)，同偶逻辑地址(A)找到真正的数据存储地址
特点：灵活，元素的存储空间可以灵活调整
```

- 内容更改

```
# 一体式
因为一体式，基本信息和存储元素是一个整体的对象，存储内容一旦大于计划容量，若要存储更多内容，只能整体搬迁，即整个顺序表独享(存储顺序表的结果信息的区域)改变了

# 分离式
因为分离式，基本信息和存储元素是两个独立的对象，他们彼此间使用基本信息后面的逻辑地址来连接，所以存储数据变动，只不过是存储的逻辑地址变动，二存储对象中的基本信息没有发生变动，所以，对于顺序表对象来说，没有发生变动
```

- 空间括容

```python
采用分离式结构的顺序表，可以在不改变表对象的前提下，将数据存储区更换为存储空间更大的区域，所有使用这个顺序表的地方都不必修改。
目前动态顺序表的扩容策略是：线性增长、倍数增长

# 线性增长
数据存储区容量一旦发生不足，每次扩容增加固定的存储区容量
特点：
确定性：因为增加数量有限制，所以节省空间
不定性：括重操作可能频繁发生，随时发生

# 倍数增长
数据存储区铜梁一旦发现不足，每次扩容加倍
特点：
确定性：每次增加容量会很大，所以扩充操作的执行次数不会频繁发生
不定性;容量扩孔，可能会造成空间浪费
```

- 常见操作

```python
# 增
插头和插尾，原来的数据顺序没有变化，保序增加
乱插，原来的数据发生变化，非保序增加
时间复杂度：插尾与乱插，O(1);插头：O(n)

# 删
减头和减尾，原来的数据顺序没有变化，保序删除
乱减，原来的数据顺序发生变化，非保序删除
时间复杂度：减尾和乱减，O(1);减头：O(n)
```

# 链表

## 链表VS顺序表

- 结构上

```
链表：灵活，跳转，消耗资源
顺序表：块
```

- 成本上

| 操作            | 链表 | 顺序表 |
| --------------- | ---- | ------ |
| 访问元素        | O(n) | O(1)   |
| 在头部插入/删除 | O(1) | O(n)   |
| 在尾部插入/删除 | O(n) | O(1)   |
| 在中间插入/删除 | O(n) | O(n)   |

##  单向链表

链表的定义：

　　链表(linked list)是由一组被称为结点的数据元素组成的数据结构，每个结点都包含结点本身的信息和指向下一个结点的地址。由于每个结点都包含了可以链接起来的地址信息，所以用一个变量就能够访问整个结点序列。也就是说，结点包含两部分信息：一部分用于存储数据元素的值，称为信息域；另一部分用于存储下一个数据元素地址的指针，称为指针域。链表中的第一个结点的地址存储在一个单独的结点中，称为头结点或首结点。链表中的最后一个结点没有后继元素，其指针域为空。　

![img](https://images0.cnblogs.com/blog/51154/201311/08101605-4dca2917e4164bcfaf78b3625b589b96.jpg)

![img](https://images0.cnblogs.com/blog/51154/201311/08101617-50347cb58527433cb83665ce43c7b338.jpg)

单向链表是链表中最简单的一种形式
```python
结点：元素+下一元素的地址
头结点：单向链表中的第一个结点
尾结点：单向链表中的最后一个结点
前驱结点：链表中间的某个结点的前一个结点
后继结点：链表中的某个结点的后一个结点

单向链表特点：
只要找到头结点，那么整个链表就能找全
尾结点中的"下一地址"内容为空(None)
保存单向链表，只需要保存头结点地址
```

代码实现

```python
class BaseNode(object):
    """单向链表的结点"""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        self.next = None

class SingleLinkedList(object):
    def __init__(self, node=None):
        self.__head = node

    # 判断是否是链表为空
    def is_empty(self):
        return self.__head is None

    # 获取链表的长度
    def length(self):
        # 定义临时变量表示当前结点，指向链表的头结点
        cur = self.__head
        # 设置计数器的初始值为0
        count = 0
        # 查找尾结点
        while cur is not None:
            # 对计数进行递增
            count += 1
            # 移动当前的结点位置
            cur = cur.next
        # 输出最终计数
        return count
    
    # 所有内容查看
    def travel(self):
        # 找到当前链表头信息
        cur = self.__head
        # 查找尾结点
        while cur is not None:
            # 输出每个结点的内容，设置分割符为空格
            print(cur.item, end=" ")
            # 移动当前的结点位置
            cur = cur.next
        # 还原分割符为换行符
        print("")

    # 搜索某个元素
    def search(self, item):
        # 找到当前链表的头信息
        cur = self.__head
        # 判断当前结点不是尾结点
        while cur is not None:
            # 若当前结点的内容就是我们要查找的，就返回True
            if cur.item == item:
                return True
            # 移动当前结点位置到下一结点
            cur = cur.next
        # 如果整个循环都找不到我们要的内容，就返回False
        return False

    # 插头增加
    def add(self, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 指定新结点的next属性为之前链表的头信息
        node.next = self.__head
        # 指定的当前链表的头结点为新结点
        self.__head = node
    
    # 插尾增加
    def append(self, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 如果当前列表为空，那么指定头信息为新结点即可
        if self.is_empty():
            self.__head = node
        # 如果当前列表不为空
        else:
            # 找到当前链表的头信息，然后找到尾结点
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            # 找到尾结点就退出循环，尾结点的next指向新结点即可
            cur.next = node

    # 指定位置增
    def insert(self, pos, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 头部添加内容
        if pos <= 0:
            self.add(item)
        # 尾部添加内容
        elif pos >= self.length():
            self.append(item)
        # 在中间添加元素
        else:
            # 找到头信息，并且开始设置计数器初始值
            cur = self.__head
            count = 0
            # 找到要插入的位置的上一个位置
            while count < (pos - 1):
                count += 1
                cur = cur.next
            # 设置新结点的next属性为当前结点的下一个结点
            node.next = cur.next
            # 设置当前结点的next属性为新接结点
            cur.next = node

    # 删除
    def remove(self, item):
        # 引入头结点
        cur = self.__head
        # 引入前驱结点，默认为None
        pre = None
        # 当前链表不为空
        while cur is not None:
            # 若当前结点的item就是要删除的内容
            if cur.item == item:
                # 若当前结点A就是当前链表的头信息
                if cur == self.__head:
                    self.__head = cur.next
                # 其他位置删除
                else:
                    # 将当前结点B的上一节点A的next属性设定为下一个结点C
                    pre.next = cur.next
                # 确定移出后，退出函数即可
                return
            # 若不是要找的删除元素，移动cur的标签，继续进行判断
            pre = cur
            # 将当前结点A的next属性为下一结点C
            cur = cur.next


def main():
    l1 = SingleLinkedList()
    l1.append(1)
    l1.append(2)
    l1.append(3)
    l1.append(4)
    l1.append(5)
    l1.travel()
    l1.remove(3)
    l1.travel()


if __name__ == '__main__':
    main()
```

## 双链表

双向链表的每个结点都有两个链接地址：

next:指向下一个结点，当此结点为最后一个结点时，指向空值

pre:指向前一个结点，当此结点为第一个节点时，指向空值

示意图：

![img](https://images0.cnblogs.com/blog/51154/201311/08102601-17d62d106d8449c8ad759d5ad264584f.png)

```python
class BaseNode(object):
    """双向链表的结点"""
    def __init__(self, item):
        self.pre = None
        self.item = item
        self.next = None

class DoubleLinkedList(object):
    def __init__(self, node=None):
        self.__head = node

    # 判断是否是链表为空
    def is_empty(self):
        return self.__head is None

    # 获取链表的长度
    def length(self):
        # 找到当前链表的头结点
        cur = self.__head
        # 设置计数器的初始值为0
        count = 0
        # 查找尾结点
        while cur is not None:
            # 对计数进行递增
            count += 1
            # 移动当前的结点位置
            cur = cur.next
        # 输出最终计数
        return count
    
    # 所有内容查看
    def travel(self):
        # 找到当前链表头信息
        cur = self.__head
        # 查找尾结点
        while cur is not None:
            # 输出每个结点的内容，设置分割符为空格
            print(cur.item, end=" ")
            # 移动当前的结点位置
            cur = cur.next
        # 还原分割符为换行符
        print("")

    # 搜索某个元素
    def search(self, item):
        # 找到当前链表的头信息
        cur = self.__head
        # 判断当前结点不是尾结点
        while cur is not None:
            # 若当前结点的内容就是我们要查找的，就返回True
            if cur.item == item:
                return True
            # 移动当前结点位置到下一结点
            cur = cur.next
        # 如果整个循环都找不到我们要的内容，就返回False
        return False

    # 头部增加
    def add(self, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 指定新结点的next属性为之前链表的头信息
        node.next = self.__head
        # 指定当前链表的头信息为新结点
        self.__head = node
        # 若链表不为空
        if node.next:
            # 修改老结点的pre指向新结点
            node.next.pre = node

    # 尾部增加
    def append(self, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 若当前列表为空，那么直接指定头信息为新结点即可
        if self.is_empty():
            self.__head = node
        # 若当前列表不为空
        else:
            # 找到当前连链表的头信息，然后找到尾结点
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            # 指定新结点的pre属性为当前结点
            node.pre = cur
            # 指定当前结点的next属性为新结点
            cur.next = node
          
    # 指定位置增加
    def insert(self, pos, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 头部添加内容
        if pos < 0:
            self.add(item)
        # 尾部添加内容
        elif pos > self.length() - 1:
            self.append(item)
        # 中间添加内容
        else:
            # 找到头信息，并开始设置计数器初始值
            cur = self.__head
            count = 0
            # 找到要插入的位置的上一个位置
            while count < pos:
                count += 1
                cur = cur.next
            # 设置新结点的next属性为当前结点
            node.next = cur
            # 设置新结点的pre属性为当前结点的上一个结点
            node.pre = cur.pre
            # 设置当前结点的上一个结点的next属性为新的结点
            cur.pre.next = node
            # 设置当前结点的pre为新结点
            cur.pre = node

    # 删除
    def remove(self, item):
        # 找到当前链表的头结点
        cur = self.__head
        # 遍历所有结点
        while cur is not None:
            # 若当前结点的item就是要删除的内容
            if cur.item == item:
                # 若当前结点A就是链表的首结点
                if cur == self.__head:
                    # 设置当前链表的头信息为当前结点A的下一个结点B
                    self.__head = cur.next
                    # 链表不是只有一个结点，新的结点的pre属性指向None
                    if cur.next:
                        cur.next.pre = None
                # 若当前结点不是头结点
                else:
                    # 尾结点
                    # 将当前结点B的上一结点A的next属性设定为下一结点C
                    cur.pre.next = cur.next
                    # 中间删
                    if cur.next:
                        cur.next.pre = cur.pre
                return 
            # 将当前结点A的next属性为下一结点C
            cur = cur.next




def main():
    node = DoubleLinkedList()
    node.add(1)
    node.add(2)
    node.add(3)
    node.travel()
    node.insert(2, 4)
    node.travel()

if __name__ == '__main__':
    main()
```

## 单向循环链表

单俩表的一个特殊形式就是单向循环链表，链表中最后一个结点的next不再为None，而是指向当前链表的头结点

```python
class BaseNode(object):
    """单向链表的结点"""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个结点的地址
        self.next = None

class CycleLinkedList(object):
    # 定义操作的接班属性
    def __init__(self, node=None):
        # 确定当前链表的头信息
        self.__head = node

    # 判断是否是链表为空
    def is_empty(self):
        return self.__head is None

    # 判断单向循环链表的长度
    def length(self):
        # 若当前链表是空链表
        if self.is_empty():
            return 0
        # 设计数器的初始值为1,不对尾结点计数
        count = 1
        # 找到当前链表的头位置
        cur = self.__head
        # 查找尾结点
        while cur.next is not self.__head:
            # 对计数进行递增
            count += 1
            # 移动当前的结点位置
            cur = cur.next
        # 输出最终计数
        return count

    # 获取单向循环链表的所有内容
    def travel(self):
        # 若果当前链表是空链表
        if self.is_empty():
            print("")
            return 
        # 找到当前链表的头信息
        cur = self.__head
        # 查找尾结点
        while cur.next is not self.__head:
            # 输出每个结点的内容，设置分隔符为空格
            print(cur.item, end=" ")
            # 移动当前的结点位置
            cur = cur.next
        # 从循环退出，cur指向的是尾结点，所以打印cur的item即可
        print(cur.item)

    # 获取单向循环链表的指定内容
    def search(self, item):
        # 若链表示空链表
        if self.is_empty():
            return False
        # 找到当前链表的头信息
        cur = self.__head
        # 判断当前结点不是尾结点
        while cur.next is not self.__head:
            # 若当前的结点内容就是我们要查找的，就返回True
            if cur.item == item:
                return True
            # 移动当前结点到下一结点
            cur = cur.next
        # 对尾结点进行判断
        if cur.item == item:
            return True
        # 若整个循环都找不到，返回False
        return False
    
    # 头增加
    def add(self, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 对于空链表，我们单独操作
        if self.is_empty():
            self.__head = node
            node.next = self.__head
        # 对于非空链表，定位首结点
        cur = self.__head
        # 查找尾结点，退出循环，表示cur处在尾结点
        while cur.next is not self.__head:
            cur = cur.next
        # 指定当前尾结点的next徐行为新结点
        cur.next = node
        # 指定新结点的next属性为之前链表的头结点
        node.next = self.__head
        # 指定的当前链表的头结点为新结点
        self.__head = node

    # 尾增加
    def append(self, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 如果当前列表为空，那么直接指定头信息为新结点即可
        if self.is_empty():
            self.__head = node
            node.next = self.__head
        # 如果当前列表不为空
        # 找到当前链表的头信息，然后找到尾结点
        cur = self.__head
        while cur.next is not self.__head:
            cur = cur.next
        # 找到尾结点就退出循环，尾结点的next指向新结点即可
        cur.next = node
        # 将新结点的next 指向当前链表的首结点
        node.next = self.__head

    # 指定位置增加
    def insert(self, pos, item):
        # 定义一个存储数据的新结点
        node = BaseNode(item)
        # 头部添加内容
        if pos <= 0:
            self.add(item)
        # 尾部添加元素
        elif pos >= self.length():
            self.append(item)
        # 中间添加元素
        else:
            # 找到头信息，并且设置计数器初始值
            cur = self.__head
            count = 0
            # 找到要插入的位置的上一个位置
            while count < (pos - 1):
                count += 1
                cur = cur.next
            # 设置新结点的next属性为当前结点的下一结点
            node.next = cur.next
            # 设置当前结点的next属性为新结点
            cur.next = node 

    # 删除
    def remove(self, item):
        # 若链表是空链表
        if self.is_empty():
            return 
        # 获取头信息,cur标签用于删除结点
        cur = self.__head
        # pre用来表示当前结点的上一结点
        pre = None
        # 链表不为空情况下
        while cur.next is not self.__head:
            # 若当前结点cur的item就是要删除的内容
            if cur.item == item:
                # 如果头结点
                if cur == self.__head:
                    # tnode用于标识尾结点
                    tnode = self.__head
                    while tnode.next is not self.__head:
                        tnode = tnode.next
                    self.__head = cur.next
                    tnode.next = self.__head
                # 如果当前结点cur不是头结点
                else:
                    # 将当前结点的上一结点的next属性设定为cur的下一个结点C
                    pre.next = cur.next
                return 
            # 若找不到要找的删除元素，将当前结点cur的上一结点作为当前结点
            pre = cur
            # 将当前结点cur的next属性为下一结点C
            cur = cur.next
        # cur是尾结点情况
        if cur.item == item:
            # 链表只有一个元素
            if cur == self.__head:
                self.__head = None
            # 链表不止一个元素
            else:
                pre.next = self.__head
    


def main():
    node = CycleLinkedList()
    node.add(1)
    node.append(2)
    node.insert(1,6)
    node.travel()
    node.remove(6)
    node.travel()

if __name__ == '__main__':
    main()
```

# 栈

栈（stack），是一种数据项按序排列的数据结构，只能在一端对数据项进行插入和删除等操作，简单来说它就是一
个数据的容器。

栈特点：
只能允许在容器的一端进行加入数据和输出数据的运算。
由于栈数据结构只允许在一端进行操作，因而按照 后进先出（LIFO, Last In First Out）的原理运作。

术语：

栈顶、栈底、栈顶元素、栈底元素、入栈、出栈

```python
class Stack: 
    """模拟栈""" 
    def __init__(self): 
        self.items = [] 

    def isEmpty(self): 
        return len(self.items)==0  

    def push(self, item): 
      	"""顶部压入"""
        self.items.append(item) 

    def pop(self): 
      	"""顶部删除"""
        return self.items.pop()  

    def peek(self): 
      	"""返回顶部数据"""
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

# 队列

## 普通队列

队列 是一种数据项按序排列的数据结构，只能在一端对数据项进行插入和删除等操作
队列是一种 先进先出的（First In First Out）的线性表，简称FIFO。允许插入的一端为队尾，允许删除的一端
为队头。队列不允许在中间部位进行操作！

```python
class Queue(object):
    """队列的基本属性"""
    def __init__(self):
        self.__items = []

    def is_empty(self):
        return self.__items == []

    def size(self):
        return len(self.__items)

    def enqueue(self, item):
      	"""给队列添加元素"""
        self.__items.append(item)

    def dequeue(self):
      	"""将队列中元素删除"""
        return self.__items.pop()
```

## 双向队列

双端队列（deque，全名double-ended queue），是一种具有队列和栈的性质的数据结构。
双端队列中的元素可以从两端弹出，其限定插入和删除操作在表的两端进行。双端队列可以在队列任意一端入队和出
队。

```python
class Queue(object):
    """队列的基本属性"""
    def __init__(self):
        self.__items = []

    def is_empty(self):
        return self.__items == []

    def size(self):
        return len(self.__items)

    def enqueue(self, item):
      	"""给队列添加元素"""
        self.__items.append(item)

    def dequeue(self):
      	"""将队列中元素删除"""
        return self.__items.pop()
```

# 树

树是一种数据结构，基本单位是节点。它是由n个节点组成一个具有 层次关系的集合。因为看像一棵 倒挂的树，也就是说它是根朝上，而叶朝下的。

基本特点

```
每个节点有零个或多个子节点；
没有父节点的节点称为根节点；
每一个非根节点有且只有一个父节点；
```
基本术语

```
父节点：含有子节点的节点；
子节点：有父节点的节点
兄弟节点：相同父节点的节点
叶节点或终端节点：没有子节点的结点；
节点的祖先：根节点到该节点之间所有父节点的集合；
节点的度：一个节点含有子节点的个数；
节点的层次：根节点到该节点所经历的结点的个数；
树的度：一棵树中，最大的节点的度称为树的度；
树的高度或深度：树中节点的最大层次；
森林：由m（m>=0）棵互不相交的树的集合称为森林；
```
树的种类

```
无序树：树中任意节点的子节点之间没有顺序关系，这种树称为无序树，也称为自由树；
有序树：树中任意节点的子节点之间有顺序关系，这种树称为有序树；

二叉树：每个节点最多含有两个子树(度小于2)的树称为二叉树；	
```
树的存储

```
顺序存储
将二叉树的所有数据，从上到下分层，每层从做到右，依次编号，然后按照编号顺序把所有元素存储到一个存储空间中，常见形式为列表
父节点和子节点关系：
  	父节点位置 i,找子节点：
  		左子节点位置：2i + 1 右子节点位置：2i + 2
  	左子节点位置 i，找父节点：
  		父节点位置：(i-1)/2
  	右子节点位置i，找父节点：
  		父节点位置：(i-2)/2

  

链式存储
树序表只是保存元素的前后关系，并没有前有的调用关系，而二叉树的重点就是关系，故二叉树采用链式存储。
链表格式：
存储数据+关系索引
```

## 二叉树

二叉树是由n（n≥0）个结点组成的有限集合、每个结点最多有两个子树的有序树。它或者是空集，或者是由一个根和称为左、右子树的两个不相交的二叉树组成。

特点

```
二叉树是有序树，即使只有一个子树，也必须区分左、右子树；
二叉树的每个结点的度不能大于2，只能取0、1、2三者之一；
二叉树中所有结点的形态有5种：空结点、无左右子树的结点、只有左子树的结点、只有右子树的结点和具有左右子树的结点。
```

分类

```
完美：除了叶子节点，都有2个子节点
完全：最后一层子补全，左对齐
完满：除了叶子，其他都有2个子节点
```

- 二叉树增加

```python
class BaseNode(object):
    """定义结点的基本属性"""
    def __init__(self, item):
        # 结点存储的内容
        self.item = item
        # 左侧子节点的索引
        self.lsub = None
        # 右侧子节点的索引
        self.rsub = None

class Tree(object):
    # 定义树结构的基本属性：根节点
    def __init__(self, node=None):
        self.root = node
    
    def add(self, item):
        # 定义要添加到树结构中的节点信息
        node = BaseNode(item)
        # 树为空，对根节点赋值
        if self.root == None:
            self.root = node
        # 根节点有数据
        else:
            # 使用一个临时表来存储我们要处理的元素
            queue = []
            # 把根节点方放到临时队列
            queue.append(self.root)
            # 只要处理队列中有要处理的元素，就一直处理下去
            while queue:
                # 从队列的头部获取要处理的元素
                cur = queue.pop(0)
                # 若左侧子节点为空
                if cur.lsub == None:
                    # 把接收的item放到左侧子节点位置
                    cur.lsub = node
                    return 
                # 在左侧节点有数据，把左侧节点的数据对象追加到待处理父节点队列末尾
                else:
                    queue.append(cur.lsub)
                 # 若右侧子节点为空
                if cur.rsub == None:
                    # 把接收的item放到右侧子节点位置
                    cur.rsub = node
                    return 
                # 在右侧节点有数据，把右侧节点的数据对象追加到待处理父节点队列末尾
                else:
                    queue.append(cur.rsub)      
```

- 二叉树遍历

```python
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
      	"""先序遍历"""
        if treenode is 0:
            return
        print treenode.data
        self.preOrder(treenode.left)
        self.preOrder(treenode.right)

    def inOrder(self,treenode):
      	"""中序遍历"""
        if treenode is 0:
            return
        self.inOrder(treenode.left)
        print treenode.data
        self.inOrder(treenode.right)

    def postOrder(self,treenode):
      	"""后序遍历"""
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

```python
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

```python
 graph = {
     'A': ['B', 'C'],
     'B': ['C', 'D'],
     'C': ['D'],
     'D': ['C'],
     'E': ['F'],
     'F': ['C']
 }
```

找到一条路径：

```python
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

```python
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

```python
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



