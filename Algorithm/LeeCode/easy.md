# 01两数之和

给定一个整数数组 `nums` 和一个目标值 `target`，请你在该数组中找出和为目标值的那 **两个** 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

**示例:**

```
给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]
```

解法

```python
# 方法一：暴力，时间复杂度o(n^2),空间复杂度O(1)
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return None

# 方法二：哈希,时间复杂度O(n),空间复杂度O(n)
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        a = {}
        for i in range(len(nums)):
            another = target - nums[i]
            if another in a:
                return [a[another], i]
            a[nums[i]] = i
        return None
```

# 07整数反转

给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

**示例 1:**

```
输入: 123
输出: 321
```

 **示例 2:**

```
输入: -123
输出: -321
```

**示例 3:**

```
输入: 120
输出: 21
```

**注意:**

假设我们的环境只能存储得下 32 位的有符号整数，则其数值范围为 [−231,  231 − 1]。请根据这个假设，如果反转后整数溢出那么就返回 0。

解法

```python
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        ret = []
        res = ''
        ab_x = str(abs(x))
        for i in ab_x:
            ret.insert(0, i)
        for i in ret:
            res += str(i)
        if int(res) > (2**31-1):
            return 0
        if x > 0:
            return int(res)
        else:
            return int(res)*(-1)
```

解法2

```python
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        INT_32_MIN = - 2**31
        INT_32_MAX = 2**31

        sign = 1
        if x < 0:
            sign = -1
            x *= -1

        x_reverse = 0.0
        while x > 0:
            x_reverse = x_reverse * 10 + x % 10
            x = x // 10
            
        x_reverse *= sign
        
        if x_reverse < INT_32_MIN or x_reverse > INT_32_MAX:
            return 0

        return int(x_reverse)
```

# 09回文数

判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

**示例 1:**

```
输入: 121
输出: true
```

**示例 2:**

```
输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
```

**示例 3:**

```
输入: 10
输出: false
解释: 从右向左读, 为 01 。因此它不是一个回文数。
```

解法

```python
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        s = str(x)
        s_list = []
        r_list = []
        for i in s:
            s_list.append(i)
            r_list.insert(0, i)
        if s_list == r_list:
            return True
        else:
            return False
```

解法二

```python
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if (x < 0) or (x % 10 == 0 and x != 0):
            return False
        
        revertedNumber = 0
        # 防止溢出，转换一半
        while x > revertedNumber:
            revertedNumber = revertedNumber*10 + x % 10
            x //= 10
        return x == revertedNumber or x == revertedNumber //10   
```

解法三

```python
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        s = str(x)
        r = s[::-1]
        return  s == r
```

# 13罗马数字转整数

罗马数字包含以下七种字符: `I`， `V`， `X`， `L`，`C`，`D` 和 `M`。

```
字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

例如， 罗马数字 2 写做 `II` ，即为两个并列的 1。12 写做 `XII` ，即为 `X` + `II` 。 27 写做  `XXVII`, 即为 `XX` + `V` + `II` 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 `IIII`，而是 `IV`。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 `IX`。这个特殊的规则只适用于以下六种情况：

- `I` 可以放在 `V` (5) 和 `X` (10) 的左边，来表示 4 和 9。
- `X` 可以放在 `L` (50) 和 `C` (100) 的左边，来表示 40 和 90。 
- `C` 可以放在 `D` (500) 和 `M` (1000) 的左边，来表示 400 和 900。

给定一个罗马数字，将其转换成整数。输入确保在 1 到 3999 的范围内。

**示例 1:**

```
输入: "III"
输出: 3
```

**示例 2:**

```
输入: "IV"
输出: 4
```

**示例 3:**

```
输入: "IX"
输出: 9
```

**示例 4:**

```
输入: "LVIII"
输出: 58
解释: L = 50, V= 5, III = 3.
```

**示例 5:**

```
输入: "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.
```

解法

```python
class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        map_d = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        for i in range(len(s)):
            if i < len(s)-1 and map_d[s[i]] < map_d[s[i+1]]:
                res -= map_d[s[i]]
            else:
                res += map_d[s[i]]
        return res           
```

# 101对称二叉树

给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 `[1,2,2,3,4,4,3]` 是对称的。

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```

但是下面这个 `[1,2,2,null,3,null,3]` 则不是镜像对称的:

```
    1
   / \
  2   2
   \   \
   3    3
```

解法

```python
# 定义二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 递归方法
class Solution:
    def isSymmetric(self, root):
        """
        :type root:TreeNode
        :rtype: bool
        """
        return self.isMirror(root, root)
    
 	def ismirror(self,l_root,r_root):
        if l_root==None and r_root==None:
            return True
        if l_root==None or r_root==None:
            return False
        if l_root.val==r_root.val:
            return self.ismirror(l_root.left,r_root.right) and  self.ismirror(l_root.right,r_root.left)
        return False
    
# 迭代方法
class Solution:
    def isSymmetric(self, root):
        """
        :type root:TreeNode
        :rtype: bool
        """
        qlist=[]
        qlist.append(l_root)
        qlist.append(r_root)
        while len(qlist)!=0:
            t1=qlist.pop()
            t2=qlist.pop()
            if(t1==None and t2==None):
                continue
            if(t1==None or t2==None):
                return False
            if(t1.val!=t2.val):
                return False
            qlist.append(t1.left)
            qlist.append(t2.right)
            qlist.append(t1.right)
            qlist.append(t2.left)
        return True
```



# 104二叉树的最大深度

```python
# 定义二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 递归方法
class Solution:
    def maxDepth(self, root):
        """
        :type root:TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            left_right = self.maxDepth(root.right)
            return max(left_height, right_height) + 1
        
# 迭代方法
class Solution:
    def maxDepth(self, root):
        """
        :type root:TreeNode
        :rtype: int
        """
        stack = []
        if root is not None:
            stack.append((1, root))
        
        depth = 0
        while stack != []:
            current_depth, root = stack.pop()
            if root is not None:
                depth = max(depth, current_depth)
                stack.append((current_depth+1, root.left))
                stack.append((current_depth+1, root.right))
        return depth
```

# 118杨辉三角

给定一个非负整数 *numRows，*生成杨辉三角的前 *numRows* 行。

在杨辉三角中，每个数是它左上方和右上方的数的和。

**示例:**

```
输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```

解法

```python
class Solution:
    def generate(self, num_rows):
        triangle = []

        for row_num in range(num_rows):
            # The first and last row elements are always 1.
            row = [None for _ in range(row_num+1)]
            row[0], row[-1] = 1, 1

            # Each triangle element is equal to the sum of the elements
            # above-and-to-the-left and above-and-to-the-right.
            for j in range(1, len(row)-1):
                row[j] = triangle[row_num-1][j-1] + triangle[row_num-1][j]

            triangle.append(row)

        return triangle
```

# 876转置矩阵

给定一个矩阵 `A`， 返回 `A` 的转置矩阵。

矩阵的转置是指将矩阵的主对角线翻转，交换矩阵的行索引与列索引。

解法

```python
# 解法一：
class Solution(object):
    def transpose(self, A):
        R, C = len(A), len(A[0])
        ans = [[None] * R for _ in range(C)]
        for r, row in enumerate(A):
            for c, val in enumerate(row):
                ans[c][r] = val
        return ans

# 解法二
# python2
class Solution(object):
    def transpose(self, A):
		return zip(*A)
# python3
class Solution(object):
    def transpose(self, A):
        res = []
        for v in zip(*A):
            res.append(v)
        return res
```

# 989数组形式的整数加法

对于非负整数 `X` 而言，*X* 的*数组形式*是每位数字按从左到右的顺序形成的数组。例如，如果 `X = 1231`，那么其数组形式为 `[1,2,3,1]`。

给定非负整数 `X` 的数组形式 `A`，返回整数 `X+K` 的数组形式。

**示例 1：**

```
输入：A = [1,2,0,0], K = 34
输出：[1,2,3,4]
解释：1200 + 34 = 1234
```

**解释 2：**

```
输入：A = [2,7,4], K = 181
输出：[4,5,5]
解释：274 + 181 = 455
```

**示例 3：**

```
输入：A = [2,1,5], K = 806
输出：[1,0,2,1]
解释：215 + 806 = 1021
```

**示例 4：**

```
输入：A = [9,9,9,9,9,9,9,9,9,9], K = 1
输出：[1,0,0,0,0,0,0,0,0,0,0]
解释：9999999999 + 1 = 10000000000
```

 

**提示：**

1. `1 <= A.length <= 10000`
2. `0 <= A[i] <= 9`
3. `0 <= K <= 10000`
4. 如果 `A.length > 1`，那么 `A[0] != 0`

解法

```python
# 错误解法一
class Solution:
    def addToArrayForm(self, A: 'List[int]', K: 'int') -> 'List[int]':
        res = []
        n = len(A)
        sum = 0
        for i, v in enumerate(A):
            sum += v*10**(n-i-1)
        sum += K
        for i in str(sum):
            res.append(int(i))
        return res
    
# 错误解法二
from functools import reduce
class Solution:
    def addToArrayForm(self, A: 'List[int]', K: 'int') -> 'List[int]':
        n = reduce(lambda x, y: x * 10 + y, A) + K
        return (list(map(int, str(n))))
    
# 正确解法
# 防治数据溢出
def demo(A,K):
    res = []
    n = len(A)
    cur = K
    while n>0 or cur >0:
        if n > 0:
            cur += A[n-1]
            n -= 1
        res.insert(0, cur%10)
        cur //= 10
    return res
```

